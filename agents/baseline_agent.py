import os
import cv2
import sys
import math
import torch
import pandas as pd
from logbook import Logger

from .learning_agent import LearningAgent
from models import get_models
from reporting.progress_bar import ProgressBar
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

NAMESPACE = 'baseline_agent'  # ++ Identifier name for logging
log = Logger(NAMESPACE)

def to_cuda(data, use_cuda):
    input_ = data.float()
    if use_cuda:
        input_ = input_.cuda()
    return input_

image_dir = "demo_imagini"
activations_dir = "demo_activari"
steer_distr_dir = "demo_steer_distr"

class BaselineAgent(LearningAgent):

    def __init__(self, cfg):
        super(BaselineAgent, self).__init__(cfg)

        use_cuda = self._use_cuda  # ++ Parent class already saves some configuration variables
        # ++ All parent variables should start with _.

        # -- Get necessary variables from cfg
        self.cfg = cfg

        # -- Initialize model
        model_class = get_models(cfg.model)

        input_shape = cfg.data_info.image_shape
        input_shape[0] *= cfg.data_info.frame_seq_len
        self.model = model_class[0](cfg, input_shape, cfg.model.nr_bins)
        # ++ All models receive as parameters (configuration namespace, input data size,
        # ++ output data size)

        self._models.append(
            self.model)  # -- Add models & optimizers to base for saving

        # ++ After adding model you can set the agent to cuda mode
        # ++ Parent class already makes some adjustments. E.g. turns model to cuda mode
        if use_cuda:
            self.cuda()

        self._bins = np.arange(-1.0, 1.0, 2.0 / cfg.model.nr_bins)
        # -- Initialize optimizers
        self.optimizer = self.get_optim(cfg.train.algorithm,
                                        cfg.train.algorithm_args, self.model)
        self.scheduler = self.get_sched(cfg.train.scheduler,
                                        cfg.train.scheduler_args, self.optimizer)
        self._optimizers.append(
            self.optimizer)  # -- Add models & optimizers to base for saving
        # -- Change settings from parent class
        # ++ Parent class automatically initializes 4 metrics: loss/acc for train/test
        # ++ E.g switch metric slope
        self.set_eval_metric_comparison(True)

        # ++ E.g. to add variable name to be saved at checkpoints
        self._save_data.append("scheduler")

        self._tensorboard_model = False
        self.loss_values_train = []
        self.loss_values_test = []

        ##### Make directories and shit for demo########

        self.img_dir = os.getcwd() + "/" + image_dir
        self.act_dir = os.getcwd() + "/" + activations_dir
        self.steer_dir = os.getcwd() + "/" + steer_distr_dir

        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)
        if not os.path.isdir(self.act_dir):
            os.mkdir(self.act_dir)
        if not os.path.isdir(self.steer_dir):
            os.mkdir(self.steer_dir)
        self.nr_img = 0

        ################################################

        super(BaselineAgent, self).__end_init__()

    def _session_init(self):
        if self._is_train:
            self.optimizer.zero_grad()

    def _train(self, data_loader):
        """
        Considering a dataloader (loaded from config.)
        Implement the training loop.
        :return training loss metric & other information
        """
        optimizer = self.optimizer
        scheduler = self.scheduler
        use_cuda = self._use_cuda
        model = self.model
        criterion = self._get_criterion
        # Remember that the first branch is the speed branch
        branches = self.model.get_branches(use_cuda)
        train_loss = 0

        if self.cfg.use_progress_bar:
            progress_bar = ProgressBar(
                'Loss: %(loss).5f', dict(loss=0), len(data_loader))

        for batch_idx, (images, speed, steer_distr, mask) in enumerate(data_loader):
            optimizer.zero_grad()
            images = to_cuda(images, use_cuda)
            steer_distr = to_cuda(steer_distr, use_cuda)
            steer_output = model(images)

            steer_output = to_cuda(steer_output, use_cuda)

            loss = criterion(steer_output, steer_distr)

            # Check if loss is nan
            if math.isnan(loss):
                print("Loss became NaN")
                sys.exit(1)

            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            if self.cfg.train.scheduler == 'ReduceLROnPlateau':
                scheduler.step(loss)
            else:
                scheduler.step()

            if self.cfg.use_progress_bar:
                progress_bar.update(
                    batch_idx, dict(loss=(train_loss / (batch_idx + 1))))

            self.loss_values_train.append(loss.item())

        if self.cfg.use_progress_bar:
            progress_bar.finish()

        return train_loss, {}

        def _get_criterion(self, steer_output, steer_distr):
            loss_steer = torch.nn.functional.mse_loss(
                steer_output, steer_distr, reduction='sum') / branch_outputs.shape[0]
            return loss_steer

        def _test(self, data_loader):
            """
            Considering a dataloader (loaded from config.)
            Implement the testing loop.
            """
            use_cuda = self._use_cuda
            model = self.model
            criterion = self._get_criterion
            test_loss = 0

            if self.cfg.use_progress_bar:
                progress_bar = ProgressBar(
                    'Loss: %(loss).5f', dict(loss=0), len(data_loader))

            for batch_idx, (images, speed, steer_distr, mask) in enumerate(data_loader):
                images = to_cuda(images, use_cuda)
                steer_distr = to_cuda(steer_distr, use_cuda)

                steer_output = model(images)
                steer_output = to_cuda(steer_output, use_cuda)

                loss = criterion(steer_output, steer_distr)

                test_loss += loss.item()

                self.loss_values_test.append(loss.item())

                if self.cfg.use_progress_bar:
                    progress_bar.update(
                        batch_idx, dict(loss=(test_loss / (batch_idx + 1))))

            if self.cfg.use_progress_bar:
                progress_bar.finish()

            return test_loss, None, {}

        def _get_steer_from_bins(self, steer_vector):
            # Pass the steer values through softmax_layer and get the bin index
            bin_index = torch.nn.functional.softmax(steer_vector).argmax()
            # plt.plot(self._bins + 1.0 / len(self._bins),
            #          torch.nn.functional.softmax(steer_vector).data.cpu()[0].numpy())
            # plt.show(block=False)

            # plt.draw()
            # plt.pause(0.0001)
            # plt.gcf().clear()

            # Get steer_value from bin
            return self._bins[bin_index] + 1.0 / len(self._bins)

        def _show_activation_image(self, raw_activation, image_activation):
            activation_map = raw_activation.data[0, 0].cpu().numpy()
            activation_map = (activation_map - np.min(activation_map)
                              ) / np.max(activation_map) - np.min(activation_map)

            activation_map = (activation_map * 255.0)

            if image_activation.shape[0] != activation_map.shape[0]:
                activation_map = scipy.misc.imresize(
                    activation_map,
                    [image_activation.shape[0], image_activation.shape[1]])

            image_activation[:, :, 1] += activation_map.astype(np.uint8)
            activation_map = cv2.applyColorMap(
                activation_map.astype(np.uint8), cv2.COLORMAP_JET)

            image_activation = cv2.resize(image_activation, (720, 460),
                                          cv2.INTER_AREA)
            image_activation = cv2.cvtColor(image_activation, cv2.COLOR_RGB2BGR)

            activation_map = cv2.resize(activation_map, (720, 460), cv2.INTER_AREA)

            cv2.imshow("activation",
                       np.concatenate((image_activation, activation_map), axis=1))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

        def run_image(self, image_raw, speed, cmd):
            self.set_eval_mode()

            image = np.transpose(image_raw, (2, 0, 1)).astype(np.float32)
            image = np.multiply(image, 1.0 / 127.5) - 1
            image = to_cuda(torch.from_numpy(image), self._use_cuda)
            image = image.unsqueeze(0)
            speed = to_cuda(torch.Tensor([speed / 80.0]), self._use_cuda)
            speed = speed.unsqueeze(0)
            # Remember that the first branch is the speed branch
            branches = self.model.get_branches(self._use_cuda)

            inter_output, speed_output, activation_map = self.model(image, speed)

            output = branches[cmd + 1](inter_output)

            steer_angle = self._get_steer_from_bins(output)
            speed_output = speed_output.data.cpu()[0].numpy()
            return steer_angle, speed_output[0], activation_map

        def run_1step(self, image_raw, speed, cmd):
            image = np.transpose(image_raw, (2, 0, 1)).astype(np.float32)
            image = np.multiply(image, 1.0 / 127.5) - 1
            image = to_cuda(torch.from_numpy(image), self._use_cuda)
            image = image.unsqueeze(0)
            speed = to_cuda(torch.Tensor([speed / 80.0]), self._use_cuda)
            speed = speed.unsqueeze(0)

            # Remember that the first branch is the speed branch
            branches = self.model.get_branches(self._use_cuda)

            inter_output, speed_output, activation_map = self.model(image, speed)

            if self.cfg.activations:
                self._show_activation_image(activation_map, np.copy(image_raw))

            output = branches[cmd + 1](inter_output)
            steer_angle = self._get_steer_from_bins(output)
            speed_output = speed_output.data.cpu()[0].numpy()
            return steer_angle, speed_output[0]

        def _eval_episode(self, file_name):
            video_file = file_name[0]
            info_file = file_name[1]
            info = pd.read_csv(info_file)

            nr_images = len(info)
            previous_speed = info['linear_speed'][0] / 80

            general_mse = steer_mse = 0

            # Get steering angles and commands
            linear_speed = info['linear_speed']
            angles = info['steer_angle']
            cmds = np.array(info['steer'], dtype=np.int)

            # Iterate throgh all video frames
            for index in range(nr_images):
                frame_name = '{}.jpg'.format(index)
                file = os.path.join(video_file, frame_name)
                frame = cv2.imread(file)

                # Normalize speed and steer ground truths
                gt_speed = linear_speed[index] / 80
                gt_steer = angles[index] * 2 / math.pi
                gt_steer = min(1, max(-1, gt_steer))

                predicted_steer, predicted_speed = self.run_1step(
                    frame, previous_speed, cmds[index])

                steer = (predicted_steer - gt_steer) ** 2
                speed = (predicted_speed - gt_speed) ** 2
                steer_mse += steer

                general_mse += 0.05 * speed + 0.95 * steer

                # log.info("Frame number {}".format(index))
                # log.info("Steer: predicted {}, ground_truth {}".format(
                #    predicted_steer, gt_steer))

                # log.info("Speed: predicted {}, ground_truth {}".format(
                #    predicted_speed * 80, gt_speed * 80))

                previous_speed = gt_speed

            general_mse /= nr_images
            steer_mse /= nr_images

            return general_mse, steer_mse

        def eval_agent(self):
            self.set_eval_mode()

            f = open(self._save_path + "/eval_results.txt", "wt")
            data_files = sorted(os.listdir(self.cfg.dataset.dataset_test_path))
            video_files = []
            for file in data_files:
                info_file = file + '.csv'
                video_files.append((os.path.join(self.cfg.dataset.dataset_test_path, file),
                                    os.path.join(self.cfg.dataset.info_test_path, info_file)))
            eval_results = []

            mean_mse = mean_steer = 0
            for video_file in video_files:
                general_mse, steer_mse = self._eval_episode(video_file)
                eval_results.append((general_mse, steer_mse))
                mean_mse += general_mse
                mean_steer += steer_mse

                f.write(
                    "****************Evaluated {} *******************\n".format(
                        video_file))
                f.write("Mean squared error is {}\n".format(str(general_mse)))
                f.write("Mean squared error for steering is {}\n".format(
                    str(steer_mse)))
                f.write("************************************************\n\n")
                f.flush()

            mean_mse /= float(len(video_files))
            mean_steer /= float(len(video_files))

            std_mse = std_steer = 0
            for i in range(len(video_files)):
                std_mse += (eval_results[i][0] - mean_mse) * (
                        eval_results[i][0] - mean_mse)
                std_steer += (eval_results[i][1] - mean_steer) * (
                        eval_results[i][1] - mean_steer)

            std_mse /= float(len(video_files))
            std_steer /= float(len(video_files))

            std_mse = math.sqrt(std_mse)
            std_steer = math.sqrt(std_steer)

            f.write("****************Final Evaluation *******************\n")
            f.write("Mean squared error is {} with standard deviation {}\n".format(
                str(mean_mse), str(std_mse)))
            f.write(
                "Mean squared error for steering is {} with standard deviation {}\n".
                    format(str(steer_mse), str(std_steer)))
            f.write("******************************************************")
            f.flush()
            f.close()

        def _control_function(self, image_input_raw, real_speed, control_input):
            """
            Implement for carla simulator run.
            :return: steer, acc, brake
            """
            print("Control input is {}".format(control_input))

            image_input = scipy.misc.imresize(image_input_raw, [
                self.cfg.data_info.image_shape[1],
                self.cfg.data_info.image_shape[2]
            ])
            image_input = np.transpose(image_input, (2, 0, 1)).astype(np.float32)
            image_input = np.multiply(image_input, 1.0 / 127.5) - 1.0
            image_input = torch.from_numpy(image_input)
            image_input = image_input.unsqueeze(0)
            speed = torch.Tensor([real_speed / 25.0])
            speed = speed.unsqueeze(0)

            # Remember that the first branch is the speed branch
            branches = self.model.get_branches(self._use_cuda)

            inter_output, predicted_speed, activation_map = self.model(
                image_input, speed)

            if self.cfg.activations:
                self._show_activation_image(activation_map,
                                            np.copy(image_input_raw))

            output = branches[control_input + 1](inter_output)

            steer = self._get_steer_from_bins(output[:, :-2])
            output = output.data.cpu()[0].numpy()
            acc, brake = output[-2], output[-1]

            predicted_speed = predicted_speed.data[0].numpy()
            real_predicted = predicted_speed * 25.0

            if real_speed < 2.0 and real_predicted > 3.0:
                acc = 1 * (5.6 / 25.0 - real_speed / 25.0) + acc
                brake = 0.0

            self.nr_img += 1
            return steer, acc, brake

        def _set_eval_mode(self):
            """
            Custom configuration when changing to evaluation mode
            """
            if self.cfg.activations:
                self.model.set_forward('forward_deconv')
            else:
                self.model.set_forward('forward_simple')
            if self._use_cuda:
                self.cuda()

        def _set_train_mode(self):
            """
            Custom configuration when changing to train mode
            """
            self.model.set_forward('forward_simple')
            if self._use_cuda:
                self.cuda()

        def _save(self, save_data, path):
            """
            Called when saving agent state. Agent already saves variables defined in the list
            self._save_data and other default options.
            :param save_data: Pre-loaded dictionary with saved data. Append here other data
            :param path: Path to folder where other custom data can be saved
            :return: should return default save_data dictionary to be saved
            """
            save_data['scheduler_state'] = self.scheduler.state_dict()
            save_data['train_epoch'] = self._train_epoch
            save_data['loss_value_train'] = self.loss_values_train
            save_data['loss_value_test'] = self.loss_values_test

            return save_data

        def _resume(self, agent_check_point_path, saved_data):
            """
            Custom resume scripts should pe implemented here
            :param agent_check_point_path: Path of the checkpoint resumed
            :param saved_data: loaded checkpoint data (dictionary of variables)
            """
            self.scheduler.load_state_dict(saved_data['scheduler_state'])
            self.scheduler.optimizer = self.optimizer
            self.model = self._models[0]
            self.optimizer = self._optimizers[0]
            self._train_epoch = saved_data['train_epoch']
            self.loss_values_train = saved_data['loss_value_train']
            self.loss_values_test = saved_data['loss_value_test']
            if not self._use_cuda:
                self.model.cpu()