import os
import math
import torch
import cv2
import h5py
from torch.optim.lr_scheduler import StepLR
from logbook import Logger

from .learning_agent import LearningAgent
from models import get_models
from reporting.progress_bar import ProgressBar
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

NAMESPACE = 'conditional_imitation_agent'  # ++ Identifier name for logging
log = Logger(NAMESPACE)

tag_names = [
    'Steer', 'Gas', 'Brake', 'Hand Brake', 'Reverse Gear', 'Steer Noise',
    'Gas Noise', 'Brake Noise', 'Position X', 'Position Y', 'Speed',
    'Collision Other', 'Collision Pedestrian', 'Collision Car',
    'Opposite Lane Inter', 'Sidewalk Intersect', 'Acceleration X',
    'Acceleration Y', 'Acceleration Z', 'Platform time', 'Game Time',
    'Orientation X', 'Orientation Y', 'Orientation Z', 'Control signal',
    'Noise', 'Camera', 'Angle'
]

image_dir = "demo_imagini"
activations_dir = "demo_activari"
steer_distr_dir = "demo_steer_distr"


def to_cuda(data, use_cuda):
    input_ = data.float()
    if use_cuda:
        input_ = input_.cuda()
    return input_


class ClassificationConditionalImitationAgent(
        LearningAgent):  # ++ Extend Learning agent

    def __init__(self, cfg):
        super(ClassificationConditionalImitationAgent, self).__init__(cfg)

        use_cuda = self._use_cuda  # ++ Parent class already saves some configuration variables
        # ++ All parent variables should start with _.

        # -- Get necessary variables from cfg
        self.cfg = cfg

        # -- Initialize model
        model_class = get_models(cfg.model)

        self.model = model_class[0](cfg, cfg.data_info.image_shape,
                                    cfg.model.nr_bins)
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
        self.scheduler = StepLR(self.optimizer, cfg.train.step_size,
                                cfg.train.decay)
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

        self.img_dir = os.getcwd() + "/" +image_dir
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

        super(ClassificationConditionalImitationAgent, self).__end_init__()

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
        branches = self.model.get_branches(use_cuda)
        train_loss = 0

        progress_bar = ProgressBar(
            'Loss: %(loss).3f', dict(loss=0), len(data_loader))

        for batch_idx, (images, speed, steer_distr, target_vector,
                        mask) in enumerate(data_loader):

            optimizer.zero_grad()
            images = to_cuda(images, use_cuda)
            speed_target = to_cuda(speed.unsqueeze(1), use_cuda)
            target_vector = to_cuda(target_vector, use_cuda)
            steer_distr = to_cuda(steer_distr, use_cuda)

            inter_output, speed_output, _ = model(images, speed_target)

            output = to_cuda(torch.zeros((mask.shape[0], 182)), use_cuda)

            for i in range(1, len(branches)):
                filter_ = (mask == (i + 1))
                output[filter_, :] = branches[i](inter_output[filter_, :])

            loss = criterion(output, speed_output, target_vector, speed_target,
                             steer_distr)

            loss.backward()
            train_loss += loss.item()

            optimizer.step()
            scheduler.step()
            progress_bar.update(
                batch_idx, dict(loss=(train_loss / (batch_idx + 1))))

            self.loss_values_train.append(loss.item())

            ################### TensorBoard Shit ################################

            #loss function

            #self._writer.add_scalar(
            #    "loss_function", loss.item(),
            #    batch_idx + self._train_epoch * len(data_loader))

            #model
            #if self._tensorboard_model is False:
            #    self._tensorboard_model = True
            #    self._writer.add_graph(model, (images, speed_target))

            #####################################################################

        progress_bar.finish()

        return train_loss, {}

    def _get_criterion(self, branch_outputs, speed_outputs, branch_target,
                       speed_target, steer_distr):

        #get lob_probabilities for outputs
        #log_outputs = torch.nn.functional.log_softmax(branch_outputs[:, :-2], dim=1)

        loss1_steer = torch.nn.functional.mse_loss(
            branch_outputs[:, :-2], steer_distr, size_average=False)

        loss1_others = (branch_target - branch_outputs[:, -2:]) * (
            branch_target - branch_outputs[:, -2:])
        loss1_others = loss1_others.sum(dim=0)

        loss_weights = self.cfg.train.loss_weights

        loss1 = loss_weights[0] * loss1_steer + (
            loss_weights[1] * loss1_others[0] +
            loss_weights[2] * loss1_others[1])

        #loss1 = 0.45 * loss1[0] + 0.45 * loss1[1] + 0.1 * loss1[2]
        #loss1 = 0.55 * loss1[0] + 0.35 * loss1[1] + 0.1 * loss1[2]
        #loss1 = 0.65 * loss1[0] + 0.30 * loss1[1] + 0.05 * loss1[2]

        loss2 = (speed_outputs - speed_target) * (speed_outputs - speed_target)
        loss2 = loss2.sum() / branch_outputs.shape[0]

        loss = (0.95 * loss1 + 0.05 * loss2) / branch_outputs.shape[0]
        return loss

    def _test(self, data_loader):
        """
        Considering a dataloader (loaded from config.)
        Implement the testing loop.
        """
        use_cuda = self._use_cuda
        model = self.model
        criterion = self._get_criterion
        branches = self.model.get_branches(use_cuda)
        test_loss = 0

        progress_bar = ProgressBar(
            'Loss: %(loss).3f', dict(loss=0), len(data_loader))

        for batch_idx, (images, speed, steer_distr, target_vector,
                        mask) in enumerate(data_loader):

            images = to_cuda(images, use_cuda)
            speed_target = to_cuda(speed.unsqueeze(1), use_cuda)
            target_vector = to_cuda(target_vector, use_cuda)
            steer_distr = to_cuda(steer_distr, use_cuda)

            inter_output, speed_output, _ = model(images, speed_target)

            output = to_cuda(torch.zeros((mask.shape[0], 182)), use_cuda)

            for i in range(1, len(branches)):
                filter_ = (mask == (i + 1))
                output[filter_, :] = branches[i](inter_output[filter_, :])

            loss = criterion(output, speed_output, target_vector, speed_target,
                             steer_distr)

            test_loss += loss.item()

            self.loss_values_test.append(loss.item())

            progress_bar.update(
                batch_idx, dict(loss=(test_loss / (batch_idx + 1))))

        progress_bar.finish()

        return test_loss, None, {}

    def _get_steer_from_bins(self, steer_vector):

        #pass the steer values through softmax_layer and get the bin index
        bin_index = torch.nn.functional.softmax(steer_vector).argmax()
        #bin_index = steer_vector.argmax()
        plt.plot(self._bins + 1.0 / len(self._bins),
                 torch.nn.functional.softmax(steer_vector).data[0].numpy())
        plt.show(block=False)

        plt.draw()
        plt.pause(0.0001)
        #plt.savefig(self.steer_dir + "/distr_" + str(self.nr_img) + ".png")
        plt.gcf().clear()
        #get steer_value from bin
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
        speed = to_cuda(torch.Tensor([speed / 90.0]), self._use_cuda)
        speed = speed.unsqueeze(0)

        branches = self.model.get_branches(self._use_cuda)

        inter_output, speed_output, activation_map = self.model(image, speed)

        if cmd == 2 or cmd == 0:
            output = branches[1](inter_output)
        elif cmd == 3:
            output = branches[2](inter_output)
        elif cmd == 4:
            output = branches[3](inter_output)
        else:
            output = branches[4](inter_output)

        steer_angle = self._get_steer_from_bins(output[:, :-2])
        output = output.data.cpu()[0].numpy()
        speed_output = speed_output.data.cpu()[0].numpy()
        return steer_angle, output[-2], output[-1], speed_output[
            0] * 90, activation_map

    def run_1step(self, image_raw, speed, cmd):

        image = np.transpose(image_raw, (2, 0, 1)).astype(np.float32)
        image = np.multiply(image, 1.0 / 127.5) - 1
        image = to_cuda(torch.from_numpy(image), self._use_cuda)
        image = image.unsqueeze(0)
        speed = to_cuda(torch.Tensor([speed / 90.0]), self._use_cuda)
        speed = speed.unsqueeze(0)

        branches = self.model.get_branches(self._use_cuda)

        inter_output, speed_output, activation_map = self.model(image, speed)

        if self.cfg.activations:
            self._show_activation_image(activation_map, np.copy(image_raw))

        if cmd == 2 or cmd == 0:
            output = branches[1](inter_output)
        elif cmd == 3:
            output = branches[2](inter_output)
        elif cmd == 4:
            output = branches[3](inter_output)
        else:
            output = branches[4](inter_output)

        steer_angle = self._get_steer_from_bins(output[:, :-2])
        output = output.data.cpu()[0].numpy()
        speed_output = speed_output.data.cpu()[0].numpy()
        return steer_angle, output[-2], output[-1], speed_output[0] * 90

    def _eval_episode(self, file_name):

        openedFile = h5py.File(file_name, 'r')

        images = openedFile["rgb"]
        ground_truth = openedFile["targets"]

        nr_images = images.shape[0]
        previous_speed = ground_truth[0][tag_names.index('Speed')]

        general_mse = steer_mse = gas_mse = break_mse = 0
        loss_weights = self.cfg.train.loss_weights

        for index in range(nr_images):

            image_gt = ground_truth[index]
            gt_steer = image_gt[tag_names.index('Steer')]
            gt_brake = image_gt[tag_names.index('Brake')]
            gt_gas = image_gt[tag_names.index('Gas')]
            gt_speed = image_gt[tag_names.index('Speed')]

            predicted_steer, predicted_gas, predicted_brake, predicted_speed = self.run_1step(
                images[index], previous_speed, image_gt[24])

            steer = (predicted_steer - gt_steer) * (predicted_steer - gt_steer)
            gas = (predicted_gas - gt_gas) * (predicted_gas - gt_gas)
            brk = (predicted_brake - gt_brake) * (predicted_brake - gt_brake)
            speed = (predicted_speed - gt_speed) * (predicted_speed - gt_speed)
            break_mse += brk
            gas_mse += gas
            steer_mse += steer

            #general_mse += (0.05 * speed + 0.95 *
            #                (0.45 * steer + 0.45 * gas + 0.1 * brk))

            general_mse += (0.05 * speed + 0.95 *
                            (loss_weights[0] * steer + loss_weights[1] * gas +
                             loss_weights[2] * brk))

            #general_mse += (0.05 * speed + 0.95 *
            #                (0.55 * steer + 0.35 * gas + 0.1 * brk))

            log.info("Frame number {}".format(index))
            log.info("Steer: predicted {},  ground_truth {}".format(
                predicted_steer, gt_steer))
            log.info("Break: predicted {},  ground_truth {}".format(
                predicted_brake, gt_brake))
            log.info("Gas: predicted {}, ground_truth {}".format(
                predicted_gas, gt_gas))

            log.info("Speed: predicted {}, ground_truth {}".format(
                predicted_speed, gt_speed))

            previous_speed = gt_speed

        break_mse /= float(nr_images)
        general_mse /= float(nr_images)
        steer_mse /= float(nr_images)
        gas_mse /= float(nr_images)

        return general_mse, gas_mse, steer_mse, break_mse

    def eval_agent(self):

        self.set_eval_mode()

        f = open(self._save_path + "/eval_results.txt", "wt")
        current_dir = os.getcwd()
        os.chdir(self.cfg.dataset.dataset_eval_path)
        h5files = sorted(os.listdir())
        eval_results = []

        mean_mse = mean_gas = mean_steer = mean_break = 0
        for hd5file in h5files:
            general_mse, gas_mse, steer_mse, break_mse = self._eval_episode(
                hd5file)
            eval_results.append((general_mse, gas_mse, steer_mse, break_mse))
            mean_mse += general_mse
            mean_gas += gas_mse
            mean_steer += steer_mse
            mean_break += break_mse

            f.write(
                "****************Evaluated {} *******************\n".format(
                    hd5file))
            f.write("Mean squared error is {}\n".format(str(general_mse)))
            f.write("Mean squared error for steering is {}\n".format(
                str(steer_mse)))
            f.write("Mean squared error for break is {}\n".format(
                str(break_mse)))
            f.write("Mean squared error for gas/acceleration is {}\n".format(
                str(gas_mse)))
            f.write("************************************************\n\n")
            f.flush()

        mean_mse /= float(len(h5files))
        mean_gas /= float(len(h5files))
        mean_steer /= float(len(h5files))
        mean_break /= float(len(h5files))

        std_mse = std_gas = std_steer = std_break = 0
        for i in range(len(h5files)):
            std_mse += (eval_results[i][0] - mean_mse) * (
                eval_results[i][0] - mean_mse)
            std_steer += (eval_results[i][2] - mean_steer) * (
                eval_results[i][2] - mean_steer)
            std_gas += (eval_results[i][1] - mean_gas) * (
                eval_results[i][1] - mean_gas)
            std_break += (eval_results[i][3] - mean_break) * (
                eval_results[i][3] - mean_break)

        std_mse /= float(len(h5files))
        std_gas /= float(len(h5files))
        std_steer /= float(len(h5files))
        std_break /= float(len(h5files))

        std_mse = math.sqrt(std_mse)
        std_gas = math.sqrt(std_gas)
        std_steer = math.sqrt(std_steer)
        std_break = math.sqrt(std_break)

        f.write("****************Final Evaluation *******************\n")
        f.write("Mean squared error is {} with standard deviation {}\n".format(
            str(mean_mse), str(std_mse)))
        f.write(
            "Mean squared error for steering is {} with standard deviation {}\n".
            format(str(steer_mse), str(std_steer)))
        f.write(
            "Mean squared error for break is {} with standard deviation {}\n".
            format(str(break_mse), str(std_break)))
        f.write(
            "Mean squared error for gas/acceleration is {} with standard deviation {}\n".
            format(str(gas_mse), str(std_gas)))
        f.write("******************************************************")
        f.flush()
        f.close()

        os.chdir(current_dir)

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

        branches = self.model.get_branches(self._use_cuda)

        inter_output, predicted_speed, activation_map = self.model(
            image_input, speed)

        if self.cfg.activations:
            self._show_activation_image(activation_map,
                                        np.copy(image_input_raw))

        if control_input == 2 or control_input == 0:
            output = branches[1](inter_output)
        elif control_input == 3:
            output = branches[2](inter_output)
        elif control_input == 4:
            output = branches[3](inter_output)
        else:
            output = branches[4](inter_output)

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
