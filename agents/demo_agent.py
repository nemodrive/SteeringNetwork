import torch
from torch.autograd import Variable
from logbook import Logger

from .learning_agent import LearningAgent
from models import get_models
from reporting.progress_bar import ProgressBar


NAMESPACE = 'demo_agent'
log = Logger(NAMESPACE)


def to_variable(data, use_cuda):
    input_, target_ = data
    input_, target_ = Variable(input_.float()), Variable(
        target_.float().unsqueeze(1))
    if use_cuda:
        input_, target_ = input_.cuda(), target_.cuda()

    return input_, target_


class DemoAgent(LearningAgent):
    def __init__(self, cfg):
        super(DemoAgent, self).__init__(cfg)

        use_cuda = self._use_cuda

        in_size, out_size = None, None

        # -- Get variables from cfg
        train_cfg = cfg.train
        self.max_grad_norm = train_cfg.max_grad_norm

        # -- Initialize model
        model_class = get_models(cfg.model)
        self.model = model_class[0](cfg.model, in_size, out_size)

        self._models.append(
            self.model)  # -- Add models & optimizers to base for saving
        if use_cuda:
            self.cuda()

        # -- Initialize optimizers
        self.optimizer = self.get_optim(cfg.train.algorithm,
                                        cfg.train.algorithm_args, self.model)
        self._optimizers.append(
            self.optimizer)  # -- Add models & optimizers to base for saving

        # -- Initialize criterion
        self.criterion = getattr(torch.nn, cfg.train.criterion)()

        # -- Change settings
        self.set_eval_metric_comparison(True)

        super(DemoAgent, self).__end_init__()

    def _session_init(self):
        if self._is_train:
            # Used variables durin training
            self.optimizer.zero_grad()

    def _train(self, data_loader):
        optimizer = self.optimizer
        use_cuda = self._use_cuda
        model = self.model
        criterion = self.criterion

        train_loss = 0

        progress_bar = ProgressBar(
            'Loss: %(loss).3f', dict(loss=0), len(data_loader))
        for batch_idx, (centers, lefts, rights) in enumerate(data_loader):

            optimizer.zero_grad()

            centers = to_variable(centers, use_cuda)
            lefts = to_variable(lefts, use_cuda)
            rights = to_variable(rights, use_cuda)

            datas = [lefts, rights, centers]
            for data in datas:
                imgs, targets = data
                outputs = model(imgs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.data[0]

            progress_bar.update(
                batch_idx, dict(loss=(train_loss / (batch_idx + 1))))

        progress_bar.finish()

        return train_loss, {}

    def _test(self, data_loader):
        model = self.model
        criterion = self.criterion
        use_cuda = self._use_cuda

        test_loss = 0

        progress_bar = ProgressBar(
            'Loss: %(loss).3f', dict(loss=0), len(data_loader))
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = Variable(inputs.float()), Variable(
                targets.float().unsqueeze(1))
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data[0]

            progress_bar.update(
                batch_idx, dict(loss=(test_loss / (batch_idx + 1))))

            break

        progress_bar.finish()

        return test_loss, None, {}

    def _control_function(self, image_input_raw, speed, control_input):
        # outputs = self.model(input)
        steer, acc, brake = 0.0, 0.9, 0.0
        return steer, acc, brake