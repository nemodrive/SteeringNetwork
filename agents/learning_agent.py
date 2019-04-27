from __future__ import print_function
import os
import torch
from logbook import Logger
import subprocess

from reporting.printing import Printer
from utils.torch_types import TorchTypes
from .agent_utils import *
from .simulator_agent import SimulatorAgent
from tensorboardX import SummaryWriter


NAMESPACE = 'learning_agent'
log = Logger(NAMESPACE)


class LearningAgent(Printer, SimulatorAgent):
    def __init__(self, cfg):
        super(LearningAgent, self).__init__(cfg.name, verbose=cfg.verbose)

        self._use_cuda = use_cuda = cfg.use_cuda
        self._data_parallel = cfg.data_parallel
        self._save_path = cfg.common.save_path
        self._save_freq = cfg.save_freq
        self._save_best = cfg.save_best
        self._save_best_freq_lim = cfg.save_best_freq_lim
        self._last_ep_saved_best = -np.inf
        self._resume_path = cfg.resume

        self.types = TorchTypes(cuda=use_cuda)

        # -- models and optimizers should be loaded after initialization in this list
        self._models = []
        self._optimizers = []

        self._is_train = True
        self._eval_agent = False
        self._train_epoch = 0

        # -- Should redefine comparison sign according to metric
        self._last_train_loss = EvalMetric(
            lower_is_better=True, name="last_train_loss")
        self._last_eval_score = EvalMetric(
            lower_is_better=True, name="last_eval_score")

        self._best_train_loss = EvalMetric(
            lower_is_better=True, name="best_train_loss")
        self._best_eval_score = EvalMetric(
            lower_is_better=True, name="best_eval_score")

        # Dictionary generator for saving data
        self._save_data = [
            "_models", "_optimizers", "_train_epoch", "_best_train_loss",
            "_best_train_loss", "_best_eval_score", "_last_train_loss",
            "_last_eval_score"
        ]

        #self._writer = SummaryWriter(log_dir=self._save_path, comment="tensorboard")
        #self._start_tensorboard()

        # Info necessary for running simulator

    def __end_init__(self):
        if self._resume_path:
            self.resume(self._resume_path)

    def set_eval_metric_comparison(self, lower_is_better):
        self._last_eval_score.set_comparison(lower_is_better)
        self._best_eval_score.set_comparison(lower_is_better)

    def set_train_metric_comparison(self, lower_is_better):
        self._last_train_loss.set_comparison(lower_is_better)
        self._best_train_loss.set_comparison(lower_is_better)

    def get_train_epoch(self):
        return self._train_epoch

    def get_eval_metrics(self):
        val, ep = self._last_eval_score.get()
        best_val, best_ep = self._best_eval_score.get()
        return ep, val, best_ep, best_val

    def get_train_metrics(self):
        val, ep = self._last_train_loss.get()
        best_val, best_ep = self._best_train_loss.get()
        return ep, val, best_ep, best_val

    def session_init(self):
        return self._session_init()

    def _session_init(self):
        raise NotImplemented

    def run_step(self, measurements, sensor_data, directions, target):
        return self._run_step(measurements, sensor_data, directions, target)

    def train(self, dataloader):
        log.info("--> Train")
        self.set_train_mode()
        self.session_init()
        self._train_epoch += 1
        train_epoch = self._train_epoch
        loss, other = self._train(dataloader)

        self._last_train_loss.set(loss, train_epoch)
        self._best_train_loss.update_if_better(loss, train_epoch)

        if not self.cfg.use_progress_bar:
            ep, last_loss, best_ep, best_loss = self.get_train_metrics()
            log.info("Last train loss: episode {}, loss {}".format(ep, last_loss))
            log.info("Best train loss: episode {}, loss {}".format(best_ep, best_loss))

        if train_epoch % self._save_freq == 0:
            self.save(prefix=DATA_SAVE_PREFIX + "_{}".format(train_epoch))

        return loss, other

    def _train(self, dataloader):
        raise NotImplemented

    def test(self, dataloader):
        log.info("--> Test")
        train_epoch = self._train_epoch

        self.set_eval_mode()
        self.session_init()

        score, is_best, other = self._test(dataloader)

        self._last_eval_score.set(score, train_epoch)
        is_best_standard = self._best_eval_score.update_if_better(
            score, train_epoch)

        if not self.cfg.use_progress_bar:
            ep, last_score, best_ep, best_score = self.get_eval_metrics()
            log.info("Last test loss: episode {}, loss {}".format(ep, last_score))
            log.info("Best test loss: episode {}, loss {}".format(best_ep, best_score))

        if self._save_best:
            if is_best is None:
                is_best = is_best_standard

            if is_best and not self._eval_agent:
                if self._last_ep_saved_best + self._save_best_freq_lim < train_epoch:
                    self.save(
                        prefix=DATA_SAVE_PREFIX + "_best".format(train_epoch))

        return score, is_best, other

    def _test(self, dataloader):
        raise NotImplemented

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        if self._is_train:
            for m in self._models:
                m.eval()
            self._set_eval_mode()
        self._is_train = False

    def _set_eval_mode(self):
        pass

    def set_train_mode(self):
        """ Set agent to training mode """
        if not self._is_train:
            for m in self._models:
                m.train()
            self._set_train_mode()
        self._is_train = True

    def _set_train_mode(self):
        pass

    @property
    def is_training(self):
        return self._is_train

    def cuda(self):
        """ Set agent to run on CUDA """
        models = self._models
        for m in models:
            m.cuda()

        if self._data_parallel[0]:

            ids_ = self._data_parallel[1]
            if not isinstance(ids_, list):
                ids_ = range(torch.cuda.device_count())

            for i in range(len(models)):
                models[i] = torch.nn.DataParallel(models[i], device_ids=ids_)

        torch.backends.benchmark = True

    def save(self, prefix="agent_data_"):
        save_data = {key: self.__dict__[key] for key in self._save_data}
        save_data = self._save(save_data, self._save_path)
        torch.save(save_data, os.path.join(self._save_path, prefix))

    def _save(self, save_data, path):
        return save_data

    def resume(self, agent_check_point_path):

        log.info("Resuming agent from {}".format(agent_check_point_path))
        data = torch.load(agent_check_point_path)

        # Resume save data
        for key in self._save_data:
            self.__dict__[key] = data[key]

        self._resume(agent_check_point_path, data)

    def _resume(self, agent_check_point_path, saved_data):
        pass

    @staticmethod
    def get_optim(algorithm, algorithm_args, model):
        _optimizer = getattr(torch.optim, algorithm)
        optim_args = vars(algorithm_args)
        return _optimizer(model.parameters(), **optim_args)

    @staticmethod
    def get_sched(algorithm, algorithm_args, optimizer):
        _scheduler = getattr(torch.optim.lr_scheduler, algorithm)
        sched_args = vars(algorithm_args)
        return _scheduler(optimizer, **sched_args)

    def _start_tensorboard(self, kill_other=True):
        if kill_other:
            os.system("killall -9 tensorboard")

        save_path = self._save_path
        logdir = "--logdir" + "=" + save_path
        port = "--port" + "=" + str(8008)
        subprocess.Popen(["tensorboard", logdir, port])

    def _end_tensorboard(self):
        os.sysyem("killall -2 tensorboard")

