# Imports for ReportAgent
import pandas as pd
import time
import os
import pickle
from logbook import Logger

from .tensorboard_summary import TensorboardSummary

NAMESPACE = 'report'
log = Logger(NAMESPACE)

class Testing:
    def __init__(self):
        self.a = 123

class ReportAgent:
    def __init__(self, cfg):
        self.name = cfg.common.name
        self.path = cfg.common.path
        self.title = cfg.common.title
        self.reporting_freq = cfg.reporting_freq
        self.ep_w = ep_w = cfg.ep_w
        self.step_w = step_w = cfg.step_w
        plot_type = cfg.plot


        # -- Plotting support
        self.use_tensorboard = use_tensorboard = "tensorboard" in plot_type or "both" in plot_type

        self.init_unsavables()

        # Other data
        self.start_time = time.time()
        self.end_time = time.time()
        self.eval_time = 0

        self.step = 0
        self.episode = 0
        self.last_eval_step = -99999999999999999

        self.metrics = dict()


    def init_unsavables(self):
        use_tensorboard = self.use_tensorboard

        # -- Tensorboard SummaryWriter
        if use_tensorboard:
            self.tensorboard_summary = TensorboardSummary(self.name, path=self.path)

    def register_metric(self, parent, fast_metric, metric_category, group):
        metric_name = fast_metric.name


    def start_report(self):
        self.start_time = time.time()

    def report(self):

        end_time = time.time()
        self.fps.update(float(self.reporting_freq) /
                        (end_time - self.start_time - self.eval_time))

        self.start_time, self.eval_time = end_time, .0

        if self.use_tensorboard:
            self.tensorboard_summary.tick(step, other_scalars=other_info)

    def evaluation_update(self, avg_rw, avg_ret, avg_len):
        self.eval_average_reward.update(avg_rw)
        self.eval_episodic_return.update(avg_ret)
        self.eval_average_length.update(avg_len)

    def close_plot_data(self):

        if self.use_tensorboard:
            self.tensorboard_summary.close()

    def save(self, prefix):
        # -- Save metrics objects for resuming training

        # Trick (tensorboard not serializable)
        tensorboard_summary = self.tensorboard_summary
        self.tensorboard_summary = None

        with open(prefix + "metrics.pkl", 'wb') as f:
            pickle.dump(self, f)

        self.tensorboard_summary = tensorboard_summary


        if self.use_tensorboard:
            self.tensorboard_summary.save_scalars(prefix + "results.json")

    @staticmethod
    def resume(resume_prefix):
        with open(resume_prefix + "metrics.pkl", 'rb') as f:
            metrics = pickle.load(f)

        metrics.init_unsavables()

        return metrics


