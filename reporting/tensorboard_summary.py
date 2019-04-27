import os
from tensorboardX import SummaryWriter
import subprocess


class TensorboardSummary(object):

    def __init__(self, name, path):
        self.save_path = save_path = os.path.join(path, "tensorboard_logs")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self._writer = SummaryWriter(log_dir=save_path, comment=name)

        self.auto_start_board = True
        self.board_started = False

    def tick(self, data):
        for plot_name, metric_dict, step in data:

            for k2, v2 in metric_dict.items():
                metric_dict[k2] = float(v2)

            self._writer.add_scalars(plot_name, metric_dict, step)

        if self.auto_start_board and not self.board_started:
            self.start_tensorboard()
            self.board_started = True

    def save_scalars(self, file_name):
        self._writer.export_scalars_to_json(file_name)

    def close(self):
        self._writer.close()

    def start_tensorboard(self, kill_other=True):
        if kill_other:
            os.system("killall -9 tensorboard")

        save_path = self.save_path
        subprocess.Popen(["tensorboard", "--logdir", save_path])
