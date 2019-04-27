import operator
import numpy as np

MODELS_KEY = "models"
OPTIMIZERS_KEY = "optimizers"
TRAIN_EPOCH_KEY = "train_epoch"
DATA_SAVE_PREFIX = "agent_data"
BEST_EVAL_SCORE_KEY = "best_eval_score"
BEST_TRAIN_LOSS_KEY = "best_train_loss"
LAST_EVAL_SCORE_KEY = "last_eval_score"
LAST_TRAIN_LOSS_KEY = "last_train_loss"


class EvalMetric:
    def __init__(self, val=None, ep=None, lower_is_better=True, name="metric"):
        self.metric = name
        self._compare = operator.lt if lower_is_better else operator.gt
        if ep is None:
            self.ep = -1
        else:
            self.ep = ep

        if val is None:
            if lower_is_better:
                self.val = np.inf
            else:
                self.val = -np.inf
        else:
            self.val = val

    def compare(self, other_eval_metric):
        return self._compare(other_eval_metric, self.val)

    def set_comparison(self, lower_is_better):
        if abs(self.val) == np.inf:
            if lower_is_better:
                self.val = np.inf
            else:
                self.val = -np.inf

        self._compare = operator.lt if lower_is_better else operator.gt

    def update_if_better(self, val, ep):
        better = self.compare(val)
        if better:
            self.val = val
            self.ep = ep
        return better

    def set(self, val, ep):
        self.val = val
        self.ep = ep

    def get(self):
        return self.val, self.ep
