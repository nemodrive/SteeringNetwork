from sklearn.model_selection import train_test_split
import random

from .dataset import get_dataset


class DataLoaderBase(object):
    def __init__(self, cfg):
        seed = cfg.data_seed
        self._seed = seed if seed != -1 else random.randint(1, 9999999)
        cfg.data_seed = self._seed
        self._dataset_cfg = cfg.dataset
        self._dataset = get_dataset(cfg.dataset)
        self._dataset_path = cfg.dataset.dataset_path
        self._dataset_path_test = cfg.dataset.dataset_eval_path
        self._test_size = cfg.test_size
        self._shuffle = cfg.shuffle
        self._no_workers = cfg.no_workers
        self._batch_size = cfg.batch_size

        self.data = None

    def get_train_loader(self):
        raise NotImplemented

    def get_test_loader(self):
        raise NotImplemented

    def load_data(self):
        raise NotImplemented

    def split_data(self, x, y):
        test_size = self._test_size
        shuffle = self._shuffle
        seed = self._seed

        x_train, x, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                       shuffle=shuffle, random_state=seed)

        return (x_train, y_train), (x, y_test)
