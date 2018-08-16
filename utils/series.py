from .mean_metric import FastMeanMetric

import torch
import pandas as pd


class Series(object):

    def __init__(self, variables: list, capacity=1024, run=1, algorithm=""):
        self._capacity = capacity
        self._size = 0
        self.run = run
        self.algorithm = algorithm

        self._data = {}
        self._metrics = {}
        for variable in variables:
            if isinstance(variable, tuple):
                metric, window = variable
                assert isinstance(metric, FastMeanMetric) and isinstance(window, int)
                self._metrics[metric.name] = (metric, window)
                name = metric.name
            else:
                name = variable
            self._data[name] = torch.FloatTensor().resize_(self._capacity)
        self._idx = torch.LongTensor().resize_(self._capacity)

    def tick(self, values=None, idx=None):
        values = {} if values is None else values
        size = self._size
        if idx is None:
            idx = (self._idx[size - 1] + 1) if size > 0 else 0
        self._idx[size] = idx

        for name, data in self._data.items():
            if name in values:
                data[size] = values[name]
            else:
                metric, window = self._metrics[name]
                data[size] = metric.get_mean(window)

        self._size += 1
        if self._size == self._capacity:
            self._capacity *= 2
            self._idx.resize_(self._capacity)
            for data in self._data.values():
                data.resize_(self._capacity)

    def df(self):
        dct = {
            "Step": self._idx[:self._size].numpy(),
            "Run": self.run,
            "Algorithm": self.algorithm
        }
        for name, data in self._data.items():
            dct[name] = data[:self._size].numpy()

        return pd.DataFrame(dct)

    def load_df(self, df):
        dfd = df.to_dict()
        self.algorithm = dfd['Algorithm'][0]
        self.run = dfd['Run'][0]

        idx = list(map(lambda d: d[1], dfd['Step'].items()))
        self._size = len(idx)
        self._capacity = 2 * self._size

        self._idx.resize_(self._capacity)
        for data in self._data.values():
            data.resize_(self._capacity)

        for i in range(self._size):
            self._idx[i] = idx[i]

        for name, data in dfd.items():
            if name == 'Step' or name == 'Run' or name == 'Algorithm':
                continue
            items = list(map(lambda d: d[1], data.items()))
            for i in range(len(items)):
                self._data[name][i] = items[i]
