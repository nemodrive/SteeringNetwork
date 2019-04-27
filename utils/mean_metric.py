import torch
from collections import namedtuple
from sys import float_info
import unittest

Variable = namedtuple("Variable", ["window", "value", "best", "trends"])


class FastMeanMetric(object):

    HIGHER = 1
    SAME = 0
    LOWER = -1

    def __init__(self, name, windows, follow=None, use_trends=None, fmt=".3f"):
        assert isinstance(windows, list)
        windows = sorted(windows)
        assert all([w >= 0 and isinstance(w, int) for w in windows])
        assert all([w > 0 for w in windows[1:]])

        if windows[0] == 0:
            use_global = True
            windows = windows[1:]
        else:
            use_global = False

        self.name = name
        self.fmt = fmt
        self.windows = windows
        self.n = n = max(windows + [0])
        self.coeff = [1. / float(w) for w in windows]
        self.crt_idx = 0
        self.seen = 0
        self.means = [.0 for _ in windows]
        self._w_to_m_idx = {w: i for (i, w) in enumerate(windows)}

        if use_trends is not None:
            assert isinstance(use_trends, int)
            self._has_trends = use_trends
            self.old_means = [.0 for _ in windows]
            self.trends = [[FastMeanMetric.SAME] * use_trends for _ in windows]
        else:
            self._has_trends = False
        self.history = torch.zeros(n)

        self.follow = follow
        if follow == "min":
            self.best = [(float_info.max, -1) for _ in windows]
            self.f = lambda a, b: a < b
        elif follow == "max":
            self.best = [(-(float_info.max - 1), -1) for _ in windows]
            self.f = lambda a, b: a > b

        self.use_global = use_global
        if use_global:
            self.global_mean = 0
            if follow == "min":
                self.global_best = (float_info.max, -1)
            elif follow == "max":
                self.global_best = (-(float_info.max - 1), -1)

    @property
    def has_best(self):
        return hasattr(self, "best")

    @property
    def has_trends(self):
        return self._has_trends

    def _update_best(self):
        seen = self.seen

        self.best = [(v, seen) if (w <= seen and self.f(v, b[0])) else b for
                     (w, v, b) in zip(self.windows, self.means, self.best)]
        if self.use_global:
            if self.f(self.global_mean, self.global_best[0]):
                self.global_best = (self.global_mean, self.seen)

    def update_many(self, values):
        n = self.n
        m = values.nelement()

        if self.use_global:
            seen = float(self.seen)
            self.global_mean = \
                (self.global_mean * seen + values.sum()) / (seen + m)

        self.seen = seen = self.seen + m

        if n > 0:
            crt_idx = self.crt_idx
            means = self.means
            history = self.history

            if m > n:
                values = values[-n:]
                m = n
            for i, (w, c) in enumerate(zip(self.windows, self.coeff)):
                if w <= m:
                    means[i] = values[-w:].float().mean()
                else:
                    correction = 0
                    old_s_idx = (crt_idx - w) % n
                    old_e_idx = old_s_idx + m
                    if old_e_idx > n:
                        old_e_idx -= n
                        correction -= history[old_s_idx:].sum()
                        correction -= history[:old_e_idx].sum()
                    else:
                        correction -= history[old_s_idx:old_e_idx].sum()
                    means[i] += (values.sum() + correction) * c

                if hasattr(self, "trends") and seen % w == 0:
                    w_trend = self.trends[i]
                    new_mean, old_mean = means[i], self.old_means[i]
                    if new_mean > old_mean:
                        new_trend = self.HIGHER
                    elif new_mean < old_mean:
                        new_trend = self.LOWER
                    else:
                        new_trend = self.SAME
                    w_trend.pop(0)
                    w_trend.append(new_trend)
                    self.old_means[i] = new_mean

            start_idx, end_idx = crt_idx, crt_idx + m
            if end_idx > n:
                end_idx -= n
                m1 = n - start_idx
                history[start_idx:].copy_(values[:m1])
                history[:end_idx].copy_(values[m1:])
            else:
                history[start_idx:end_idx].copy_(values)

            self.crt_idx = end_idx % n

        if hasattr(self, "best"):
            self._update_best()

    def update(self, value):
        n = self.n
        crt_idx = self.crt_idx
        means = self.means
        history = self.history
        seen = float(self.seen)
        if self.use_global:
            self.global_mean = (self.global_mean * seen + value) / (seen + 1)

        self.seen = seen = self.seen + 1

        if n > 0:
            for i, (w, c) in enumerate(zip(self.windows, self.coeff)):
                old_idx = (crt_idx - w) % n
                means[i] += (value - history[old_idx]) * c

                if hasattr(self, "trends") and seen % w == 0:
                    w_trend = self.trends[i]
                    new_mean, old_mean = means[i], self.old_means[i]
                    if new_mean > old_mean + 1e-8:
                        new_trend = self.HIGHER
                    elif new_mean < old_mean - 1e-8:
                        new_trend = self.LOWER
                    else:
                        new_trend = self.SAME
                    w_trend.pop(0)
                    w_trend.append(new_trend)
                    self.old_means[i] = new_mean

            history[crt_idx] = value
            self.crt_idx = (self.crt_idx + 1) % n

        if hasattr(self, "best"):
            self._update_best()

    def get(self):
        result = []
        use_best, use_trends = hasattr(self, "best"), hasattr(self, "trends")
        if self.use_global:
            result.append(
                Variable(window=0,
                         value=self.global_mean,
                         best=(self.global_best if use_best else None),
                         trends=None)
            )
        for i, w in enumerate(self.windows):
            if w <= self.seen:
                result.append(
                    Variable(window=w, value=self.means[i],
                             best=(self.best[i] if use_best else None),
                             trends=(self.trends[i] if use_trends else None))
                )

        return result

    def get_mean(self, w):
        if w == 0:
            return self.global_mean
        elif w <= self.seen:
            return self.means[self._w_to_m_idx[w]]
        elif self.seen > 0:
            return self.history[:self.seen].mean()
        else:
            return 0


class TestFastMetric(unittest.TestCase):

    def _test_fast_metric(self):
        import random
        n = random.randint(10000, 20000)
        all_values = torch.randn(n).mul_(10.)
        start_idx = 0
        while start_idx < n:
            left = n - start_idx
            end_idx = start_idx + random.randint(min(10, left), min(200, left))
            value = random.randint(-50, +50)
            all_values[start_idx:end_idx].add_(value)
            start_idx = end_idx

        how_many = random.randint(1, 20)
        windows = torch.LongTensor().resize_(how_many).random_(1, n)
        if random.random() < .5:
            windows[how_many - 1] = n
        if random.random() < .5:
            windows[0] = 0

        windows = sorted(list(set(windows.tolist())))

        metric_one = FastMeanMetric("Rand", windows, follow="max", use_trends=5)

        i = 0
        while i < n:
            how_many = min(random.randint(1, 32), n - i)
            if random.random() < .5:
                for k in range(how_many):
                    metric_one.update(all_values[i + k])
            else:
                metric_one.update_many(all_values[i:(i + how_many)])

            i = i + how_many
            alive = len([w for w in windows if w <= i])

            result_one = metric_one.get()

            self.assertEqual(len(result_one), alive)
            for var in result_one:
                w = var.window
                m = var.value
                if w == 0:
                    w = i
                self.assertAlmostEqual(all_values[(i - w):i].mean(), m)

    def test_fast_metric(self, n=100):
        import sys
        for _ in range(n):
            self._test_fast_metric()
            sys.stdout.write(".")
            sys.stdout.flush()


if __name__ == "__main__":
    unittest.main()
