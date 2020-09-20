import os
import time

import numpy as np


class Metrics:
    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._metrics = []

    def start(self):
        self._start_time = time.perf_counter()
        self._end_time = self._start_time
        return self

    def update(self):
        now = time.perf_counter()
        self._metrics.append((now - self._end_time, self.iter_per_sec(), self.sec_per_iter()))
        self._end_time = now

    def elapsed(self):
        return self._end_time - self._start_time

    def iter_per_sec(self):
        """Calculates approximate iterations per second"""
        if self._end_time == self._start_time:
            return 0
        return len(self) / self.elapsed()

    def sec_per_iter(self):
        if self._end_time == self._start_time:
            return 0
        """Calculates approximate seconds per iteration"""
        return self.elapsed() / len(self)

    def __len__(self):
        return len(self._metrics)

    def get(self):
        return self._metrics

    def save(self, filename):
        dirname = os.path.dirname(os.path.abspath(filename))
        os.makedirs(dirname, exist_ok=True)

        np.savetxt(filename, self._metrics)

