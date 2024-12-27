import numpy as np
import copy as copy
import mdptoolbox
from _logger import setup_logging
import os


def random_normal_trunc(mean, dev, low, up):
    x = np.random.normal(mean, dev)
    return np.clip(x, low, up)


class alpha_random_process:
    def __init__(self, alpha, dev, interval, name="iid"):
        self._attacker_start = alpha
        self._attacker = alpha
        self._other = 1 - alpha
        self._other_start = self._other
        self._dev = dev
        self._interval = interval
        if self._interval[0] == 0:
            self._interval = (1e-6, self._interval[1])
        self._name = name

    def reset(self):
        self._other = self._other_start
        self._attacker = self._attacker_start
        return self.get()

    def next(self):
        if self._name == "iid":
            lower_bound = self._interval[0]
            upper_bound = self._interval[1]
            self._attacker = random_normal_trunc(
                self._attacker_start, self._dev, lower_bound, upper_bound
            )
            self._other = 1 - self._attacker
        elif self._name == "brown":
            lower_bound = (1 - self._interval[1]) * self._attacker / self._interval[1]
            upper_bound = (1 - self._interval[0]) * self._attacker / self._interval[0]
            self._other = random_normal_trunc(
                self._other, self._dev, lower_bound, upper_bound
            )

        return self.get()

    def get(self):
        return self._attacker / (self._attacker + self._other)


class real_alpha_process:
    def __init__(self, alpha, interval, array):
        self._start_alpha = alpha
        self._alpha = alpha
        self._array = array
        self._interval = interval
        avg = 0.0
        length = 1600
        for i in range(length):
            avg += array[i]
        avg /= length
        self._attacker_hashrate = alpha * avg
        self._pointer = 0
        self._interval = interval

    def reset(self):
        self._alpha = self._start_alpha
        self._pointer = 0

    def get(self):
        return self._alpha

    def next(self):
        self._pointer += 1
        while self._array[self._pointer] == 0:
            self._pointer += 1
        self._alpha = np.clip(
            self._attacker_hashrate / self._array[self._pointer],
            self._interval[0],
            self._interval[1],
        )
        return self.get()

    def get_total(self):
        return self._array[self._pointer]
