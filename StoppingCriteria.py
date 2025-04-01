import numpy as np


class Iterations:
    def __init__(self, number):
        self.number = number

    def __call__(self, gd):
        return len(gd.history) >= self.number

class SequenceEps:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, gd, point):
        return np.abs(np.subtract(gd.history[-1], point)) < self.eps