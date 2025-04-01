import numpy as np


class Iterations:
    def __init__(self, number):
        self.number = number

    def __call__(self, gd, point):
        return len(gd.history()) >= self.number

class IterationsPlus:
    def __init__(self, number, plus):
        self.iterations = Iterations(number)
        self.plus = plus

    def __call__(self, gd, point):
        return self.iterations(gd, point) or self.plus(gd, point)

class SequenceEps:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, gd, point):
        return np.linalg.norm(np.subtract(gd.history()[-1], point)) < self.eps
