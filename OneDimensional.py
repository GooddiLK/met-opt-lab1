import numpy as np

def l_alpha(gd, c1, x_k, alpha):
    return gd.func(x_k) + c1 * alpha * gd.grad(x_k)

class Backtracking:
    def __init__(self, alpha_0, c1, q):
        self.alpha_0 = alpha_0
        self.c1 = c1
        self.q = q

    def calculate_alpha(self, gd, x_k):
        alpha = self.alpha_0
        while True:
            next_point = gd.next_point(x_k, alpha)
            if gd.stoppingCriteria(gd, next_point):
                return alpha
            if gd.func(next_point) < l_alpha(gd, self.c1, x_k, alpha):
                return alpha
            else:
                alpha = self.q * alpha

class Armiho:
    def __init__(self, alpha_0, c1, q):
        self.bctrk = Backtracking(alpha_0, c1, q)

    def learning_rate(self, gd):
        return self.bctrk.calculate_alpha(gd, gd.__history__[-1])