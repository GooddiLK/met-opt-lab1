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

class Armijo:
    def __init__(self, alpha_0, c1, q):
        self.bctrk = Backtracking(alpha_0, c1, q)

    def learning_rate(self, gd):
        return self.bctrk.calculate_alpha(gd, gd.history()[-1])

class BinarySearch:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, gd, x_k, c1, c2, l, r):
        any_r = False
        while r - l > self.eps:
            m = (l + r) / 2
            mp = gd.next_point(x_k, m)
            b1 = gd.func(mp) < l_alpha(gd, c1, x_k, m)
            pos0 = gd.grad(x_k) >= 0
            if pos0:
                b2 = c2 * gd.grad(x_k) >= gd.grad(mp)
            else:
                b2 = c2 * gd.grad(x_k) <= gd.grad(mp)
            pos_grad = gd.grad(mp) >= 0
            if not b1:
                r = m
                continue
            if not b2:
                l = m
                continue
            any_r = True
            if pos_grad and pos0 or not pos_grad and not pos0:
                l = m
            else:
                r = m
        return (l + r) / 2, any_r

class Wolfe:
    def __init__(self, c1, c2, alpha_0, eps):
        if c1 >= c2:
            raise Exception("c1 must be less than c2")
        self.c1 = c1
        self.c2 = c2
        self.alpha_0 = alpha_0
        self.eps = eps

    def learning_rate(self, gd):
        bs = BinarySearch(self.eps)
        lr, any_r = bs(gd, gd.history()[-1], self.c1, self.c2, 0, self.alpha_0)
        if not any_r:
            return 0
        return lr
