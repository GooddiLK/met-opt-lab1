# В этом файле содержатся алгоритмы планирования шага
# GradientDescent требует, чтобы объект планирования шага обладал методом learning_rate,
# принимающим объект типа GradientDescent и возвращающим текущий шаг


import numpy as np


# Уравнение касательной для функции, содержащейся в gd к точке x_k, с дополнительным коэффициентом c1
def l_alpha(gd, c1, x_k, alpha):
    return gd.func(x_k) + c1 * alpha * gd.grad(x_k)


# Класс, реализующий метод backtracking
class Backtracking:
    def __init__(self, alpha_0, c1, q):
        self.alpha_0 = alpha_0
        self.c1 = c1
        self.q = q

    # Функция ищущая следующий подходящий alpha для метода Армихо
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


# Класс, реализующий метод Армихо
class Armijo:
    def __init__(self, alpha_0, c1, q):
        if not (0 < c1 < 1):
            raise Exception("Incorrect parameters")
        self.bctrk = Backtracking(alpha_0, c1, q)

    def learning_rate(self, gd):
        return self.bctrk.calculate_alpha(gd, gd.history()[-1])


# Класс, реализующий бинпоиск подходящей точки для алгоритма Вольфа
class BinarySearch:
    def __init__(self, eps):
        self.eps = eps

    def __call__(self, gd, x_k, c1, c2, l, r):
        while r - l > self.eps:
            m = (l + r) / 2
            mp = gd.next_point(x_k, m)
            b1 = gd.func(mp) < l_alpha(gd, c1, x_k, m)
            pos0 = gd.grad(x_k) >= 0
            if pos0:
                b2 = c2 * gd.grad(x_k) >= gd.grad(mp)
            else:
                b2 = c2 * gd.grad(x_k) <= gd.grad(mp)
            if not b1:
                r = m
                continue
            if not b2:
                l = m
                continue
            return m
        return 0


# Класс, реализующий метод Вольфа
class Wolfe:
    def __init__(self, c1, c2, alpha_0, eps):
        if not (0 < c1 < c2 < 1):
            raise Exception("Incorrect parameters")
        self.c1 = c1
        self.c2 = c2
        self.alpha_0 = alpha_0
        self.eps = eps

    def learning_rate(self, gd):
        bs = BinarySearch(self.eps)
        return bs(gd, gd.history()[-1], self.c1, self.c2, 0, self.alpha_0)
