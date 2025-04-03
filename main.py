import numpy as np

from GradientDescent import GradientDescent
from LearningRateScheduling import LearningRateSchedulingConstant, LearningRateSchedulingLinear, \
    LearningRateSchedulingExponential, LearningRateSchedulingPolynomial
from OneDimensional import Armijo, Wolfe
from StoppingCriteria import Iterations, SequenceEps, SequenceValueEps

func_table = [
    [lambda x: x[0] ** 2 - 10, lambda x: 2 * x[0]],
    [lambda x: x[0] ** 2 + x[1] ** 2, lambda x: [2 * x[0], 2 * x[1]]],
    [lambda x: x[0] ** 2 + 20 * x[1] ** 2 - 8 * x[0] * x[1] - x[1], lambda x: [2 * x[0] - 8 * x[1], 40 * x[1] - 8 * x[0] - 1]],
    [lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
      lambda x: [2 * (x[0] ** 2 + x[1] - 11) * (2 * x[0]) + 2 * (x[0] + x[1] ** 2 - 7), 2 * (x[0] + x[1] ** 2 - 7) * (2*x[1]) + 2 * (x[0] ** 2 + x[1] - 11)]]
]


def print_res(gd_inst, point, iterations):
    r = gd_inst(point, iterations)
    print("кол-во итераций + конечная точка")
    print(len(r[0]), r[0][-1])
    print("кол-во вызовов функции и ее производной")
    print(r[1:])
    print("---------------------------------------")


def run(func_number, learning_rate, stopping_criteria, point, iterations):
    gd = GradientDescent(func_table[func_number][0], func_table[func_number][1], learning_rate, stopping_criteria)
    print_res(gd, point, iterations)


if __name__ == "__main__":
    # run(2, LearningRateSchedulingPolynomial(0.5, 1), SequenceValueEps(0.0001), [2, 0], 0)
    # print("--------")
    # run(2, Armijo(10, 0.9, 0.00001, 0.001), SequenceValueEps(0.0001), [2, 0], 0)
    # print("--------")
    # run(2, Wolfe(1,0.99, 0.00001, 0.001, 0.4), SequenceValueEps(10 ** -10), [2, 0], 0)
    run(3, Armijo(1, 0.9, 0.0001, 0.001), SequenceEps(10 ** -10), [np.longdouble(3), np.longdouble(3)], 5 * 10 ** 3)
    run(3, Wolfe(12, 0.001, 0.1, 0.0001), SequenceEps(10 ** -10), [3, 3], 5 * 10 ** 3)
    # run(1, Armijo(1, 0.99, 0.00001, 0.001), SequenceEps(10 ** -10), [10], 5 * 10 ** 6)
    # run(1, Wolfe(1,0.99, 0.00001, 0.001, 0.4), SequenceEps(10 ** -10), [10], 5 * 10 ** 6)
    # print("--------")
