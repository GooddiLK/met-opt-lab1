import numpy as np
from contourpy import as_z_interp
from matplotlib.colorizer import Colorizer

from GradientDescent import GradientDescent
from numpy import arange
from numpy import meshgrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from LearningRateScheduling import LearningRateSchedulingConstant, LearningRateSchedulingLinear, \
    LearningRateSchedulingExponential, LearningRateSchedulingPolynomial
from OneDimensional import Armijo, Wolfe
from StoppingCriteria import Iterations, SequenceEps, SequenceValueEps

func_table = [
    [lambda x: x[0] ** 2 - 10, lambda x: 2 * x[0]],
    # lambdas = ((a+c) +- sqrt( (a-c)^2 + 4b^2 ))/2
    # x^2 + y^2; a = 1, b = 0, c = 1; lambdas = 1, 1; Число обусловленности = 1
    [lambda x: x[0] ** 2 + x[1] ** 2, lambda x: [2 * x[0], 2 * x[1]]],
    # 3x^2 - 4xy + 10y^2; a = 3; b = -4; c = 10; lambdas = 2.3699, 23.63; Число обусловленности = 9,97
    [lambda x: 3 * x[0] ** 2 + 10 * x[1] ** 2 - 4 * x[0] * x[1],
     lambda x: [6 * x[0] - 4 * x[1], 20 * x[1] - 4 * x[0]]],

    [lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2,
     lambda x: [2 * (x[0] ** 2 + x[1] - 11) * (2 * x[0]) + 2 * (x[0] + x[1] ** 2 - 7),
                2 * (x[0] + x[1] ** 2 - 7) * (2 * x[1]) + 2 * (x[0] ** 2 + x[1] - 11)]],
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

def to2(func):
    return lambda x, y: func([x, y])


def show(func_number, rng, grid, last_points, learning_rate, stopping_criteria, point, iterations):
    func = func_table[func_number][0]
    gd = GradientDescent(func, func_table[func_number][1], learning_rate, stopping_criteria)
    r = gd(point, iterations)[0]
    if last_points > 0:
        r = r[-last_points:]
    lx, ly = r[-1]
    xaxis = arange(-rng + lx, rng + lx, grid)
    yaxis = arange(-rng + ly, rng + ly, grid)
    x, y = meshgrid(xaxis, yaxis)
    results = to2(func)(x, y)
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot_surface(x, y, results, cmap='jet', alpha=0.5)
    rx = np.array([i[0] for i in r])
    ry = np.array([i[1] for i in r])
    rz = to2(func)(rx, ry)
    axis.scatter(rx, ry, rz, c="green")
    plt.show()


if __name__ == "__main__":
    # run(2, LearningRateSchedulingPolynomial(0.5, 1), SequenceValueEps(0.0001), [2, 0], 0)
    # run(2, Armijo(10, 0.9, 0.00001, 0.001), SequenceValueEps(0.0001), [2, 0], 0)
    # run(2, Wolfe(1,0.99, 0.00001, 0.001, 0.4), SequenceValueEps(10 ** -10), [2, 0], 0)
    rng = 2
    # show(3, rng, rng/20, 0, Armijo(1, 0.9, 0.0001, 0.001), SequenceEps(10 ** -10), [np.longdouble(3), np.longdouble(3)], 5 * 10 ** 3)

    f_num = 2
    test_point = [10, 2]
    iter_max = 10**4
    run(f_num, LearningRateSchedulingConstant(1), SequenceValueEps(0.0001), test_point, iter_max)
    run(f_num, LearningRateSchedulingLinear(0.5, 1), SequenceValueEps(0.0001), test_point, iter_max)
    run(f_num, LearningRateSchedulingExponential(1), SequenceValueEps(0.0001), test_point, iter_max)
    run(f_num, LearningRateSchedulingPolynomial(0.5, 1), SequenceValueEps(0.0001), test_point, iter_max)

    #run(1, Armijo(1, 0.9, 0.0001, 0.001), SequenceValueEps(0.001), [2, 2], 5 * 10**3)
    #show(1, rng, rng/20, 300, Armijo(1, 0.9, 0.0001, 0.001), SequenceEps(10 ** -3), [np.longdouble(3), np.longdouble(3)], 5 * 10 ** 3)

    # run(3, Armijo(1, 0.9, 0.0001, 0.001), SequenceEps(10 ** -10), [np.longdouble(3), np.longdouble(3)], 5 * 10 ** 3)
    # run(3, Wolfe(12, 0.001, 0.1, 0.0001), SequenceEps(10 ** -10), [3, 3], 5 * 10 ** 3)
    # run(1, Armijo(1, 0.99, 0.00001, 0.001), SequenceEps(10 ** -10), [10], 5 * 10 ** 6)
    # run(1, Wolfe(1,0.99, 0.00001, 0.001, 0.4), SequenceEps(10 ** -10), [10], 5 * 10 ** 6)
