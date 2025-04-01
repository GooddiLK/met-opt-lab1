from GradientDescent import GradientDescent
from LearningRateScheduling import LearningRateSchedulingConstant
from OneDimensional import Armijo, Wolfe
from StoppingCriteria import Iterations, SequenceEps

func_table = [
    [lambda x: x ** 2, lambda x: 2 * x],
    [lambda x: x[0] ** 2 + x[1] ** 2, lambda x: [2 * x[0], 2 * x[1]]]
]


def print_res(gd_inst, point, iterations):
    print("\n".join([str(i) for i in gd_inst(point, iterations)[0]]))


def run(func_number, learning_rate, stopping_criteria, point, iterations):
    gd = GradientDescent(func_table[func_number][0], func_table[func_number][1], learning_rate, stopping_criteria)
    print_res(gd, point, iterations)


if __name__ == "__main__":
    run(1, LearningRateSchedulingConstant(0.4), SequenceEps(0.0001), [2, 0], 100)
    print("--------")
    run(1, Armijo(3, 0.0001, 0.5), SequenceEps(0.0001), [2, 0], 100)
    print("--------")
    run(1, Wolfe(0.0001, 0.0005, 3, 0.0001), SequenceEps(0.0001), [2, 0], 100)
    print("--------")
