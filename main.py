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


# gd = GradientDescent(fu, gr, LearningRateSchedulingConstant(0.25), SequenceEps(0.0001))
# print("\n".join([str(i) for i in gd(2)[0]]))
# print("\n----------\n")
# gd = GradientDescent(fu, gr, Armijo(3, 0.0001, 0.5), SequenceEps(0.0001))
# print("\n".join([str(i) for i in gd(2)[0]]))
# print("\n----------\n")
# gd = GradientDescent(fu, gr, Wolfe(0.0001, 0.0005, 3, 0.0001), SequenceEps(0.0001))
# print("\n".join([str(i) for i in gd(20)[0]]))
gd = GradientDescent(func_table[1][0], func_table[1][1], LearningRateSchedulingConstant(0.4), SequenceEps(0.0001))
print_res(gd, [2, 0], 0)
