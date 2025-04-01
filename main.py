from GradientDescent import GradientDescent
from LearningRateScheduling import LearningRateSchedulingConstant
from OneDimensional import Armiho
from StoppingCriteria import Iterations, SequenceEps

def fu(x):
    return x ** 2

def gr(x):
    return 2 * x

gd = GradientDescent(fu, gr, LearningRateSchedulingConstant(0.25), SequenceEps(0.0001))
print("\n".join([str(i) for i in gd(2)[0]]))
print("\n----------\n")
gd = GradientDescent(fu, gr, Armiho(3, 0.0001, 0.5), SequenceEps(0.0001))
print("\n".join([str(i) for i in gd(2)[0]]))
