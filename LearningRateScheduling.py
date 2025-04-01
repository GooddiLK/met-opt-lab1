from math import sqrt, exp


class LearningRateSchedulingConstant:
    def __init__(self, learning_rate):
        self.learning_rate_c = learning_rate

    def learning_rate(self, gd):
        return self.learning_rate_c

class LearningRateSchedulingLinear:
    def __init__(self, k, learning_rate_0):
        self.k = k
        self.learning_rate_0 = learning_rate_0

    def learning_rate(self, gd):
        return self.learning_rate_0 - self.k * gd.epoch()

def h0(epoch):
    return 1/sqrt(epoch + 1)

class LearningRateSchedulingExponential:
    def __init__(self, lambdaParameter):
        self.lambdaParameter = lambdaParameter

    def learning_rate(self, gd):
        epoch = gd.epoch()
        return h0(epoch) * exp(-self.lambdaParameter * epoch)

class LearningRateSchedulingPolynomial:
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def learning_rate(self, gd):
        epoch = gd.epoch()
        return h0(epoch) * (self.beta * epoch + 1) ** (-self.alpha)
