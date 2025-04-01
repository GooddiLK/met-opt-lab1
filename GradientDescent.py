import numpy as np

class GradientDescent:
    def __init__(self, func, grad, learningRate, stoppingCriteria):
        self.funcFunc = func
        self.gradFunc = grad
        self.learningRate = learningRate
        self.stoppingCriteria = stoppingCriteria
        self.funcDict = dict()
        self.gradDict = dict()

    def next_point(self, point, learningRate):
        return np.add(point, np.multiply(self.vector, learningRate))

    def epoch(self):
        return len(self.history) - 1

    def func(self, x_k):
        if x_k in self.funcDict:
            return self.funcDict[x_k]
        self.funcCalculation += 1
        f = self.funcFunc(x_k)
        self.funcDict[x_k] = f
        return f

    def grad(self, x_k):
        if x_k in self.gradDict:
            return self.gradDict[x_k]
        self.gradCalculation += 1
        g = self.gradFunc(x_k)
        self.gradDict[x_k] = g
        return g

    def __call__(self, startPoint):
        point = startPoint
        self.history = [startPoint]
        self.funcCalculation = 0
        self.gradCalculation = 0
        while True:
            self.vector = np.multiply(self.grad(point), -1)
            point = self.next_point(point, self.learningRate.learning_rate(self))
            b = self.stoppingCriteria(self, point)
            self.history.append(point)
            if b:
                break
        return self.history, self.funcCalculation, self.gradCalculation
