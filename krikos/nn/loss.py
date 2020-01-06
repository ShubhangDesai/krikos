import numpy as np


class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()
        self.cache = {}

    def forward(self, input, y):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class SoftmaxCrossEntropy(Loss):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, input, y):
        batch_size = input.shape[0]
        indeces = np.arange(batch_size)

        exp = np.exp(input)
        norm = (exp.T / np.sum(exp, axis=1)).T
        self.cache["norm"], self.cache["y"], self.cache["indeces"] = norm, y, indeces

        losses = -np.log(norm[indeces, y])
        return np.sum(losses) / batch_size

    def backward(self):
        norm, y, indeces = self.cache["norm"], self.cache["y"], self.cache["indeces"]
        dloss = norm
        dloss[indeces, y] -= 1
        return dloss