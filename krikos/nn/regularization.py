import numpy as np

class Regularization(object):
    def __init__(self, weight):
        super(Regularization, self).__init__()
        self.weight = weight

    def forward(self, param):
        raise NotImplementedError

    def backward(self, param):
        raise NotImplementedError


class L2(Regularization):
    def __init__(self, weight):
        super(L2, self).__init__(weight)

    def forward(self, param):
        return self.weight * np.sum(param * param)

    def backward(self, param):
        return 2 * self.weight * param