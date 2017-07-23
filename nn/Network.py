import numpy as np

from nn.Layer import BatchNorm, BatchNorm2d, Dropout


class Network(object):
    def __init__(self):
        super(Network, self).__init__()
        self.diff = (BatchNorm, BatchNorm2d, Dropout)

    def train(self, input, target):
        raise NotImplementedError

    def eval(self, input):
        raise NotImplementedError


class Sequential(Network):
    def __init__(self, layers, loss, lr, regularization=None):
        super(Sequential, self).__init__()
        self.layers = layers
        self.loss = loss
        self.lr = lr
        self.regularization = regularization

    def train(self, input, target):
        layers = self.layers
        loss = self.loss
        regularization = self.regularization

        l = 0
        for layer in layers:
            if isinstance(layer, self.diff):
                layer.mode = "train"

            input = layer.forward(input)
            if regularization is not None:
                for _, param in layer.params.items():
                    l += regularization.forward(param)

        l += loss.forward(input, target)
        dout = loss.backward()

        for layer in reversed(layers):
            dout = layer.backward(dout)

            for param, grad in layer.grads.items():
                if regularization is not None:
                    grad += regularization.backward(layer.params[param])
                layer.params[param] -= self.lr * grad

        return np.argmax(input, axis=1), l

    def eval(self, input):
        layers = self.layers

        for layer in layers:
            if isinstance(layer, self.diff):
                layer.mode = "test"

            input = layer.forward(input)

        return np.argmax(input, axis=1)