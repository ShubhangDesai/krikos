import numpy as np

class Network(object):
    def __init__(self, layers, loss, lr):
        super(Network, self).__init__()
        self.layers = layers
        self.loss = loss
        self.lr = lr

    def train(self, input, target):
        layers = self.layers
        loss = self.loss

        for layer in layers:
            input = layer.forward(input)

        l = loss.forward(input, target)
        dout = loss.backward()

        for layer in reversed(layers):
            dout = layer.backward(dout)

            for param, grad in layer.grads.items():
                layer.params[param] -= self.lr * grad

        return np.argmax(input, axis=1), l

    def eval(self, input):
        layers = self.layers

        for layer in layers:
            input = layer.forward(input)

        return np.argmax(input, axis=1)