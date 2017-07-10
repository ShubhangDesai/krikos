import numpy as np

class Layer(object):
    def __init__(self):
        super(Layer, self).__init__()
        self.params = {}
        self.cache = {}
        self.grads = {}

    def forward(self, input):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError

class Linear(Layer):
    def __init__(self, input_dim, output_dim):
        super(Linear, self).__init__()
        self.params["W"] = np.random.randn(input_dim, output_dim) * 0.01
        self.params["b"] = np.zeros(output_dim)

    def forward(self, input):
        output = np.matmul(input, self.params["W"]) + self.params["b"]
        self.cache["input"] = input
        return output

    def backward(self, dout):
        input = self.cache["input"]
        self.grads["W"] = np.matmul(input.T, dout)
        self.grads["b"] = np.sum(dout, axis=0)

        dout = np.matmul(dout, self.params["W"].T)
        return dout