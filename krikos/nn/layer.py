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


class Convolutional(Layer):
    def __init__(self, channels, num_filters, kernel_size, stride=1, pad=0):
        super(Convolutional, self).__init__()
        self.params["W"] = np.random.randn(num_filters, channels, kernel_size, kernel_size) * 0.01
        self.params["b"] = np.zeros(num_filters)

        self.stride = stride
        self.pad = pad

        self.F = num_filters
        self.HH, self.WW = kernel_size, kernel_size

    def forward(self, input):
        N, C, H, W = input.shape
        F, HH, WW = self.F, self.HH, self.WW
        stride, pad = self.stride, self.pad

        H_prime = 1 + (H + 2 * pad - HH) / stride
        W_prime = 1 + (W + 2 * pad - WW) / stride
        assert H_prime.is_integer() and W_prime.is_integer(), 'Invalid filter dimension'
        H_prime, W_prime = int(H_prime), int(W_prime)

        out = np.zeros((N, F, H_prime, W_prime))
        filters = self.params["W"].reshape(F, C * HH * WW)
        x_pad = np.pad(input, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

        for i in range(H_prime):
            h_start = i * stride
            h_end = h_start + HH
            for j in range(W_prime):
                w_start = j * stride
                w_end = w_start + WW

                kernel = x_pad[:, :, h_start:h_end, w_start:w_end]
                kernel = kernel.reshape(N, C * HH * WW)

                conv = np.matmul(kernel, filters.T) + self.params["b"]
                out[:, :, i, j] = conv

        self.cache["input"] = input
        return out

    def backward(self, dout):
        input = self.cache["input"]
        stride, pad = self.stride, self.pad
        N, C, H, W = input.shape
        F, HH, WW = self.F, self.HH, self.WW
        _, _, H_prime, W_prime = dout.shape

        H_pad, W_pad = H + 2 * pad, W + 2 * pad
        dx = np.zeros((N, C, H_pad, W_pad))
        dw = np.zeros_like(self.params["W"])
        db = np.sum(dout, axis=(0, 2, 3))
        filters = self.params["W"].reshape(F, C * HH * WW)
        x_pad = np.pad(input, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

        for i in range(H_prime):
            h_start = i * stride
            h_end = h_start + HH
            for j in range(W_prime):
                w_start = j * stride
                w_end = w_start + WW

                piece = dout[:, :, i, j]
                x_piece = x_pad[:, :, h_start:h_end, w_start:w_end].reshape(N, C * HH * WW)
                dx_piece = np.matmul(piece, filters)
                dw_piece = np.matmul(piece.T, x_piece)

                dx[:, :, h_start:h_end, w_start:w_end] += dx_piece.reshape(N, C, HH, WW)
                dw += dw_piece.reshape(F, C, HH, WW)

        dx = dx[:, :, pad:H_pad - pad, pad:W_pad - pad]
        self.grads["W"], self.grads["b"] = dw, db

        return dx

class MaxPooling(Layer):
    def __init__(self, kernel_size, stride=1, pad=0):
        super(MaxPooling, self).__init__()
        self.stride = stride
        self.pad = pad

        self.HH, self.WW = kernel_size, kernel_size

    def forward(self, input):
        N, C, H, W = input.shape
        HH, WW, stride = self.HH, self.WW, self.stride

        H_prime = (H - HH) / stride + 1
        W_prime = (W - WW) / stride + 1
        out = np.zeros((N, C, H_prime, W_prime))

        if not H_prime.is_integer() or not W_prime.is_integer():
            raise Exception('Invalid filter dimension')

        H_prime, W_prime = int(H_prime), int(W_prime)

        for i in range(H_prime):
            h_start = i * stride
            h_end = h_start + HH
            for j in range(W_prime):
                w_start = j * stride
                w_end = w_start + WW

                kernel = input[:, :, h_start:h_end, w_start:w_end]
                kernel = kernel.reshape(N, C, HH * WW)
                max = np.max(kernel, axis=2)

                out[:, :, i, j] = max

        self.cache['input'] = input
        return out

    def backward(self, dout):
        input = self.cache['input']
        N, C, H, W = input.shape
        HH, WW, stride = self.HH, self.WW, self.stride

        H_prime = int((H - HH) / stride + 1)
        W_prime = int((W - WW) / stride + 1)
        dx = np.zeros_like(input)

        for i in range(H_prime):
            h_start = i * stride
            h_end = h_start + HH
            for j in range(W_prime):
                w_start = j * stride
                w_end = w_start + WW

                max = dout[:, :, i, j]

                kernel = input[:, :, h_start:h_end, w_start:w_end]
                kernel = kernel.reshape(N, C, HH * WW)
                indeces = np.argmax(kernel, axis=2)
                grads = np.zeros_like(kernel)
                for n in range(N):
                    for c in range(C):
                        grads[n, c, indeces[n, c]] = max[n, c]

                dx[:, :, h_start:h_end, w_start:w_end] += grads.reshape(N, C, HH, WW)

        return dx


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        self.cache["shape"] = input.shape

        return input.reshape(input.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache["shape"])


# ACTIVATIONS

class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, input):
        mask = input >= 0
        self.cache["mask"] = mask
        input[~mask] = 0
        return input

    def backward(self, dout):
        mask = self.cache["mask"]
        dout = dout * mask
        return dout

# REGULARIZATION

class BatchNorm(Layer):
    def __init__(self, dim, epsilon=1e-5, momentum=0.9):
        super(BatchNorm, self).__init__()
        self.params['gamma'] = np.ones(dim)
        self.params['beta'] = np.zeros(dim)

        self.running_mean, self.running_var = np.zeros(dim), np.zeros(dim)
        self.epsilon, self.momentum = epsilon, momentum

        self.mode = "train"

    def forward(self, input):
        gamma, beta = self.params['gamma'], self.params['beta']
        running_mean, running_var = self.running_mean, self.running_var
        epsilon, momentum = self.epsilon, self.momentum

        if self.mode == 'train':
            mean, var = np.mean(input, axis=0), np.var(input, axis=0)
            norm = (input - mean) / np.sqrt(var + epsilon)
            output = gamma * norm + beta

            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var

            self.running_mean, self.running_var = running_mean, running_var
            self.cache['input'], self.cache['norm'], self.cache['mean'], self.cache['var'] = input, norm, mean, var
        else:
            norm = (input - running_mean) / np.sqrt(running_var)
            output = gamma * norm + beta

        return output

    def backward(self, dout):
        input, norm, mean, var = self.cache['input'], self.cache['norm'], self.cache['mean'], self.cache['var']
        gamma, beta = self.params['gamma'], self.params['beta']
        epsilon = self.epsilon
        N, _ = dout.shape

        self.grads['beta'] = np.sum(dout, axis=0)
        self.grads['gamma'] = np.sum(dout * norm, axis=0)

        dshift1 = 1 / (np.sqrt(var + epsilon)) * dout * gamma

        dshift2 = np.sum((input - mean) * dout * gamma, axis=0)
        dshift2 = (-1 / (var + epsilon)) * dshift2
        dshift2 = (0.5 / np.sqrt(var + epsilon)) * dshift2
        dshift2 = (2 * (input - mean) / N) * dshift2

        dshift = dshift1 + dshift2

        dx1 = dshift
        dx2 = -1 / N * np.sum(dshift, axis=0)
        dx = dx1 + dx2

        return dx

class BatchNorm2d(BatchNorm):
    def __init__(self, dim, epsilon=1e-5, momentum=0.9):
        super(BatchNorm2d, self).__init__(dim, epsilon, momentum)

    def forward(self, input):
        N, C, H, W = input.shape
        output = super(BatchNorm2d, self).forward(input.reshape(N * H * W, C))
        return output.reshape((N, C, H, W))

    def backward(self, dout):
        N, C, H, W = dout.shape
        dx = super(BatchNorm2d, self).backward(dout.reshape(N * H * W, C))
        return dx.reshape((N, C, H, W))


class Dropout(Layer):
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.p = p
        self.mode = "train"

    def forward(self, input):
        p, mode = self.p, self.mode

        if mode == 'train':
            mask = np.random.choice([0, 1], size=input.shape, p=[p, 1 - p])
            output = input * mask / (1 - p)
            self.cache['mask'] = mask
        else:
            output = input

        return output

    def backward(self, dout):
        p, mask = self.p, self.cache['mask']
        dx = dout * mask / (1 - p)

        return dx