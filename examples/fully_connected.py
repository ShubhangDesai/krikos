import numpy as np
from krikos.nn.layer import Linear
from krikos.nn.network import Sequential
from krikos.nn.loss import SoftmaxCrossEntropy
import krikos.data.utils as utils

X = np.array([[1, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0],
              [0, 1, 1, 0], [1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0],
              [0, 0, 1, 1], [1, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0],
              [1, 1, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1]])
y = np.array([[0], [0], [0], [1], [1], [1], [0], [1], [1], [0], [1], [0], [1], [1], [0], [0]])

X_train = X[:8, :]
X_val = X[8:12, :]
X_test = X[12:16, :]
y_train = y[:8]
y_val = y[8:12]
y_test = y[12:16]

layers = [Linear(4, 2)]
loss = SoftmaxCrossEntropy()

fully_connected_network = Sequential(layers, loss, 1e-2)

for i in range(4000):
    indeces = np.random.choice(X_train.shape[0], 4)
    batch = X_train[indeces, :]
    target = y_train[indeces]

    pred, loss = fully_connected_network.train(batch, target)

    if (i+1) % 25 == 0:
        accuracy = utils.eval_accuracy(pred, target)
        print("Training Accuracy: %f" % accuracy)

    if (i+1) % 100 == 0:
        accuracy = utils.eval_accuracy(fully_connected_network.eval(X_val), y_val)
        print("Validation Accuracy: %f" % accuracy)
        print()

accuracy = utils.eval_accuracy(fully_connected_network.eval(X_test), y_test)
print("Test Accuracy: %f" % accuracy)