from Layer import Linear
from Loss import SoftmaxCrossEntropyLoss
from Network import *

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

def eval_accuracy(pred, target):
    target = np.reshape(target, (target.shape[0]))
    correct = np.sum(pred == target)
    accuracy = correct / pred.shape[0] * 100
    return accuracy

layers = [Linear(4, 2)]
loss = SoftmaxCrossEntropyLoss()

fully_connected_network = Network(layers, loss, 1e-2)

for i in range(500):
    indeces = np.random.choice(X_train.shape[0], 4)
    batch = X_train[indeces, :]
    target = y_train[indeces]

    pred, loss = fully_connected_network.train(batch, target)

    if (i+1) % 25 == 0:
        accuracy = eval_accuracy(pred, target)
        print("Training Accuracy: %f" % accuracy)

    if (i+1) % 100 == 0:
        accuracy = eval_accuracy(fully_connected_network.eval(X_val), y_val)
        print("Validation Accuracy: %f" % accuracy)
        print()

accuracy = eval_accuracy(fully_connected_network.eval(X_test), y_test)
print("Test Accuracy: %f" % accuracy)