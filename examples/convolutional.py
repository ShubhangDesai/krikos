from krikos.data.loader import *
from krikos.nn.layer import Convolutional, Flatten, ReLU, BatchNorm2d, MaxPooling
from krikos.nn.network import Sequential
from krikos.nn.regularization import L2

from krikos.data.utils import *
from krikos.nn.loss import SoftmaxCrossEntropy

loader = CIFAR10Loader(batch_size=16)

layers = [Convolutional(3, 5, 4, stride=2),
          ReLU(),
          BatchNorm2d(5),
          MaxPooling(2, stride=1),

          Convolutional(5, 7, 4, stride=2),
          ReLU(),
          BatchNorm2d(7),
          MaxPooling(2, stride=1),

          Convolutional(7, 10, 5, stride=1),
          Flatten()]
loss = SoftmaxCrossEntropy()
conv_network = Sequential(layers, loss, 1e-3, regularization=L2(0.01))

for i in range(10000):
    batch, labels = loader.get_batch()

    pred, loss = conv_network.train(batch, labels)

    if (i + 1) % 100 == 0:
        accuracy = eval_accuracy(pred, labels)
        print("Training Accuracy: %f" % accuracy)

    if (i + 1) % 500 == 0:
        accuracy = eval_accuracy(conv_network.eval(loader.validation_set), loader.validation_labels)
        print("Validation Accuracy: %f \n" % accuracy)

accuracy = eval_accuracy(conv_network.eval(loader.test_set), loader.test_labels)
print("Test Accuracy: %f \n" % accuracy)