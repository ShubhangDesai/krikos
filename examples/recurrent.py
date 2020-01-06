from krikos.nn.layer import VanillaRNN, LSTM, Flatten, Linear, ReLU
from krikos.nn.regularization import L2
from krikos.nn.loss import SoftmaxCrossEntropy
from krikos.nn.network import Sequential
from krikos.data.utils import eval_accuracy
from examples.RecurrentTestLoader import RecurrentTestLoader

loader = RecurrentTestLoader(16)

layers = [LSTM(1, 4),
          ReLU(),
          Flatten(),
          Linear(8, 2)]
loss = SoftmaxCrossEntropy()
recurrent_network = Sequential(layers, loss, 1e-3, regularization=L2(0.01))

for i in range(10000):
    batch, labels = loader.get_batch()

    pred, loss = recurrent_network.train(batch, labels)

    if (i + 1) % 100 == 0:
        accuracy = eval_accuracy(pred, labels)
        #print(loss)
        print("Training Accuracy: %f" % accuracy)

    if (i + 1) % 500 == 0:
        accuracy = eval_accuracy(recurrent_network.eval(loader.validation_set), loader.validation_labels)
        print("Validation Accuracy: %f \n" % accuracy)

accuracy = eval_accuracy(recurrent_network.eval(loader.test_set), loader.test_labels)
print("Test Accuracy: %f \n" % accuracy)