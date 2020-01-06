import numpy as np
import pickle

class Loader(object):
    def __init__(self, batch_size):
        super(Loader, self).__init__()
        self.batch_size = batch_size

        self.train_set, self.train_labels = None, None
        self.validation_set, self.validation_labels = None, None
        self.test_set, self.test_labels = None, None

    def load_data(self, path):
        raise NotImplementedError

    def get_batch(self):
        indeces = np.random.choice(self.train_set.shape[0], self.batch_size, replace=False)
        batch = np.array([self.train_set[i] for i in indeces])
        labels = np.array([self.train_labels[i] for i in indeces])

        return batch, labels


class CIFAR10Loader(Loader):
    def __init__(self, batch_size, path="datasets/cifar-10-batches-py"):
        super(CIFAR10Loader, self).__init__(batch_size)

        train, validation, test = self.load_data(path)
        self.train_set, self.train_labels = train
        self.validation_set, self.validation_labels = validation
        self.test_set, self.test_labels = test

        self.train_set, mean, std = self.preprocess(self.train_set)
        self.validation_set = (self.validation_set - mean)/std
        self.test_test = (self.test_set - mean)/std


    def load_data(self, path):
        train_set, train_labels = np.zeros((0, 3, 32, 32)), np.zeros((0))
        validation_set, validation_labels = None, None
        test_set, test_labels = None, None

        files = [path + "/data_batch_%d" % (i+1) for i in range(5)]
        files.append(path + "/test_batch")
        for file in files:
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')

            batch_set = dict[b"data"].reshape(10000, 3, 32, 32)
            batch_labels = np.array(dict[b"labels"]).reshape(10000)

            if "5" in file:
                validation_set, validation_labels = batch_set, batch_labels
            elif "test" in file:
                test_set, test_labels = batch_set, batch_labels
            else:
                train_set = np.concatenate((train_set, batch_set))
                train_labels = np.concatenate((train_labels, batch_labels))

        return (train_set, train_labels.astype(np.int32)), (validation_set, validation_labels.astype(np.int32)), (test_set, test_labels.astype(np.int32))

    def preprocess(self, set):
        mean, std = np.mean(set, axis=0), np.std(set, axis=0)
        set -= mean
        set /= std
        return set, mean, std