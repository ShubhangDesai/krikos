from krikos.data.loader import Loader
import numpy as np

class RecurrentTestLoader(Loader):
    def __init__(self, batch_size):
        super(RecurrentTestLoader, self).__init__(batch_size)

        train, validation, test = self.load_data()
        self.train_set, self.train_labels = train
        self.validation_set, self.validation_labels = validation
        self.test_set, self.test_labels = test

    def load_data(self, path=None):
        timeseries = np.random.randint(1, 10, size=(16000, 2, 1))

        targets = timeseries[:, 0] - timeseries[:, 1]
        neg, pos = np.where(targets <= 0), np.where(targets > 0)
        targets[neg], targets[pos] = 0, 1

        timeseries_train = timeseries[:8000, :]
        timeseries_val = timeseries[8000:12000, :]
        timeseries_test = timeseries[12000:16000, :]

        targets_train = targets[:8000]
        targets_val = targets[8000:12000]
        targets_test = targets[12000:16000]

        return (timeseries_train, targets_train.astype(np.int32)), (timeseries_val, targets_val.astype(np.int32)), (timeseries_test, targets_test.astype(np.int32))