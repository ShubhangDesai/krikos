import numpy as np

def eval_accuracy(pred, target):
    target = np.reshape(target, (target.shape[0]))
    correct = np.sum(pred == target)
    accuracy = correct / pred.shape[0] * 100
    return accuracy