import math


def sigmoid(x):
    _lambda = 1.0
    exp = math.exp(-1 * _lambda * x)
    return 1.0 / (1 + exp)


def relu(x):
    return max(0, x)


functions = {
        'sigmoid': sigmoid,
        'relu': relu
        }
