import numpy as np


def step_function(x):
    if x >= 1:
        return 1
    return 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def linear(x):
    return x


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


test = step_function(-30)
test_sigmoid = sigmoid(2.1)
test_tanh = tanh(2.1)
test_relu = relu(2.1)
test_linear = linear(2.1)
test_softmax = softmax([5, 2, 1.3])

print(f"Step Function: {test}")
print(f"Sigmoid: {test_sigmoid}")
print(f"Tanh: {test_tanh}")
print(f"Relu: {test_relu}")
print(f"Linear: {test_linear}")
print(f"Softmax: {test_softmax}")
0.28
