import numpy as np


class Activation:
    # multi-class classification
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    # binary classification
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # return activation function from dict
    def get_activation_function(self, activation):
        activation_dict = {
            "softmax": lambda x: self.softmax(x),
            "sigmoid": lambda x: self.sigmoid(x)
        }

        return activation_dict[activation]

    def get_derivative(self, activation):
        derivative_dict = {
            # "softmax": lambda x: self.softmax(x),
            "sigmoid": lambda x: self.sigmoid_derivative(x)
        }

        return derivative_dict[activation]