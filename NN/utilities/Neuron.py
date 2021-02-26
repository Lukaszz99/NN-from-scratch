import numpy as np
from Activation import Activation

from utilities.Activation import Activation


class Neuron:
    def __init__(self, input, weights, activation="sigmoid"):
        self.activation = Activation().get_activation_function(activation)

        # input matrix: [n_examples, n_features]
        self.input = input

        # weights matrix: [1, n_examples + 1]. +1 because first element is a bias
        self.weights = weights

    def calc(self):
        z = np.dot(self.input, self.weights[1:]) + self.weights[0]
        return self.activation(z)



print(Neuron([[0.5, 0.2], [1, 0.2]], [[1.0], [0.5], [4.0]]).calc())
