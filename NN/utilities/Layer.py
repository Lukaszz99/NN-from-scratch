import numpy as np
from Activation import Activation


class Dense:
    def __init__(self, units, input_layer, activation="sigmoid", name=''):
        """

        :param units: units in this layer
        :param input_layer: layer before this layer
        :param activation: activation function for neurons in this layer
        """

        # units in layer
        self._units = units

        # units in layer
        self._input_layer = input_layer

        self._activation = Activation().get_activation_function(activation)
        self._derivative = Activation().get_derivative(activation)

        self._name = name

        # weights matrix: [units_before, units_in_layer]
        self._weights = np.random.uniform(-0.01, 0.01, (input_layer._get_input_cols(), units))

        # bias in layer
        self._bias = np.random.random()

        self._layer_activation = 0

    def _get_input_cols(self):
        return self._weights.shape[1]

    def _compute_layer_activation(self, input_matrix):
        """

        :return: layer's activation
        """
        # input_matrix = self._input_layer._get_input()

        z = np.dot(input_matrix, self._weights) + self._bias

        self._layer_activation = self._activation(z)

        return self._layer_activation

    def _get_derivative(self):
        return self._derivative(self._layer_activation)

    def _get_weights(self):
        return self._weights

    def _compute_layer_loss(self, loss, weights, derivative):
        return np.dot(loss, weights.T) * derivative

    def _update_weights(self, activation, loss, lr):
        # print(self._weights.shape)
        self._weights -= lr * np.dot(activation.T, loss)
        self._bias -= lr * np.sum(loss, axis=0)

class Input:
    def __init__(self, X):
        """

        :param X: Training set for model
        """
        self._X = X
        self._input_shape = (X.shape[0], X.shape[1])

    def _get_input(self):
        return self._X

    def _get_input_cols(self):
        return self._X.shape[1]


class Output:
    def __init__(self, l2=1.0):  # jeszcze trzeba dodac w agrumencie wagi ze wszystkich warstw
        self._l2 = l2

    def _compute_loss(self, predict, y):
        loss = (predict - y)

        return loss
