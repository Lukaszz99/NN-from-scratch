import numpy as np
import matplotlib.pyplot as plt
from .Layer import Dense, Input, Output

from math import floor


class Model:
    def __init__(self):
        self._layers = []
        self._history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'lr': 0,
            'desc': ''
        }

    def add(self, layer):
        self._layers.append(layer)

    def fit(self, X, y, X_val=None, y_val=None, epochs=1, batch_size=32, lr=0.001):
        for epoch in range(epochs):
            steps = floor(X.shape[0] / batch_size)

            self._lr = lr

            for step in range(steps):
                # print(f'Epoch: {epoch+1} Step: {step+1}')
                start_idx = batch_size * step
                stop_idx = batch_size * step + batch_size

                inp = X[start_idx:stop_idx]
                self.y = y[start_idx: stop_idx]


                # forward propagation
                network_output = self._compute_forward(inp)

                ###################
                # BACKPROPAGATION #
                ###################

                output_loss = Output()._compute_loss(network_output[-1], self.y)
                layer_losses = [output_loss]

                for w_idx, layer_idx in zip(range(len(self._layers), 1, -1), (range(len(self._layers)-1, 0, -1))):
                    loss = layer_losses[-1]
                    weights = self._layers[w_idx - 1]._get_weights()
                    derivative = self._layers[layer_idx - 1]._get_derivative()
                    layer_loss = self._layers[layer_idx - 1]._compute_layer_loss(loss, weights, derivative)
                    layer_losses.append(layer_loss)


                # weights update
                for layer, activation_idx, loss in zip(self._layers[::-1], range(len(network_output)-1,0 ,-1), layer_losses):
                    activation = network_output[activation_idx-1]
                    layer._update_weights(activation, loss, self._lr)

                self._batch_output = network_output[-1]

            self._history['loss'].append(self._compute_cost(self._batch_output, self.y))
            self._history['accuracy'].append(self._compute_accuracy(self.predict(X[:100]), y[:100]))

            if X_val.all():
                self._history['val_loss'].append(self._compute_cost(self._compute_forward(X_val)[-1], y_val))
                self._history['val_accuracy'].append(self._compute_accuracy(self.predict(X_val), y_val))

        self._update_history()

    def _compute_forward(self, input_array):
        output = [input_array]
        for layer in self._layers:
            output.append(layer._compute_layer_activation(output[-1]))
        return output

    def _compute_cost(self, predict, y):
        term1 = -y * np.log(predict)
        term2 = (1 - y) * np.log(1 - predict)
        return np.sum(term1 - term2)

    def predict(self, X):
        output = self._compute_forward(X)[-1]
        predicted = np.argmax(output, axis=1)
        return predicted

    def _compute_accuracy(self, pred_y, y):
        # y powinien być typu np.array 1D
        value_y = np.argmax(y, axis=1)
        return np.sum(pred_y == value_y) / value_y.shape[0]

    def get_history(self):
        return self._history

    def _update_history(self):
        self._history['lr'] = self._lr
        self._history['desc'] = f'Hidden layers: {len(self._layers)-1} {[layer._units for layer in self._layers[:-1]]}'
