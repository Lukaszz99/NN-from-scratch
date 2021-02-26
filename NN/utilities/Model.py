import numpy as np
import matplotlib.pyplot as plt
from Layer import Dense, Input, Output


class Model:
    def __init__(self):
        self._layers = []
        self._history = []

    def add(self, layer):
        self._layers.append(layer)

    def fit(self, X, y, epochs=1, batch_size=32, lr=0.001):
        self.X = X
        self.y = y

        for epoch in range(epochs):
            for batch_idx in range(batch_size):
                inp = [X]
                # forward propagation
                for layer in self._layers:
                    inp.append(layer._compute_layer_activation(inp[-1]))

                ###################
                # BACKPROPAGATION #
                ###################

                output_loss = Output()._compute_loss(inp[-1], y)
                layer_losses = [output_loss]

                for w_idx, layer_idx in zip(range(len(self._layers), 1, -1), (range(len(self._layers)-1, 0, -1))):
                    loss = layer_losses[-1]
                    weights = self._layers[w_idx - 1]._get_weights()
                    derivative = self._layers[layer_idx - 1]._get_derivative()
                    layer_loss = self._layers[layer_idx - 1]._compute_layer_loss(loss, weights, derivative)
                    layer_losses.append(layer_loss)


                # weights update
                for layer, activation_idx, loss in zip(self._layers[::-1], range(len(inp)-1,0 ,-1), layer_losses):
                    activation = inp[activation_idx-1]
                    layer._update_weights(activation, loss, lr)

                self._batch_output = inp[-1]

            self._history.append(self._compute_cost(self._batch_output))

    def _compute_cost(self, predict):
        term1 = -self.y * np.log(predict)
        term2 = (1 - self.y) * np.log(1 - predict)
        return np.sum(term1 - term2)

    def get_history(self):
        return self._history


if __name__ == '__main__':
    input_shape = (150, 5)  # 3 examples, 5 features

    X = np.random.uniform(0, 1, input_shape)
    y = np.random.randint(0, 2, (150, 2))
    print(y.shape)

    input_layer = Input(X)

    h1 = Dense(10, input_layer, name='h1')
    h2 = Dense(5, h1, name='h2')
    h3 = Dense(2, h2, name='h3')


    model = Model()

    model.add(h1)
    model.add(h2)
    model.add(h3)

    epochs = 100
    batch_size = 1
    lr = 0.001
    model.fit(X, y, epochs, batch_size, lr)

    # wykresy
    hist = model.get_history()
    plt.plot(range(epochs), hist)
    plt.show()
