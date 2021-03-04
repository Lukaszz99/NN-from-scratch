from utilities.DataSetReader import load_mnist
from utilities.Layer import Dense, Input
from utilities.Model import Model

from sklearn.preprocessing import OneHotEncoder

import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = 'E:\\VI Sem\\MN\\lab\\NN\\data'
    images, labels = load_mnist(path, 't10k')

    size = 6000
    val_size = 50
    X = images[:size]
    labels_ohe = OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()
    y = labels_ohe[:size]

    X_val = images[-val_size:]
    y_val = labels_ohe[-val_size:]

    # Create model
    model = Model()

    # Create layers
    input_layer = Input(X)
    h1 = Dense(50, input_layer, name='h1')
    h2 = Dense(10, h1, name='output')

    model.add(h1)
    model.add(h2)

    epochs = 500
    batch_size = 32
    lr = 0.01

    model.fit(X=X, y=y, X_val=X_val, y_val=y_val, epochs=epochs, batch_size=batch_size, lr=lr)

    predict = model.predict(X_val)

    # plt.plot(range(epochs), model.get_history()['loss'], label='loss')
    # plt.plot(range(epochs), model.get_history()['val_loss'], label='val_loss')
    plt.plot(range(epochs), model.get_history()['accuracy'], label='accuracy')
    plt.plot(range(epochs), model.get_history()['val_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()
