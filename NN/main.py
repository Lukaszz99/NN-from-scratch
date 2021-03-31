from utilities.DataSetReader import load_mnist
from utilities.Layer import Dense, Input
from utilities.Model import Model

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt


def load_data(train_size=5000, val_size=500, test_size=500):
    path = 'E:\\VI Sem\\MN\\lab\\NN\\data'
    images, labels = load_mnist(path, 't10k')

    if (train_size + val_size + test_size) >= 10000:
        print(f'Sum of sizes is too big - {train_size+val_size+test_size}. Should be less than 10000')
    else:
        X_train = images[:train_size]
        labels_ohe = OneHotEncoder().fit_transform(labels.reshape(-1, 1)).toarray()
        y_train = labels_ohe[:train_size]

        X_val = images[-val_size:]
        y_val = labels_ohe[-val_size:]

        X_test = images[train_size:train_size + test_size]
        y_test = labels[train_size:train_size + test_size]

        return X_train, y_train, X_val, y_val, X_test, y_test

def plot_history(epochs, hist, name):
    # plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(2)
    desc = f"{hist['desc']}, LR={hist['lr']}"
    fig.suptitle(desc)
    ax[0].plot(range(epochs), hist['loss'], label='Train loss')
    ax[0].plot(range(epochs), hist['val_loss'], label='Val_loss')
    ax[0].set(ylabel='Loss')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(range(epochs), hist['accuracy'], label='Train accuracy')
    ax[1].plot(range(epochs), hist['val_accuracy'], label='Val_accuracy')
    ax[1].set(xlabel='epoch', ylabel='Accuracy')
    ax[1].legend()
    ax[1].grid()

    # plt.tight_layout()
    plt.savefig(name)
    plt.show()

if __name__ == '__main__':

    X_train, y_train, X_val, y_val, X_test, y_test = load_data(2000, 100, 10)

    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_val = (X_val - np.mean(X_val)) / np.std(X_val)

    # Create model
    model = Model()

    # Create layers
    input_layer = Input(X_train)
    h1 = Dense(128, input_layer, name='h1')
    h2 = Dense(64, h1, name='h2')
    # h3 = Dense(20, h2, name='h3')
    h4 = Dense(10, h2, name='output')

    model.add(h1)
    model.add(h2)
    # model.add(h3)
    model.add(h4)

    epochs = 150
    batch_size = 32
    lr = 0.001

    # model.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val, epochs=epochs, batch_size=batch_size, lr=lr)

    # predict = model.predict(X_test)
    # print(predict)

    print(y_test)

    # plot_history(epochs, model.get_history(), name="128_64_001")
