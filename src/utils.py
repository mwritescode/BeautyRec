import numpy as np

import matplotlib.pyplot as plt

class SquareLoss():

    def compute(self, y, y_pred):
        y = y.reshape((-1, 1))
        return 0.5 * np.power((y - y_pred), 2)

    def grad(self, y, y_pred):
        y = y.reshape((-1, 1))
        return -(y - y_pred)

def train_test_split(data, test_prc=0.25, random_seeed=42):
    data_size = len(data)
    test_size = int(data_size * test_prc)
    np.random.seed(random_seeed)
    test_indexes = np.random.randint(low=0, 
                                    high=data_size, 
                                    size=test_size)
    test_split = data[data.index.isin(test_indexes)].copy().reset_index(drop=True)
    train_split = data[~data.index.isin(test_indexes)].copy().reset_index(drop=True)
    return train_split, test_split

def plot_losses(train_err, val_err = None):
    _, ax = plt.subplots()
    ax.plot(train_err, color='b', label='Training', linewidth=3)
    if val_err is not None:
        ax.plot(val_err, color='g', label='Validation', linewidth=3)
    plt.legend()
    plt.show()

def plot_predictions(actual, predicted):
    _, ax = plt.subplots()
    ax.plot(actual, label='Actual values', color='b', linewidth=3)
    ax.plot(predicted, label='Predicted values', color='g', linewidth=3)
    plt.legend()
    plt.show()