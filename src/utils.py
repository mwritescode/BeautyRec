import numpy as np

import matplotlib.pyplot as plt

class FactorScheduler():
    def __init__(self, alpha=0.9, initial_lr=0.01, min_lr=1e-7):
        self.alpha = alpha
        self.lr = initial_lr
        self.min_lr = min_lr
    
    def __call__(self):
        self.lr = max(self.min_lr, self.lr * self.alpha)
        return self.lr

class MMSE():
    def compute(self, y, y_pred):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        mask = np.where(y == 0, 0, 1)
        loss = mask * (y - y_pred)**2 / np.sum(mask)
        return loss
    
    def grad(self, y, y_pred):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        mask = np.where(y == 0, 0, 1)
        return -2*mask*(y-y_pred) / np.sum(mask)

class SquareLoss():

    def compute(self, y, y_pred):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        return 0.5 * np.power((y - y_pred), 2)

    def grad(self, y, y_pred):
        if len(y.shape) == 1:
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
    ax.set_title('Loss value per epoch')
    y_lim = train_err[1]
    ax.plot(train_err, color='b', label='Training')
    if val_err is not None:
        ax.plot(val_err, color='g', label='Validation')
        y_lim = val_err[1]
    ax.set_ylim(bottom=0, top=y_lim)
    plt.legend()
    plt.show()

def plot_learning_rates(weights, bias, n_unit):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,8))
    title = fig.suptitle(
        'Example of weights and bias learning rates for neuron {} of layer 0'. format(n_unit + 1 ),
        fontsize="x-large")
    title.set_y(0.95)
    fig.subplots_adjust(top=0.85)
    ax1.plot(weights, color='b', linewidth=3)
    ax1.set_title('Weight lr')
    ax2.plot(bias, color='g', linewidth=3)
    ax2.set_title('Bias lr')
    plt.show()