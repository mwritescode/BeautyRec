import numpy as np

import matplotlib.pyplot as plt

class FactorScheduler():
    """ Implements a learning rate scheduler.

        Each time it's called it decreases the initial learning rate
        by a fixed factor alpha.

        Parameters
        ----------
        alpha : float, optional
            Decreasing factor for the learning rate
        initial_lr : float, optional
            Initial learning rate
        min_lr : float, optional
            Minimul value that the learning rate can reach

        Arguments
        ---------
        alpha : float, optional
            Decreasing factor for the learning rate
        lr : float, optional
            Current learning rate
        min_lr : float, optional
            Minimul value that the learning rate can reach

    """
    def __init__(self, alpha=0.9, initial_lr=0.01, min_lr=1e-7):

        self.alpha = alpha
        self.lr = initial_lr
        self.min_lr = min_lr
    
    def __call__(self):

        self.lr = max(self.min_lr, self.lr * self.alpha)
        return self.lr

class MMSE():

    def compute(self, y, y_pred):
        """ Computes the Masked Mean Squared Error (MMSE).

        Parameters
        ----------
        y : numpy.array
            Actual values
        y_pred : numpy.array
            Predicted values

        Returns
        -------
        loss : numpy.array
            Loss value for each of the values in (y, y_pred)

        """
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        mask = np.where(y == 0, 0, 1)
        loss = mask * (y - y_pred)**2 / np.sum(mask)
        return loss
    
    def grad(self, y, y_pred):
        """ Computes the gradient of the MMSE with respect to the predicted values.

        Parameters
        ----------
        y : numpy.array
            Actual values
        y_pred : numpy.array
            Predicted values

        Returns
        -------
        grad: numpy.array
            Gradient of the MMSE with respect to y_pred

        """
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        mask = np.where(y == 0, 0, 1)
        grad = -2*mask*(y-y_pred) / np.sum(mask)
        return grad

class SquareLoss():

    def compute(self, y, y_pred):
        """ Computes the Square Loss.

        Parameters
        ----------
        y : numpy.array
            Actual values
        y_pred : numpy.array
            Predicted values

        Returns
        -------
        loss : numpy.array
            Loss value for each of the values in (y, y_pred)

        """
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        loss = 0.5 * np.power((y - y_pred), 2)
        return loss

    def grad(self, y, y_pred):
        """ Computes the gradient of the Square Loss with respect to the predicted values.

        Parameters
        ----------
        y : numpy.array
            Actual values
        y_pred : numpy.array
            Predicted values

        Returns
        -------
        grad: numpy.array
            Gradient of the Square Loss with respect to y_pred

        """
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        grad = -(y - y_pred)
        return grad

def train_test_split(data, test_prc=0.25, random_seeed=42):
    """ Splits the data in train and test sets.

    Parameters
    ----------
    data : pandas.DataFrame
        The initial dataset
    test_prc : float, optional
        Percentage of data to be included in the test set
    random_seeed : int, optional
        Random seed for reproducibility

    Returns
    -------
    train_split : numpy.array
        The training set
    test_split : numpy.array
        The test set

    """
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
    """ Plots training loss and, if present, validation loss.

    Parameters
    ----------
    train_err : list
        List of errors on the training set
    val_err : list, optional
        List of errors on the validation set

    """
    _, ax = plt.subplots()
    ax.set_title('Loss value per epoch')
    ax.plot(train_err, color='b', label='Training')
    if val_err is not None:
        ax.plot(val_err, color='g', label='Validation')
    plt.legend()
    plt.show()

def plot_learning_rates(weights, bias, n_unit):
    """ Plot the learning rate changes for one weight and one bias.

    Parameters
    ----------
    weights : list
        Learning rate changes per epoch for a weight
    bias : list
        learning rate changes for epoch for one bias
    n_unit : int
        Index of the neuron from which the learning rate changes 
        have been recorded

    """
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