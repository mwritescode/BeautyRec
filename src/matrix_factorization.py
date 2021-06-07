import numpy as np

import matplotlib.pyplot as plt

from utils import FactorScheduler

class MatrixFactorization():
    """ Implements the Matrix Factorization algorithm developed by Simon Funk.

    More in detail, computes the predicted rating for user u and item i
    as μ + bi + bu + xu.T * yi, where μ is a global bias term, bi is the 
    bias asscociated with item i, bu is the bias asscociated with user u,
    xu is the vector of latent features characterizing user u and yi is the
    vector of latent features characterizing item i.
    Uses SGD to minimize the loss function L computed as the sum of the squared 
    errors plus an L2 regularization term.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Each row is a triplet of the form (product_id, ratings, user_id)
    num_users : int
        Number of unique users in the dataset
    num_items : int
        Number of unique products in the dataset
    num_latent_factors : int, optional
        Number of latent factors to compute
    
    Attributes
    ----------
    ratings : pandas.DataFrame
        Each row is a triplet of the form (product_id, ratings, user_id)
    user_bias: numpy.array
        Bias term for each user of the dataset
    item_bias: numpy.array
        Bias term for each item of the dataset
    global_bias: int
        Arithmentic mean of the input ratings
    U: numpy.array
        User latent factors matrix
    I: numpy.array
        Item latent factor matrix
    global_rmse: list
        List of the training errors for every epoch of training
    validation_rmse: list
        List of errors on the validation set for evry epoch of training
    val_failures: int
        Number of times in a row in which the current validation error is
        higher than the previous one. Used to implement early stopping 
    lr_scheduler: FactorScheduler, optional
        Learning rate scheduler to use during training. Is instantiated only
        when you pass lr_scheduler=True to the fit method.

    """
    def __init__(self, ratings, num_users, num_items,  num_latent_factors=12):

        self.ratings = ratings
        self.user_bias = np.zeros(num_users)
        self.item_bias = np.zeros(num_items)
        self.global_bias = np.mean(self.ratings.rating)
        self.U = np.random.random((num_users, num_latent_factors))
        self.I = np.random.random((num_latent_factors, num_items))
        self.global_rmse = []
        self.validation_rmse = []
        self.val_failures = 0
    
    def predict(self, user, item):
        """ Computes the predicted rating given a user id and an item id.

        Parameters
        ----------
        user : int
            User id
        item : int
            Item id

        Returns
        -------
        pred: int
            Predicted rating for given user and item

        """
        pred = self.global_bias + self.user_bias[user] + self.item_bias[item]
        pred += self.U[user, :].dot(self.I[:, item])
        return pred
    
    def _on_train_begins(self, val):
        """ Prints initial info when training starts.

        Parameters
        ----------
        val : numpy.array or None
            The validation set on which to check the 
            training progress

        """
        self.global_rmse.append(self._compute_rmse(self.ratings))
        header_string = '{} \t | \t {} \t '.format('Iteration', 'RMSE')
        num_dashes = 40
        if val is not None:
            header_string += ' \t | \t {}'.format('Validation RMSE')
            self.validation_rmse.append(self._compute_rmse(val))
            num_dashes = 70
        print(num_dashes*'-')
        print(header_string)
        print(num_dashes*'-')

    def train(self, max_iter=20, regularize=0.1, learning_rate=0.05, val=None, lr_scheduler=False):
        """ Starts the algorithm training using SGD.

        Also plots the training and validation errors when training is complete
        and, if lr_scheduler = True, also plots the changes in the learning rate.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of training epochs
        regularize : float, optional
            Parameter for the L2 regularization term
        learning_rate : float, optional
            SGD initial learning rate
        val : numpy.array, optional
            Eventual validation set on which the errors the rmse should be
            computed. If it is specified then the algorithm automatically implements 
            early stopping
        lr_scheduler : bool, optional
            Whether to use a learning rate scheduler to decrease the learning
            rate as the training goes on

        """
        self._on_train_begins(val)
        lr_accumulator = []
        if lr_scheduler:
            self.lr_scheduler = FactorScheduler(initial_lr=learning_rate)
        iteration = 0
        while iteration < max_iter and self.val_failures < 3:
            if lr_scheduler:
                learning_rate = self.lr_scheduler()
                lr_accumulator.append(learning_rate)
            self._sgd_step(regularize, learning_rate)
            result_msg = '{:.0f} \t\t | \t {:.4f} '.format(iteration +1, self.global_rmse[-1])
            if val is not None:
                self.validation_rmse.append(self._compute_rmse(val))
                result_msg += ' \t | \t {:.4f} '.format(self.validation_rmse[-1])
                if self.validation_rmse[-1] >= self.validation_rmse[-2]:
                    self.val_failures += 1
                else:
                    self.val_failures = 0
            iteration += 1
            print(result_msg)
        self._plot_rmse(val=val)
        self._plot_lr(lr_accumulator)
    
    def _sgd_step(self, regularize, learning_rate):
        """ Compute a single epoch of training.

        Parameters
        ----------
        regularize : float
            Parameter for the L2 regularization   
        learning_rate : float
            Learning rate to use for this epoch

        """
        ratings = self.ratings.sample(frac=1)
        num_ratings = len(ratings)
        sse = 0
        for row in ratings.itertuples(index=False):
            item, rating, user = row
            prediction = self.predict(user, item)
            err = rating - prediction
            sse +=  err ** 2
            self.user_bias[user] += learning_rate * (err - regularize*self.user_bias[user])
            self.item_bias[item] += learning_rate * (err - regularize*self.item_bias[item])
            self.U[user, :] += learning_rate * (err*self.I[:, item] - regularize*self.U[user, :])
            self.I[:, item] += learning_rate * (err*self.U[user, :] - regularize*self.I[:, item])
        self.global_rmse.append(np.sqrt(sse / num_ratings))

    def _plot_rmse(self, val=False):
        """ Plots the training loss per epoch.

        If val is not None also plots the validation loss per epoch.

        Parameters
        ----------
        val : numpy.array of None
            Eventual validation set on which to compute the loss

        """
        _, ax = plt.subplots()
        ax.plot(self.global_rmse, linewidth=3, color='blue', label='Train RMSE')
        ax.set_title('RMSE vs. Number of Iterations')
        if val is not None:
            ax.plot(self.validation_rmse, linewidth=3, color='green', label='Validation RMSE')
            ax.legend()
        plt.show()
    
    def _plot_lr(self, lr_accumulator):
        """ Plots the learning rate per epoch.

        Parameters
        ----------
        lr_accumulator : list
            The learning rates used by SGD at each step, as produced
            the FactorScheduler

        """
        if lr_accumulator:
            plt.plot(lr_accumulator, linewidth=3, color='green')
            plt.title('Learning rate per epoch.')
            plt.show()

    def _predict_all(self, data):
        """ Predicts the ratings for a list of (user_id, item_id).

        Parameters
        ----------
        data : pandas.DataFrame
            Validation data for which to predict the ratings

        Returns
        -------
        preds : numpy.array
            Predicted ratings for the given user and item ids

        """
        preds = np.zeros(len(data))
        for row in data.itertuples():
            index, item, _, user = row
            preds[index] = self.predict(user, item)
        return preds
    
    def _compute_rmse(self, data):
        """ Computes the RMSE for the given data points.

        Parameters
        ----------
        data : pandas.DataFrame
            Triplets (user_id, rating, item_id)

        Returns
        -------
        rmse : float

        """
        actual = data.rating.values
        pred = self._predict_all(data)
        rmse = np.sqrt(np.sum((actual - pred) **2) /len(pred))
        return rmse
