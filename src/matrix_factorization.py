import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import FactorScheduler, train_test_split
from data_prep import remove_nicknames

class MatrixFactorization():
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
        pred = self.global_bias + self.user_bias[user] + self.item_bias[item]
        pred += self.U[user, :].dot(self.I[:, item])
        return pred
    
    def _on_train_begins(self, val):
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
        self._plot_rmse(iterations=range(iteration +1), val=val)
        self._plot_lr(lr_accumulator)
    
    def _sgd_step(self, regularize, learning_rate):
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

    def _plot_rmse(self, iterations, val=False):
        _, ax = plt.subplots()
        ax.plot(iterations, self.global_rmse, linewidth=3, color='blue', label='Train RMSE')
        ax.set_title('RMSE vs. Number of Iterations')
        if val is not None:
            ax.plot(iterations, self.validation_rmse, linewidth=3, color='green', label='Validation RMSE')
            ax.legend()
        plt.show()
    
    def _plot_lr(self, lr_accumulator):
        if lr_accumulator:
            plt.plot(lr_accumulator, linewidth=3, color='green')
            plt.title('Learning rate per epoch.')
            plt.show()

    def _predict_all(self, data):
        preds = np.zeros(len(data))
        for row in data.itertuples():
            index, item, _, user = row
            preds[index] = self.predict(user, item)
        return preds
    
    def _compute_rmse(self, data):
        actual = data.rating.values
        pred = self._predict_all(data)
        return np.sqrt(np.sum((actual - pred) **2) /len(pred))
