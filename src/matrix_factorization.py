import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import train_test_split
from data_prep import remove_nicknames

class MatrixFactorization():
    def __init__(self, ratings, num_users, num_items,  num_latent_factors=12):
        self.ratings = ratings
        self._n_users = num_users
        self._n_items = num_items
        self.user_bias = np.zeros(self._n_users)
        self.item_bias = np.zeros(self._n_items)
        self.global_bias = np.mean(self.ratings.rating)
        self.U = np.random.random((self._n_users, num_latent_factors))
        self.I = np.random.random((num_latent_factors, self._n_items))
        self.global_rmse = []
        self.validation_rmse = []
    
    def predict(self, user, item):
        pred = self.global_bias + self.user_bias[user] + self.item_bias[item]
        pred += self.U[user, :].dot(self.I[:, item])
        return pred

    def train(self, max_iter=20, regularize=0.1, learning_rate=0.05, val=None):
        self.global_rmse.append(self._compute_rmse(self.ratings))
        header_string = '{} \t | \t {} \t '.format('Iteration', 'RMSE')
        num_dashes = 40
        if val is not None:
            header_string += ' \t | \t {}'.format('Validation RMSE')
            self.validation_rmse.append(self._compute_rmse(val))
            num_dashes = 70
            fails = 0
        print(num_dashes*'-')
        print(header_string)
        print(num_dashes*'-')
        iterations = max_iter
        for i in range(max_iter):
            self._sgd_step(regularize, learning_rate)
            result_msg = '{:.0f} \t\t | \t {:.4f} '.format(i +1, self.global_rmse[-1])
            if val is not None:
                self.validation_rmse.append(self._compute_rmse(val))
                result_msg += ' \t | \t {:.4f} '.format(self.validation_rmse[-1])
                if self.validation_rmse[-1] >= self.validation_rmse[-2]:
                    fails += 1
                else:
                    fails = 0
            print(result_msg)
            if val is not None and fails >= 3:
                iterations = i + 1
                break    
        self._plot_rmse(iterations=range(iterations +1), val=val)
    
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

ratings = pd.read_csv('../data/ratings.csv', sep='\t')
ratings = remove_nicknames(ratings)
num_users = len(ratings.buyer_id.unique())
num_items = ratings.product_id.max() + 1
train, val = train_test_split(ratings)
model = MatrixFactorization(train, num_users, num_items, num_latent_factors=20)
model.train(max_iter=20, learning_rate=0.01, regularize=0.5, val=val)