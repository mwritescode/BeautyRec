import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from data_prep import build_user_ratings_matrix, remove_nicknames

class MatrixFactorization():
    def __init__(self, user_item_matrix, num_latent_factors=12):
        self.user_item_matrix = user_item_matrix
        self._n_users, self._n_items = self.user_item_matrix.shape
        self.user_bias = np.zeros(self._n_users)
        self.item_bias = np.zeros(self._n_items)
        self.global_bias = np.mean(self.user_item_matrix[np.nonzero(self.user_item_matrix)])
        self.U = np.random.random((self._n_users, num_latent_factors))
        self.I = np.random.random((num_latent_factors, self._n_items))
    
    def predict(self, user, item):
        pred = self.global_bias + self.user_bias[user] + self.item_bias[item]
        pred += self.U[user, :].dot(self.I[:, item])
        return pred

    def train(self, max_iter=20, regularize=0.1, learning_rate=0.05):
        training_ratings = len(self.user_item_matrix.nonzero()[0])
        for i in range(max_iter):
            sse = 0
            print('step: ', i)
            sse = self._sgd_step(regularize, learning_rate, sse=sse)
            print('rms: ', sse / training_ratings)
    
    def _sgd_step(self, regularize, learning_rate, sse):
        valid_indexes = np.argwhere(self.user_item_matrix > 0)
        np.random.shuffle(valid_indexes)
        for user, item in valid_indexes:
            prediction = self.predict(user, item)
            err = self.user_item_matrix[user, item] - prediction
            sse += err ** 2
            self.user_bias[user] += learning_rate * (err - regularize*self.user_bias[user])
            self.item_bias[item] += learning_rate * (err - regularize*self.item_bias[item])
            self.U[user, :] += learning_rate * (err*self.I[:, item] - regularize*self.U[user, :])
            self.I[:, item] += learning_rate * (err*self.U[user, :] - regularize*self.I[:, item])
        return sse

    #def _predict_all(self):
    #    preds = np.zeros(self.user_item_matrix.shape)
    #    for user in range(self._n_users):
    #        for item in range(self._n_items):
    #            preds[user, item] = self.predict(user, item)
    #    return preds
    
    #def _compute_rmse(self):
    #    actual = self.user_item_matrix[self.user_item_matrix.nonzero()]
    #    pred = self._predict_all()
    #    pred = pred[self.user_item_matrix.nonzero()]
    #    return np.sqrt(np.square(np.subtract(actual, pred))/len(pred))

ratings = pd.read_csv('../data/ratings.csv', sep='\t')
prods = pd.read_csv('../data/products.csv', sep='\t')
ratings = remove_nicknames(ratings)
n_users = len(ratings.buyer_id.unique())
n_items = len(prods.id.unique())
user_item_matrix = build_user_ratings_matrix(n_items, n_users, ratings)
model = MatrixFactorization(user_item_matrix, num_latent_factors=15)
model.train(max_iter=25, learning_rate=0.001)