import numpy as np
import math

EPS = 10 ** (-8)

class RMSProp():
    def __init__(self, learning_rate=0.01, gamma=0.9):
        self.lr = learning_rate
        self.gamma = gamma
        self.s_w = None
        self.s_bias = None
        self.reference_index = 0
        self.learning_rates = {
            'weights': [],
            'bias': []
        }

    def update_weights(self, weights, grad):
        if self.s_w is None:
            self.s_w = np.zeros(weights.shape)
        self.s_w = self.gamma * self.s_w + (1 - self.gamma) * (grad**2)
        lr = self.lr / np.sqrt(self.s_w + EPS)
        self._save_one_learning_rate(lr, weights)
        W = weights - lr*grad
        return W
    
    def update_bias(self, bias, grad):
        if self.s_bias is None:
            self.s_bias = np.zeros(bias.shape)
        self.s_bias = self.gamma * self.s_bias + (1 - self.gamma) * (grad**2)
        lr = self.lr / np.sqrt(self.s_bias + EPS)
        self._save_one_learning_rate(lr, bias, type='bias')
        b = bias - lr*grad
        return b
    
    def _save_one_learning_rate(self, lr, params, type='weights'):
        lr = lr.ravel()
        if not self.learning_rates[type]:
            params = params.ravel()
            self.raference_index = np.random.randint(0, len(params))
        self.learning_rates[type].append(lr[self.reference_index])

class Adam():
    def __init__(self, learning_rate=0.01, b1=0.9, b2=0.99):
        self.lr = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.v_W = None
        self.v_bias = None
        self.s_W = None
        self.s_bias = None
        self.reference_index = 0
        self.learning_rates = {
            'weights': [],
            'bias': []
        }

    def update_weights(self, weights, grad):
        if self.s_W is None:
            self.v_W = np.zeros(weights.shape)
            self.s_W = np.zeros(weights.shape)
        
        self.v_W = self.b1 * self.v_W + (1- self.b1) * grad
        self.s_W = self.b2 * self.s_W + (1 - self.b2) * (grad**2)

        v_hat = self.v_W / (1 - self.b1)
        s_hat = self.s_W / (1 - self.b2)
        
        update = self.lr * v_hat / (np.sqrt(s_hat) + EPS)
        return weights - update
    
    def update_bias(self, bias, grad):
        if self.s_bias is None:
                self.v_bias = np.zeros(bias.shape)
                self.s_bias = np.zeros(bias.shape)
            
        self.v_bias = self.b1 * self.v_bias + (1- self.b1) * grad
        self.s_bias = self.b2 * self.s_bias + (1 - self.b2) * (grad**2)

        v_hat = self.v_bias / (1 - self.b1)
        s_hat = self.s_bias / (1 - self.b2)
        
        update = self.lr * v_hat / (np.sqrt(s_hat) + EPS)
        return bias - update