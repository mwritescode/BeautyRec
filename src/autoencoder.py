import numpy as np

import copy

from layers import DenseLayer, LeakyReLU
from utils import MMSE

class SequentialModel():
    def __init__(self, layers=None):
        self.layers = self._check_input_sizes(layers)
        self.optimizer = None
        self.loss = None
        self.errors = {
            'training': [],
            'validation': []
        }
    
    def _check_input_sizes(self, layers):
        if layers is not None:
            for i, layer in enumerate(layers):
                if i > 0:
                    layer.set_input_shape(layers[i-1].get_output_shape())
                layer.initialize_params()
        return layers
    
    def add(self, layer):
        if self.layers:
            layer.set_input_shape(self.layers[-1].get_output_shape())
            layer.initialize_params()
        self.layers.append(layer)
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        for layer in self.layers:
            layer.compile(copy.deepcopy(optimizer))
        self.loss = loss
    
    def _on_train_begins(self, val):
        header_string = '{} \t | \t {} \t '.format('Iteration', 'Train Loss')
        num_dashes = 40
        if val is not None:
            header_string += ' \t | \t {}'.format('Validation Loss')
            num_dashes = 80
        print(num_dashes*'-')
        print(header_string)
        print(num_dashes*'-')
    
    def _predict_val_loss(self, val_set, fails, early_stopping):
        x_val, y_val = val_set
        y_val_pred = self.predict(x_val)
        val_loss = np.mean(self.loss.compute(y_val, y_val_pred))
        self.errors['validation'].append(val_loss)
        result_msg = ' \t\t | \t {:.4f} '.format(val_loss)
        if early_stopping:
            if val_loss > self.errors['validation'][-1]:
                fails += 1
            else:
                fails = 0
        return fails, result_msg

    def fit(self, x, y, num_epochs=100, val_set=None, early_stopping=False, patience=3):
        fails = 0
        iteration = 0
        self._on_train_begins(val_set)
        while iteration < num_epochs and fails < patience:
            y_pred = self._forward_pass(x)
            loss = np.mean(self.loss.compute(y, y_pred))
            result_msg = '{:.0f} \t\t | \t {:.4f} '.format(iteration +1, loss)
            self.errors['training'].append(loss)
            if val_set is not None:
                fails, new_msg = self._predict_val_loss(val_set, fails, early_stopping)
                result_msg += new_msg
            loss_grad = self.loss.grad(y, y_pred)
            print(result_msg)
            self._backward_pass(loss_grad)
            iteration += 1
        return self.errors
    
    def predict(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def _forward_pass(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def _backward_pass(self, accumulated_grad):
        for layer in reversed(self.layers):
            accumulated_grad = layer.backward_pass(accumulated_grad)
    
    def print_summary(self):
        print(' ' + 80*'-')
        empty_line = 3*('|' + 26*' ')+ '|'
        
        for layer in self.layers:
            print(empty_line)
            print(layer.to_string())
            print(empty_line)
            print(' ' + 80*'-')

class Autoencoder():
    def __init__(self, input_dim, num_latent_factors=20):
        self.input_dim = input_dim
        self.latent_factors =num_latent_factors
        encoder = self._build_encoder()
        decoder = self._build_decoder()
        self.model = SequentialModel(encoder + decoder)
    
    def compile(self, optimizer):
        loss = MMSE()
        self.model.compile(optimizer, loss)
    
    def fit(self, x, y, num_epochs=100, val_set=None, early_stopping=False, patience=3):
        errors = self.model.fit(x,y,num_epochs, val_set, early_stopping, patience)
        return errors

    def predict(self, x):
        self.model.predict(x)
    
    def print_summary(self):
        self.model.print_summary()
    
    def _build_encoder(self):
        encoder = [
            DenseLayer(num_units=128, input_shape=self.input_dim),
            LeakyReLU(),
            DenseLayer(num_units=self.latent_factors)
        ]
        return encoder
    
    def _build_decoder(self):
        decoder = [
            DenseLayer(num_units=128),
            LeakyReLU(),
            DenseLayer(num_units=self.input_dim)
        ]
        return decoder