import numpy as np
import pandas as pd
from data_prep import remove_nicknames
import matplotlib.pyplot as plt

import copy

from layers import DenseLayer, LeakyReLU
from optimizers import Adam, RMSProp
from utils import SquareLoss

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
    
    def fit(self, x, y, num_epochs=100):
        for i in range(num_epochs):
            y_pred = self._forward_pass(x)
            loss = np.mean(self.loss.compute(y, y_pred))
            self.errors['training'].append(loss)
            loss_grad = self.loss.grad(y, y_pred)
            print('Epoch {} \t Loss: {} '.format(i, loss))
            self._backward_pass(loss_grad)
        return self.errors['training']
    
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
        

layer1 = DenseLayer(num_units=60, input_shape=40)
#layer2 = DenseLayer(num_units=120)
#layer3 = LeakyReLU()
layer5 = DenseLayer(num_units=40)

model = SequentialModel(layers=[layer1])
model.add(layer5)
#model.print_summary()
optimizer = Adam(learning_rate=0.01)
loss = SquareLoss()
model.compile(optimizer=optimizer, loss=loss)
x = np.linspace(0, 20, num=40)
#noise = np.random.normal(3,1,500)
y = (x ** 2)
#x_test = np.linspace(0, 500, num=32)
#y_test = np.sin(x_test)
errors = model.fit(x, y, num_epochs=400)
print('The final RMS is:', errors[-1])
_, ax = plt.subplots()
ax.plot(errors)
plt.show()
y_pred = model.predict(x)
y_pred = y_pred.reshape(-1)
_, ax = plt.subplots()
ax.plot(x, y, label='Actual values',color='b')
ax.plot(x, y_pred, label='Predicted values', color='r')
plt.legend()
plt.show()