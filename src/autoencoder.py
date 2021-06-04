import numpy as np
import pandas as pd
from data_prep import remove_nicknames
import matplotlib.pyplot as plt

import copy

from layers import DenseLayer, LeakyReLU
from optimizers import Adam, RMSProp
from utils import SquareLoss, plot_losses, plot_predictions

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
    
    def fit(self, x, y, num_epochs=100, val_set=None, early_stopping=False, patience=5):
        fails = 0
        for i in range(num_epochs):
            if fails <= patience:
                y_pred = self._forward_pass(x)
                loss = np.mean(self.loss.compute(y, y_pred))
                self.errors['training'].append(loss)
                if val_set is not None:
                    x_val, y_val = val_set
                    y_val_pred = self.predict(x_val)
                    val_loss = np.mean(self.loss.compute(y_val, y_val_pred))
                    self.errors['validation'].append(val_loss)
                    if early_stopping:
                        if val_loss > self.errors['validation'][-1]:
                            fails += 1
                        else:
                            fails = 0
                loss_grad = self.loss.grad(y, y_pred)
                print('Epoch {} \t Loss: {} '.format(i, loss))
                self._backward_pass(loss_grad)
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
        

layer1 = DenseLayer(num_units=1, input_shape=1)
#layer2 = DenseLayer(num_units=256)
#layer3 = LeakyReLU()
#layer5 = DenseLayer(num_units=1)

model = SequentialModel(layers=[layer1])
#model.add(layer5)
#model.print_summary()
optimizer = RMSProp(learning_rate=0.01)
loss = SquareLoss()
model.compile(optimizer=optimizer, loss=loss)
x1 = np.linspace(0, 100, num=650)
x_val = np.linspace(25, 250, num=200)
x_val = x_val.reshape((-1, 1))
y_val = x_val +1 
#noise = np.random.normal(3,1,500)
y = x1 + 1
#x_test = np.linspace(0, 500, num=32)
#y_test = np.sin(x_test)
#x = np.column_stack((x1, x2))
x1 = x1.reshape((-1, 1))
errors = model.fit(x1, y, num_epochs=250, val_set=(x_val, y_val), early_stopping=True)
print('The final RMS is:', errors['training'][-1])
plot_losses(errors['training'], errors['validation'])
y_pred = model.predict(x1)
plot_predictions(y, y_pred)
y_val_pred = model.predict(x_val)
plot_predictions(y_val, y_val_pred)

learning_rates = model.layers[0].optimizer.learning_rates
_, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.plot(learning_rates['weights'], label='Example of weights lr', color='b', linewidth=3)
ax2.plot(learning_rates['bias'], label='Example of bias lr', color='g', linewidth=3)
plt.legend()
plt.show()