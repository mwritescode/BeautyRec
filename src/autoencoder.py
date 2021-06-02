import numpy as np
from numpy.core.numeric import empty_like

class DenseLayer():
    def __init__(self, num_units, input_shape=None):
        self.num_units = num_units
        self.in_shape = (input_shape,)
        self.inputs = None
        self.W = None
        self.bias = None
        self.optimizer = None
    
    def initialize_params(self):
        self.W = np.random.random((self.in_shape[0], self.num_units))
        self.bias = np.random.random((1, self.num_units))
    
    def set_input_shape(self, shape):
        self.in_shape = shape
    
    def get_output_shape(self):
        return (self.num_units,)
    
    def compile(self, optimizer):
        self.optimizer = optimizer
    
    def forward_pass(self, x, training=True):
        self.inputs = x
        out = np.dot(self.W, x) + self.bias
        return out
    
    def backward_pass(self, accumulated_grad):
        # Compute gradients
        weights_grad = np.dot(self.inputs.T, accumulated_grad)
        bias_grad = np.sum(accumulated_grad, axis=0, keepdims=True)

        accumulated_grad = np.dot(accumulated_grad, self.W.T)

        self.W = self.optimizer.update_weights(self.W, weights_grad)
        self.bias = self.optimizer.update_bias(self.bias, bias_grad)

        return accumulated_grad
    
    def to_string(self):
        info_line = info_line = '|{:^26s}|{:^26s}|{:^26s}|'
        input = 'Inputs: ({},)'.format(self.in_shape[0])
        output = 'Outputs: ({},)'.format(self.num_units)
        description = info_line.format(self.__class__.__name__,
                                        input,
                                        output)
        return description 
    

class Model():
    def __init__(self, layers=None):
        self.layers = layers
        self._check_input_sizes()
        self.optimizer = None
        self.loss = None
        self.errors = {
            'training': [],
            'validation': []
        }
    
    def _check_input_sizes(self):
        if self.layers:
            for i, layer in enumerate(self.layers[1:]):
                layer.set_input_shape(self.layers[i].get_output_shape())
    
    def add(self, layer):
        if self.layers:
            layer.set_input_shape(self.layers[-1].get_output_shape())
            layer.initialize_params()
        self.layers.append(layer)
    
    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        for layer in self.layers:
            layer.compile(optimizer)
        self.loss = loss
    
    def fit(self, x, y, num_epochs=100):
        for _ in range(num_epochs):
            y_pred = self._forward_pass(x)
            loss = self.loss.compute(y, y_pred)
            self.errors['training'].append(loss)
            loss_grad = self.loss.grad(y. y_pred)
            self._backward_pass(loss_grad)
        return self.errors['training']

    def _forward_pass(self, x, training=True):
        output = x
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def _backward_pass(self, accumulated_grad):
        for layer in reversed(self.layers):
            accumulated_grad = layer.backward_pass(accumulated_grad)
        return accumulated_grad
    
    def print_summary(self):
        print(' ' + 80*'-')
        empty_line = 3*('|' + 26*' ')+ '|'
        
        for layer in self.layers:
            print(empty_line)
            print(layer.to_string())
            print(empty_line)
            print(' ' + 80*'-')

layer1 = DenseLayer(num_units=128, input_shape=60)
layer2 = DenseLayer(num_units=64)
layer3 = DenseLayer(num_units=256)

model = Model(layers=[layer1, layer2, layer3])
model.print_summary()