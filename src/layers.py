import numpy as np

from abc import ABC, abstractmethod

class Layer(ABC):
    
    @abstractmethod
    def forward_pass(self, x):
        pass
    
    @abstractmethod
    def backward_pass(self, accumulated_grad):
        pass

    @abstractmethod
    def get_output_shape(self):
        pass

    def compile(self, optimizer):
        pass
    
    def initialize_params(self):
        pass
    
    def set_input_shape(self, shape):
        self.in_shape = shape

    def to_string(self):
        info_line = info_line = '|{:^80s}|'
        description = info_line.format(self.__class__.__name__)
        return description
    

class DenseLayer(Layer):
    def __init__(self, num_units, input_shape=None):
        self.num_units = num_units
        self.in_shape = (input_shape, 1)
        self.inputs = None
        self.W = None
        self.bias = None
        self.optimizer = None
    
    def __call__(self, x):
        return np.dot(x, self.W) + self.bias
    
    def initialize_params(self):
        self.W = np.random.random((self.in_shape[0], self.num_units))
        self.bias = np.zeros((1, self.num_units))
    
    def get_output_shape(self):
        return (self.num_units, 1)
    
    def compile(self, optimizer):
        self.optimizer = (optimizer)
    
    def forward_pass(self, x):
        self.inputs = x.reshape(self.in_shape)
        out = np.dot(x, self.W) + self.bias
        return out
    
    def backward_pass(self, accumulated_grad):
        # Compute gradients
        weights_grad = np.dot(self.inputs, accumulated_grad)
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

class LeakyReLU(Layer):
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.in_shape = None
        self.inputs = None
    
    def __call__(self, x):
        self.in_shape = x.shape
        self.inputs = x
        out = np.where(x < 0, self.alpha * x, x)
        return out
    
    def _gradient(self):
        out = np.where(self.inputs < 0, self.alpha, 1)
        return out
    
    def get_output_shape(self):
        return self.in_shape
    
    def forward_pass(self, x):
        return self(x)
    
    def backward_pass(self, accumulated_grad):
        out = accumulated_grad * self._gradient()
        return out
    
    def to_string(self):
        info_line = info_line = '|{:^26s}|{:^26s}|{:^26s}|'
        input = 'Inputs: ({},)'.format(self.in_shape[0])
        output = 'Outputs: ({},)'.format(self.in_shape[0])
        description = info_line.format(self.__class__.__name__,
                                       input,
                                       output)
        return description

