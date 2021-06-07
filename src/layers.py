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
    """ Implements a custom dense layer.

    Parameters
    ----------
    num_units : int
        Number of neurons of the layer
    input_shape : int/tuple, optional
        The shape of the layer input

    Arguments
    ---------
    num_units : int
        Number of neurons of the layer
    in_shape : tuple
        The shape of the layer input
    inputs : numpy.array
        Layer input
    W : numpy.array
        Layer Weights
    bias : numpy.array
        Layer biases
    optimizer : RMSProp/Adam
        Optimizer to use for training the layer

    """
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
        """ Initializes the layer's trainable parameters.
        
        The weights are initialized randomly in the interval [0, 1),
        while the biases are initialized to zero.
        
        """
        self.W = np.random.random((self.in_shape[0], self.num_units))
        self.bias = np.zeros((1, self.num_units))
    
    def get_output_shape(self):
        """ Returns the layer's output shape. """
        return (self.num_units, 1)
    
    def compile(self, optimizer):
        """ Initializes the optimizer to be used for training.

        Parameters
        ----------
        optimizer : RMSProp/Adam
            The optimizer that will be used to train the layer

        """
        self.optimizer = optimizer
    
    def forward_pass(self, x):
        """ Computes the output of the layer with the current beights and biases.

        Parameters
        ----------
        x : numpy.array
            Input to the forward pass

        Returns
        -------
        out : numpy.array
            Layer output computed on the given input

        """
        self.inputs = x
        if len(self.inputs.shape) == 1:
            self.inputs = self.inputs.reshape((-1,1))
        out = np.dot(x, self.W) + self.bias
        return out
    
    def backward_pass(self, accumulated_grad):
        """ Executes the backward pass for the layer training and updates its params.

        Parameters
        ----------
        accumulated_grad : numpy.array
            Gradient loss computed at the previous layers

        Returns
        -------
        accumulated_grad: numpy.array
            New gredient loss, computed as a dot product between the previous
            one and the current layer's weights.

        """
        weights_grad = np.dot(self.inputs.T, accumulated_grad)
        bias_grad = np.sum(accumulated_grad, axis=0, keepdims=True)

        accumulated_grad = np.dot(accumulated_grad, self.W.T)
        self.W = self.optimizer.update_weights(self.W, weights_grad)
        self.bias = self.optimizer.update_bias(self.bias, bias_grad)

        return accumulated_grad
    
    def to_string(self):
        """ Prints a description for the layer.

        The description includes the layer's name, its input shape and
        its output shape. It's to be used to produce a summary for the model.
        
        """
        info_line = info_line = '|{:^26s}|{:^26s}|{:^26s}|'
        input = 'Inputs: ({},)'.format(self.in_shape[0])
        output = 'Outputs: ({},)'.format(self.num_units)
        description = info_line.format(self.__class__.__name__,
                                        input,
                                        output)
        return description 

class LeakyReLU(Layer):
    """ Implements a custom Leaky ReLU activation function layer.

    Parameters
    ----------
    alpha : float, optional
        Scope of the curve for all x < 0

    Arguments
    ---------
    alpha : float
        Scope of the curve for all x < 0
    in_shape : tuple
        Layer input shape
    inputs : numpy.array
        Layer inputs

    """
    def __init__(self, alpha=0.2):

        self.alpha = alpha
        self.in_shape = None
        self.inputs = None
    
    def __call__(self, x):
        out = np.where(x < 0, self.alpha * x, x)
        return out
    
    def _gradient(self):
        """ Computes the gradient of the Leaky ReLU function."""
        out = np.where(self.inputs < 0, self.alpha, 1)
        return out
    
    def get_output_shape(self):
        """ Returns the layer's output shape. """
        return self.in_shape
    
    def forward_pass(self, x):
        """ Computes the output of the layer with the current beights and biases.

        Parameters
        ----------
        x : numpy.array
            Input to the forward pass

        Returns
        -------
        out : numpy.array
            Layer output computed on the given input

        """
        self.in_shape = x.shape
        self.inputs = x
        return self(x)
    
    def backward_pass(self, accumulated_grad):
        """ Executes the backward pass for the network training.

        Parameters
        ----------
        accumulated_grad : numpy.array
            Gradient loss computed at the previous layers

        Returns
        -------
        out: numpy.array
            New gredient loss, computed as a dot product between the previous
            one and the current layer's gradient.

        """
        out = accumulated_grad * self._gradient()
        return out
    
    def to_string(self):
        """ Prints a description for the layer.

        The description includes the layer's name, its input shape and
        its output shape. It's to be used to produce a summary for the model.
        
        """
        info_line = info_line = '|{:^26s}|{:^26s}|{:^26s}|'
        input = 'Inputs: ({},)'.format(self.in_shape[0])
        output = 'Outputs: ({},)'.format(self.in_shape[0])
        description = info_line.format(self.__class__.__name__,
                                       input,
                                       output)
        return description

