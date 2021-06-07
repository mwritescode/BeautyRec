import numpy as np

EPS = 10 ** (-8)

class RMSProp():
    """ Implements the RMSProp optimizer.

    Note that this aglorithm uses coordinate-adaptive learning rates
    and, to do so, accumulates the squares of the gradient in a 
    state vecor which is updated using learky average.

    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate to use for the parameter's updates
    gamma : float, optional
        Parameter for the leaky averaging

    Arguments
    ---------
    lr : float
        The initial learning rate to use for the parameter's updates
    gamma : float
        Parameter for the leaky averaging
    s_w : numpy.array
        State vector where the squares of the gradient for the weights
        are accumulated
    s_bias : numpy.array
        State vector where the squares of the gradient for the bias
        are accumulated
    reference_index : int
        Index of the neuron for which we want to save the learning rates
    learning_rates : dict
        Accumulator where the learning rates for the neuron specified by
        reference_index are stored at each update
    """
    def __init__(self, learning_rate=0.01, gamma=0.9):

        self.lr = learning_rate
        self.gamma = gamma
        self.s_w = None
        self.s_bias = None
        self.reference_index = 18
        self.learning_rates = {
            'weights': [],
            'bias': []
        }

    def update_weights(self, weights, grad):
        """ Updates the weights given the gradient.

        The update rule is:
        W = W - lr / (sqrt(s_w + EPS)) * gradient
        where EPS is just a small factor needed as to make sure 
        that we never divide by zero

        Parameters
        ----------
        weights : numpy.array
            Current value for the weights
        grad : numpy.array
            Current value for the greadients

        Returns
        -------
        W : numpy.array
            The updated weights

        """
        if self.s_w is None:
            self.s_w = np.zeros(weights.shape)
        self.s_w = self.gamma * self.s_w + (1 - self.gamma) * (grad**2)
        lr = self.lr / np.sqrt(self.s_w + EPS)
        self._save_one_learning_rate(lr)
        W = weights - lr*grad
        return W
    
    def update_bias(self, bias, grad):
        """ Updates the biases given the gradient.

        The update rule is:
        bias = bias - lr / (sqrt(s_bias + EPS)) * gradient
        where EPS is just a small factor needed as to make sure 
        that we never divide by zero

        Parameters
        ----------
        bias : numpy.array
            Current value for the biases
        grad : numpy.array
            Current value for the greadients

        Returns
        -------
        b : numpy.array
            The updated biases
        """
        if self.s_bias is None:
            self.s_bias = np.zeros(bias.shape)
        self.s_bias = self.gamma * self.s_bias + (1 - self.gamma) * (grad**2)
        lr = self.lr / np.sqrt(self.s_bias + EPS)
        self._save_one_learning_rate(lr, type='bias')
        b = bias - lr*grad
        return b
    
    def _save_one_learning_rate(self, lr, type='weights'):
        """ Stores one lr in the learning_rates accumulator.

        Parameters
        ----------
        lr : numpy.array
            Learning rates that have just been computed
        type : str
            Either 'weights' or 'bias', indicates whether the 
            learning rates in lr refer to the weights of to the biases

        """
        lr = lr.ravel()
        self.learning_rates[type].append(lr[self.reference_index])

class Adam():
    """ Implements the Adam optimizer.

    Note that this algorithm uses leaky averaging to obtain an estimate
    both of the momentum and of the second moment of the gradient and 
    stores them in two accumulators. The leaky averaging uses two different 
    paramteres, b1 and b2. Then this two state vectors get normalized and 
    used to compute the parameters' update.

    Parameters
    ----------
    learning_rate : float, optional
        The initial learning rate to use for the parameter's updates
    b1 : float, optional
        Parameter for the leaky averaging used to estimate the momentum
    b2 : float, optional
        Parameter for the leaky averaging used to estimate the second moment

    Arguments
    ---------
    lr : float
        The initial learning rate to use for the parameter's updates
    b1 : float
        Parameter for the leaky averaging used to estimate the momentum
    b2 : float
        Parameter for the leaky averaging used to estimate the second moment
    time_step: int
        Number of epochs run until now
    v_W : numpy.array
        Accumulator for the momentum of the weights gradient
    v_bias : numpy.array
        Accumulator for the moemntum of the weights biases
    s_W : numpy.array
        Accumulator for the second moment of the weights gradient
    s_bias : numpy.array
        Accumulator for the second moment of the bias gradient
    reference_index : int
        Index of the neuron for which we want to save the learning rates
    learning_rates : dict
        Accumulator where the learning rates for the neuron specified by
        reference_index are stored at each update
    """
    def __init__(self, learning_rate=0.01, b1=0.9, b2=0.99):

        self.lr = learning_rate
        self.b1 = b1
        self.b2 = b2
        self.time_step = 1
        self.v_W = None
        self.v_bias = None
        self.s_W = None
        self.s_bias = None
        self.reference_index = 18
        self.learning_rates = {
            'weights': [],
            'bias': []
        }

    def update_weights(self, weights, grad):
        """ Updates the weights given the gradient.

        The update rule is:
        W = W - lr * v_hat / (sqrt(s_hat + EPS))
        where EPS is just a small factor needed as to make sure 
        that we never divide by zero and v_hat and s_hat are the
        regularized first and second momentum for the weights.

        Parameters
        ----------
        weights : numpy.array
            Current value for the weights
        grad : numpy.array
            Current value for the greadients

        Returns
        -------
        W : numpy.array
            The updated weights

        """
        if self.s_W is None:
            self.v_W = np.zeros(weights.shape)
            self.s_W = np.zeros(weights.shape)
        
        self.v_W = self.b1 * self.v_W + (1- self.b1) * grad
        self.s_W = self.b2 * self.s_W + (1 - self.b2) * (grad**2)

        v_hat = self.v_W / (1 - self.b1 ** self.time_step)
        s_hat = self.s_W / (1 - self.b2 ** self.time_step)
        
        lr = self.lr / (np.sqrt(s_hat) + EPS)
        self._save_one_learning_rate(lr)
        return weights - lr * v_hat
    
    def update_bias(self, bias, grad):
        """ Updates the biases given the gradient.

        The update rule is:
        W = W - lr * v_hat / (sqrt(s_hat + EPS))
        where EPS is just a small factor needed as to make sure 
        that we never divide by zero and v_hat and s_hat are the
        regularized first and second momentum for the biases.

        Parameters
        ----------
        weights : numpy.array
            Current value for the biases
        grad : numpy.array
            Current value for the greadients

        Returns
        -------
        W : numpy.array
            The updated biases

        """
        if self.s_bias is None:
                self.v_bias = np.zeros(bias.shape)
                self.s_bias = np.zeros(bias.shape)
            
        self.v_bias = self.b1 * self.v_bias + (1- self.b1) * grad
        self.s_bias = self.b2 * self.s_bias + (1 - self.b2) * (grad**2)

        v_hat = self.v_bias / (1 - self.b1 ** self.time_step)
        s_hat = self.s_bias / (1 - self.b2 ** self.time_step)
        
        lr = self.lr / (np.sqrt(s_hat) + EPS)
        self._save_one_learning_rate(lr, type='bias')
        self.time_step +=1 
        return bias - lr * v_hat
    
    def _save_one_learning_rate(self, lr, type='weights'):
        """ Stores one lr in the learning_rates accumulator.

        Parameters
        ----------
        lr : numpy.array
            Learning rates that have just been computed
        type : str
            Either 'weights' or 'bias', indicates whether the 
            learning rates in lr refer to the weights of to the biases

        """
        lr = lr.ravel()
        self.learning_rates[type].append(lr[self.reference_index])