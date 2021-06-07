import numpy as np

import copy

from layers import DenseLayer, LeakyReLU
from utils import MMSE

class SequentialModel():
    """ Bulds up a Neural Network as a linear sequence of Layers.

        Parameters
        ----------
        layers : list, optional
            Layers to be added to the model

        Arguments
        ---------
        layers : list
            Layers of the model
        optimizer : RMSProp/Adam
            Optimization algorithm to be used for training
        loss : MMSE/SquareLoss
            Loss function to be used during training
        errors: dict
            Training and validation losses per epoch

        """
    def __init__(self, layers=None):

        self.layers = self._check_input_sizes(layers)
        self.optimizer = None
        self.loss = None
        self.errors = {
            'training': [],
            'validation': []
        }
    
    def _check_input_sizes(self, layers):
        """ Checks wheter an input size has been specified for every 
        layer in the sequence. If it has not, then assigns it using 
        the output size of the previous layer.

        Also initializes weights and biases for every layer.

        Parameters
        ----------
        layers : list
            Sequence of layers
        
        Returns
        -------
        layers : list
            Sequence of layers with all their trainable parameters 
            correctly initialized

        """
        if layers is not None:
            for i, layer in enumerate(layers):
                if i > 0:
                    layer.set_input_shape(layers[i-1].get_output_shape())
                layer.initialize_params()
        return layers
    
    def add(self, layer):
        """ Adds a layer to the Model and initializes its parameters.

        Parameters
        ----------
        layer : Layer
            Layer to be added to the sequence

        """
        if self.layers:
            layer.set_input_shape(self.layers[-1].get_output_shape())
            layer.initialize_params()
        self.layers.append(layer)
    
    def compile(self, optimizer, loss):
        """ Initializes optimizer and loss function for the model.

        Parameters
        ----------
        optimizer : RMSProp/Adam
            Optimizer to be used during training
        loss : MMSE/SquareLoss
            Loss function to be used during training

        """
        self.optimizer = optimizer
        for layer in self.layers:
            layer.compile(copy.deepcopy(optimizer))
        self.loss = loss
    
    def _on_train_begins(self, val):
        """ Prints initial info when training starts.

        Parameters
        ----------
        val : tuple or None
            The validation set on which to check the 
            training progress in the format (x_val, y_val)
            where x_val and y_val are both numpy arrays

        """
        header_string = '{} \t | \t {} \t '.format('Iteration', 'Train Loss')
        num_dashes = 40
        if val is not None:
            header_string += ' \t | \t {}'.format('Validation Loss')
            num_dashes = 80
        print(num_dashes*'-')
        print(header_string)
        print(num_dashes*'-')
    
    def _predict_val_loss(self, val_set, fails, early_stopping):
        """ Computes the loss on the validation set.

        Parameters
        ----------
        val_set : tuple
            Validation set on which to check the 
            training progress in the format (x_val, y_val)
            where x_val and y_val are both numpy arrays
        fails : int
            Number of times in which the validation loss has 
            been found to be higher than the previous one
        early_stopping : bool
            Whether to implement early stopping or not

        Returns
        -------
        fails : int 
            Number of times in which the validation loss has 
            been found to be higher than the previous one
        result_msg : str
            Informative string to print for the user to keep track 
            of the validation loss

        """
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
        """ Starts the Model's training

        Parameters
        ----------
        x : numpy.array
            Features on which the network is trained
        y : numpy.array
            Actual targed values
        num_epochs : int, optional
            Number of epochs for which the training will last
        val_set : tuple, optional
            Validation set on which to check the 
            training progress in the format (x_val, y_val)
            where x_val and y_val are both numpy arrays
        early_stopping : bool, optional
            Whether to stop the training early if the validation
            loss stos descreasing
        patience : int, optional
            The number of epochs the validation loss has to stop
            decreasing for the training to be stopped when 
            early_stopping = True

        Returns
        -------
        errors : dict
            Training and validation losses per epoch

        """
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
        """ Predicts the model's output given the current parameters.

        Parameters
        ----------
        x : numpy.array
            Input features

        Returns
        -------
        x : numpy.array
            Computed output value

        """
        for layer in self.layers:
            x = layer(x)
        return x

    def _forward_pass(self, x):
        """ Computes the output of the model with the current beights and biases.

        Parameters
        ----------
        x : numpy.array
            Input to the forward pass

        Returns
        -------
        output : numpy.array
            Layer output computed on the given input

        """
        output = x
        for layer in self.layers:
            output = layer.forward_pass(output)
        return output
    
    def _backward_pass(self, accumulated_grad):
        """ Executes the backward pass for the model training.

        Parameters
        ----------
        accumulated_grad : numpy.array
            Gradient loss computed at the previous layers

        """
        for layer in reversed(self.layers):
            accumulated_grad = layer.backward_pass(accumulated_grad)
    
    def print_summary(self):
        """ Prints a summary for the model.

        For each layer in the model, the summapy includes its name,
        its input shape and its output shape.
        
        """
        print(' ' + 80*'-')
        empty_line = 3*('|' + 26*' ')+ '|'
        
        for layer in self.layers:
            print(empty_line)
            print(layer.to_string())
            print(empty_line)
            print(' ' + 80*'-')

class Autoencoder():
    """ Implements a simple Autoencoder for recommendation engines.

        Parameters
        ----------
        input_dim : int
            Number of features to take as input initially   
        num_latent_factors : int, optional
            Number features that the latent representation
            must have

        Arguments
        ---------
        input_dim : int
            Number of features to take as input initially  
        latent_factors : int
            Number features that the latent representation
            must have
        encoder : list
            List of layers that make up the encoder
        decoder : list
            List of layers that make up the decoder
        model : SequentialModel
            Neural network composed of a list of layers equal to
            the concatenation between encoder and decoder

    """
    def __init__(self, input_dim, num_latent_factors=20):

        self.input_dim = input_dim
        self.latent_factors =num_latent_factors
        encoder = self._build_encoder()
        decoder = self._build_decoder()
        self.model = SequentialModel(encoder + decoder)
    
    def compile(self, optimizer):
        """ Initializes optimizer and loss function for the model.

        Note that for this network the loss is always the MMSE.

        Parameters
        ----------
        optimizer : RMSProp/Adam
            Optimizer to be used during training
    
        """
        loss = MMSE()
        self.model.compile(optimizer, loss)
    
    def fit(self, x, y, num_epochs=100, val_set=None, early_stopping=False, patience=3):
        """ Starts the autoencoder's training

        Parameters
        ----------
        x : numpy.array
            Features on which the network is trained
        y : numpy.array
            Actual targed values
        num_epochs : int, optional
            Number of epochs for which the training will last
        val_set : tuple, optional
            Validation set on which to check the 
            training progress in the format (x_val, y_val)
            where x_val and y_val are both numpy arrays
        early_stopping : bool, optional
            Whether to stop the training early if the validation
            loss stos descreasing
        patience : int, optional
            The number of epochs the validation loss has to stop
            decreasing for the training to be stopped when 
            early_stopping = True

        Returns
        -------
        errors : dict
            Training and validation losses per epoch
        
        """
        errors = self.model.fit(x,y,num_epochs, val_set, early_stopping, patience)
        return errors

    def predict(self, x):
        """ Predicts the model's output given the current parameters.

        Parameters
        ----------
        x : numpy.array
            Input features

        Returns
        -------
        x : numpy.array
            Computed output value

        """
        self.model.predict(x)
    
    def print_summary(self):
        """ Prints a summary for the model. """
        self.model.print_summary()
    
    def _build_encoder(self):
        """ Buils the encoder's list """
        encoder = [
            DenseLayer(num_units=128, input_shape=self.input_dim),
            LeakyReLU(),
            DenseLayer(num_units=self.latent_factors)
        ]
        return encoder
    
    def _build_decoder(self):
        """ Builds the decoder's list"""
        decoder = [
            DenseLayer(num_units=128),
            LeakyReLU(),
            DenseLayer(num_units=self.input_dim)
        ]
        return decoder