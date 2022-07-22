# -*- coding: utf-8 -*-
"""
Contain the class implenting the layer of the neural network.

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np
from scipy.sparse import random

import WeightInitializer
from WeightInitializer import WeightInitializer as weight_initializer
import ActivationFunction
from ActivationFunction import ActivationFunction as activation_function


class Layer:
    """ Represent a standard dense layer of a neural network. """
    def __init__(
            self,
            n_units = 10,
            activation_func = ActivationFunction.Sigmoid(),
            l2_reg = 0,
            l1_reg = 0,
            w_init = WeightInitializer.GlorotUniformInitializer(),
            dropout_p = 1,
            freeze_layer = False,
            random_state = None
            ):
        """
        Parameters:
        n_units (int) optional: Number of units. Default value is 10.
        activation_func (ActivationFunction) optional: Activation function of the layer.
                                            Default value is Sigmoid().
        l2_reg (float) optional: Coefficient for L2 regularization. Default value is 0.
        l1_reg (float) optional: Coefficient for L1 regularization. Default value is 0.
        w_init (WeightInitializer) optional: The weights initializer. 
                                            Default value is GlorotUniformInitializer().
        dropout_p (float) optional : Density of the dropout mask. The default is 1.
        freeze_layer (boolean) optional : If True, the weights of the layer will not be updated
                                            (used for extreme learnig). The default is False.
        random_state (int, None) : Seed for the rng. If not None, the results are replicable.
        """
        
        assert(isinstance(n_units, (int)) and n_units > 0), "Layer.__init__: n_units must be an integer > 0" 
        assert(isinstance(activation_func, (activation_function))), "Layer.__init__: activation_func must inherit from class ActivationFunction" 
        assert(isinstance(l2_reg, (float, int)) and l2_reg >= 0), "Layer.__init__: l2_reg must be a real number >= 0" 
        assert(isinstance(l1_reg, (float, int)) and l2_reg >= 0), "Layer.__init__: l1_reg must be a real number >= 0" 
        assert(isinstance(w_init, (weight_initializer))), "Layer.__init__: weight_initializer must inherit from class WeightInitializer" 
        assert(isinstance(dropout_p, (float, int)) and dropout_p >= 0 and dropout_p <= 1), "Layer.__init__: dropout_p must be a real number between 0 and 1" 
        assert(isinstance(freeze_layer, (bool))), "Layer.__init__: freeze_layer must be a boolean"
        assert(isinstance(random_state, (int, type(None)))), "Layer.__init__: random_state must be an integer or None"
        
        self.n_units = n_units
        self.activation_func = activation_func
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.w_init = w_init
        self.dropout_p = dropout_p
        self.freeze_layer = freeze_layer
        self.random_state = random_state
        
        self.valid_hyperparams = ['n_units', 'activation_func', 'l2_reg',
                                  'l1_reg', 'w_init', 'dropout_p', 'freeze_layer']
    
    def set_w(self, n_features):
        """ Initialize the weights matrix and bias vector using the w_init class.
            Initialize to zero the velocity matrices.

        Parameters:
        ----------
        n_features (int): Number of columns of the weights matrix and number
                          of elements of the bias vector.
        
        Note: The number of rows of the weight matrix is the number of units in the layer.
        """
        assert(isinstance(n_features, (int)) and n_features > 0), "Layer.set_w: n_features must be an integer > 0"
        
        (self.w, self.b) = self.w_init.init_weights(n_rows = self.n_units, 
                                                    n_columns = n_features,
                                                    random_state = self.random_state)
        
        self.deltaw = np.zeros(shape = self.w.shape)
        self.deltab = np.zeros(shape = self.b.shape)
        
        self.old_deltaw = np.zeros(shape = self.w.shape)
        self.old_deltab = np.zeros(shape = self.b.shape)
    
    def feed_forward(self, W, b, X):
        """ Internal method. It calculates the NET matrix, so the dot product between the 
            weights matrix and the input matrix (including the bias) and returns the output
            of the activation function applied to it.

        Parameters:
        W (np.ndarray): The weights matrix for the dot product.
        b (np.ndarray): The bias vector for the dot product.
        X (np.ndarray): The input matrix for the dot product.

        Returns:
        np.ndarray: The output of the activation function applied to the NET matrix.
        """
        assert(isinstance(X,(np.ndarray))), 'Layer.feed_forward X is not np.ndarray'
        assert(isinstance(W,(np.ndarray))), 'Layer.feed_forward W is not np.ndarray'
        assert(isinstance(b,(np.ndarray))), 'Layer.feed_forward b is not np.ndarray'
        
        # Check if w has been initialized
        assert(X.size != 0), 'Layer.feed_forward: please initialize X first'
        assert(W.size != 0), 'Layer.feed_forward: please initialize w first'
        assert(b.size != 0), 'Layer.feed_forward: please initialize b first'
        
        assert(W.shape[0] == b.shape[0]), "Layer.feed_forward: W and b must have the same number of rows"
        assert (W.shape[1] == X.shape[0]), 'Layer.feed_forward input is not of the same shape of the weight matrix'
        
        # 'wx' is a matrix where the number of rows is equal to the number of units
        # in the layer and the number of columns is the number of elements in the batch.
        # So, for each column of 'wx', we have to sum the bias vector; to do this in one step,
        # we create the matrix 'repeat', where each column is the vector b.
        # We sum 'repeat' and 'wx' to obtain the NET matrix.
        repeat = np.repeat(b, repeats = X.shape[1], axis = 1)
        self.wx = np.dot(W, X) + repeat
        output = self.activation_func.op(self.wx)
        
        return output
    
    def feed_forward_training(self, X) :
        """ Apply feed_forward() to the input data and, if necessary, apply dropout
            to the output. It should be called only during the training phase, by the 
            method fit() of the neural network.     

        Parameters:
        X (np.ndarray): The input matrix for the dot product.

        Returns:
        np.ndarray: The output of this layer for the input X.
        """
        assert(isinstance(X,(np.ndarray))), 'Layer.feed_forward_training X is not np.ndarray'
        assert(X.size != 0), 'Layer.feed_forward_training: please initialize X first'
        assert(self.w.shape[1] == X.shape[0]), 'Layer.feed_forward_training input is not of the same shape of the weight matrix'
        
        # Apply feed_forward()
        self.last_output = self.feed_forward(self.w, self.b, X)
        self.last_input = X
        
        # Apply dropout (if necessary)
        if(self.dropout_p < 1):
            self.dropout_mask = random(m = self.last_output.shape[0],
                                       n = self.last_output.shape[1], 
                                       format = "csr", 
                                       density = self.dropout_p,
                                       random_state = self.random_state)
            
            self.dropout_mask.data[:] = 1
            self.last_output = self.dropout_mask.multiply(self.last_output).toarray()
        
        return self.last_output
    

    def feed_forward_test(self, X):
        """ Apply feed_forward() to the input data and, if necessary, multiply the weight
            matrix and the dropout density. It should be called only during the generalization
            phase, by the method predict() and score() of the neural network.  

        Parameters:
        X (np.ndarray): The input matrix for the dot product.

        Returns:
        np.ndarray: The output of this layer for the input X.
        """
        assert(isinstance(X,(np.ndarray))), 'Layer.feed_forward_test X is not np.ndarray'
        assert(X.size != 0), 'Layer.feed_forward_test: please initialize X first'
        assert(self.w.shape[1] == X.shape[0]), 'Layer.feed_forward_test input is not of the same shape of the weight matrix'
        
        return self.feed_forward(self.w*self.dropout_p, self.b, X)
    
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        """ Set an hyperparameter identified by hyperparam_name to
            the value specified by hyperparam_value.
        
        Parameters:
        hyperparam_name (list of string): Path to the hyperparameter to be set.
        hyperparam_value (any): Value to assign to the hyperparameter
        """
        assert(len(hyperparam_path) > 0), "Layer.setup_hyperparam: must contain at least 1 element!"
        assert(hyperparam_path[0] in self.valid_hyperparams), "Layer.setup_hyperparam: hyperparam_path[0] is not a valid hyperparam"
        
        path_len = len(hyperparam_path)
        if(path_len == 1):
            setattr(self, hyperparam_path[0], hyperparam_value)
        else:
            att_to_set = getattr(self, hyperparam_path[0])
            att_to_set.setup_hyperparam(hyperparam_path[1:], hyperparam_value)