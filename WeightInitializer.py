# -*- coding: utf-8 -*-
"""
Classes implenting the various weights initializers.

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

from abc import ABC, abstractmethod
import numpy as np
import math

class WeightInitializer(ABC):
    """ Abstract class implementing the weights initializer. """
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def init_weights(self, n_rows, n_columns, random_state = None):
        """ Initialize the weights.

        Parameters:
        n_rows (int) : Number of rows of the weights matrix
        n_columns (int) : Number of columns of the weights matrix
        random_state (int, None) : Seed for the rng. If not None, the results are replicable.

        Returns:
        np.ndarray: initialized weights matrix.
        """
        pass
    
    @abstractmethod
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        """ Set an hyperparameter identified by hyperparam_name to
            the value specified by hyperparam_value.
        
        Parameters:
        hyperparam_name (list of string): Path to the hyperparameter to be set.
        hyperparam_value (any): Value to assign to the hyperparameter
        """
        pass

    
class UniformWeightInitializer(WeightInitializer):
    """ Class implementing the uniform weights initializers. """
    def __init__(self,
               low = -0.25,
               high = 0.25):
        """
        Parameters:
        low (float, int) optional: Lower bound of the uniform distribution.
                            Default value is -0.25.
        high (float, int) optional: Upper bound of the uniform distribution.
                            Default value is 0.25.
        """
        
        super().__init__()
        
        assert(isinstance(low, (float, int))), "UniformWeightInitializer.__init__: low must be a float or an integer"
        assert(isinstance(high, (float, int))), "UniformWeightInitializer.__init__: high must be a float or an integer"
       
        self.low = low
        self.high = high
        
        self.valid_hyperparams = ['low', 'high']
        
    def init_weights(self, n_rows, n_columns, random_state = None):
        assert(isinstance(n_rows, (int)) and n_rows >= 1), 'UniformWeightInitializer.init_weights: n_rows must be an integer >= 1'
        assert(isinstance(n_columns, (int)) and n_columns >= 1), 'UniformWeightInitializer.init_weights: n_columns must be an integer >= 1'
        assert(isinstance(random_state, (int, type(None)))), "UniformWeightInitializer.init_weights: random_state must be an integer or None"

        w = np.random.RandomState(seed = random_state).uniform(self.low, self.high, size = (n_rows, n_columns))
        b = np.random.RandomState(seed = random_state).uniform(self.low, self.high, size = (n_rows, 1))
        return (w, b)
    
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        assert(hyperparam_path[0] in self.valid_hyperparams), "UniformWeightInitializer.setup_hyperparam: hyperparam_path[0] is not a valid hyperparam"        
        setattr(self, hyperparam_path[0], hyperparam_value)

        
class GlorotUniformInitializer(WeightInitializer):
    """ Class implementing the Glorot weights initializer. """
    def __init__(self):
        super().__init__()
        
    def init_weights(self, n_rows, n_columns, random_state = None):
        assert(isinstance(n_rows, (int)) and n_rows >= 1), 'UniformWeightInitializer.init_weights: n_rows must be an integer >= 1'
        assert(isinstance(n_columns, (int)) and n_columns >= 1), 'UniformWeightInitializer.init_weights: n_columns must be an integer >= 1'
        assert(isinstance(random_state, (int, type(None)))), "UniformWeightInitializer.init_weights: random_state must be an integer or None"

        # fan-in and fan-out are effectively the number of rows and columns.
        fan_range = math.sqrt(6/(n_columns + n_rows))
        w = np.random.RandomState(seed = random_state).uniform(-fan_range, fan_range, size = (n_rows, n_columns))
        b = np.zeros(shape = (n_rows, 1))
        return (w, b)
    
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        raise Exception("Glorot has no hyperparams!")