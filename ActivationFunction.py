# -*- coding: utf-8 -*-
"""
Classes implenting the various activation functions

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

from abc import ABC, abstractmethod
from scipy.special import expit, logit
import numpy as np

class ActivationFunction(ABC):
    """ Abstract class implementing the activation function """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def op(self, x):
        """ Apply the activation function element-wise.

        Parameters:
        x (np.ndarray): Matrix of the inputs.

        Returns:
        np.ndarray: Matrix of the same size of x, where we have applied the activation
                    function element-wise.   
        """
        pass
    
    @abstractmethod
    def bprop(self, x):
        """ Apply the derivative of the activation function element-wise.

        Parameters:
        x (np.ndarray): Matrix of the inputs.

        Returns:
        np.ndarray: Matrix of the same size of x, where we have applied the derivative of
                    the activation function element-wise.   
        """
        pass
    
    @abstractmethod
    def convert_output(self, x):
        """ Used only in case of classification task to convert the outputs to 0 or 1.

        Parameters:
        x (np.ndarray): Matrix of the inputs.

        Returns
        np.ndarray: Matrix of the same size of x, where each elements is rounded to 0 or 1.
        """
        pass

class Sigmoid(ActivationFunction):
    """ Class implementing the Sigmoid activation function """
    def __init(self):
        super().__init__()
        
    def op(self, x):
        return expit(x)
    
    def bprop(self, x):
        sigmoid = expit(x)
        return sigmoid*(1 - sigmoid)
        # return np.multiply(sigmoid, (1 - sigmoid))
    
    def convert_output(self, x):
        x = np.where(x > .5, 1, 0)
        return x

class ReLU(ActivationFunction):
    """ Class implementing the ReLU activation function """
    def __init(self):
        super().__init__()
        
    def op(self, x):
        return np.where(x >= 0, x, 0)
    
    def bprop(self, x):
        return np.where(x >= 0, 1, 0)
    
    def convert_output(self, x):
        return x
    
class Linear(ActivationFunction):
    """ Class implementing the Linear activation function """
    def __init(self):
        super().__init__()
        
    def op(self, x):
        return x
    
    def bprop(self, x):
        return np.ones(shape = x.shape)
    
    def convert_output(self, x):
        return x
    
class Softmax(ActivationFunction):
    """ Class implementing the Softmax activation function.
        (Not used in the ML project) """
    def __init(self):
        super().__init__()
        
    def stable_softmax(self, x):
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator/denominator
        
        return softmax
        
    def op(self, x):
        #we calculate softmax(x + z) where z = max(elements in x's column)
        return np.apply_along_axis(self.stable_softmax, 0, x)
    
    def bprop(self, x):
        s = self.op(x)
        return s*(1 - s)
    
    def convert_output(self, x):
        argmax = x.argmax(axis=0)
        x[:] = 0
        x[argmax, np.arange(len(argmax))] = 1
        return x
    
class Tanh(ActivationFunction):
    """ Class implementing the Hyperbolic Tangent activation function """
    def __init(self):
        super().__init__()
        
    def op(self, x):
        r = np.tanh(x)
        return r 
    
    def bprop(self, x):
        tanh2 = np.square(np.tanh(x))
        return 1 - tanh2
    
    def convert_output(self, x):
        x = np.where(x > 0, 1, 0)
        return x
    
class LReLU(ActivationFunction):
    """ Class implementing the Leaky ReLU activation function (alpha=0.01) """
    def __init(self):
        super().__init__()
        
    def op(self, x):
        return np.where(x > 0, x, .01*x)
    
    def bprop(self, x):
        return np.where(x > 0, 1, .01)
    
    def convert_output(self, x):
        return x