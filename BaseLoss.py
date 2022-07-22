# -*- coding: utf-8 -*-
"""
Classes implenting the various loss functions

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np
from numpy.linalg import matrix_power

from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """ Abstract class implementing the loss function """
    def __init__(self):
        super().__init__()
    
     
    @abstractmethod
    def calculate_de_do(self, d, o):
        """ Calculate the derivative of the error with respect to the output.
        
        Parameters:
        d (np.ndarray): Matrix of the expected outputs.
        o (np.ndarray): Matrix of the actual outputs.
        
        Returns:
        np.ndarray: The derivative of the error between d and o.
        """
        pass
    
    @abstractmethod
    def calculate_loss_value(self, d, o):
        """ Calculate the total error between the expected output
            and the actual output.
        
        Parameters:
        d (np.ndarray): Matrix of the expected outputs.
        o (np.ndarray): Matrix of the actual outputs.
        
        Returns:
        int: The total error between d and o.
        """
        pass
    
class SSE(BaseLoss):
    """ class implementing the Sum of Squared Error loss """
    def __init__(self):
        super().__init__()
        
    def calculate_de_do(self, d, o):
        return 2*(o-d)

    def calculate_loss_value(self, d, o):
        diff = (o-d)
        squared = np.square(diff)
        squared = np.asmatrix(squared)
        s = np.matrix.sum(squared)
        return s
    
class MSE(BaseLoss):
    """ class implementing the Mean Squared Error loss """
    def __init__(self):
        super().__init__()
        
    def calculate_de_do(self, d, o):
        return 2*(o-d)/(d.shape[1])
    
    def calculate_loss_value(self, d, o):
        # diff = (o-d)
        # squared = np.square(diff)
        # squared = np.asmatrix(squared)
        # s = np.matrix.sum(squared)
        # return s/(d.shape[1])
    
        return (np.square(o - d)).mean()

  
class MEE(BaseLoss):
    """ class implementing the Mean Euclidean Error loss """  
    def __init__(self):
        super().__init__()
        
    def calculate_de_do(self, d, o):
        difference = (o-d)
        
        if(not np.any(difference)):
            return np.zeros(d.shape)
        
        sqrt = np.square(difference)
        
        ones = np.ones(shape = (1, d.shape[0]))
        sqrt = np.dot(ones, sqrt)
        sqrt = np.sqrt(sqrt)
        
        result = difference / (sqrt * d.shape[1])
        return result

    def calculate_loss_value(self, d, o):
        difference = (o-d)
        sqrt = np.square(difference)
        
        ones = np.ones(shape = (1, d.shape[0]))
        sqrt = np.dot(ones, sqrt)
        sqrt = np.sqrt(sqrt)
        
        ones = np.ones(shape = (d.shape[1], 1))
        result = np.dot(sqrt, ones)
        result = float(result / d.shape[1])
        
        return result
    
class Accuracy(BaseLoss):
    """ class implementing the accuracy metric. It canno be used to
        calculate the loss, but only the total accuracy. 
    
    Raises:
    Exception: if we call calcualte_de_do() method.   
    """  
    def __init__(self):
        super().__init__()
        
    def calculate_de_do(self, d, o):
        raise Exception("Accuracy cannot be used as loss function")
    
    def calculate_loss_value(self, d, o):
        diff = np.abs(d - o)
        misclassified_count = np.sum(diff)
        size = np.size(d)
        
        return (size - misclassified_count)/size

class CrossEntropy(BaseLoss):
    """ class implementing the CrossEntropy loss.
        Altrhough both binary and categorical cross entropy have been
        implemented, only the binary one has been extensively tested."""
        
    def __init__(self):
        super().__init__()
        
    def calculate_de_do(self, d, o):
        if(d.shape[0] == 1):
            return -(d / o) + ((1 - d)/(1 - o))
        else:
            return -(d / o)

    def calculate_loss_value(self, d, o):
        if(d.shape[0] == 1):
            log_oi = np.log(o)
            di_log_oi = np.multiply(d, log_oi)
            
            log_1_oi = np.log(1-o)
            di_log_oi_b = np.multiply((1 - d), log_1_oi)
            
            return -np.sum(di_log_oi + di_log_oi_b)/d.shape[1]
        else:
            log = np.log(o)
            # print(o)
            return -np.sum(np.multiply(d, log))
    