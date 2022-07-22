# -*- coding: utf-8 -*-
"""
Contain the class implenting the generator of the folds for our machine learning model.

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

from abc import ABC, abstractmethod
import numpy as np
from random import Random

class BaseFoldGenerator(ABC):
    """ Abstract class implementing the folds generator. """
    def __init__(self, k, 
                 random_state = None):
        """
        Parameters:
        k (int, float): If k>1, it represents the number of folds.
                        If 0<k<1, it represents the fraction of the training 
                        data to be used as training fold. 
                        (1-k is used as validation fold fraction)
        random_state (int, None): Seed for the rng. If not None, the results are replicable.
        """
        super().__init__()
        
        assert(k > 0 and k != 1), "BaseFoldGenerator.__init__ k must be a number higher than zero and not equal to 1"
        assert(k < 1 or (k>1 and isinstance(k, int))), "BaseFoldGenerator.__init__ if k > 1, k must be an integer"
        assert(isinstance(random_state, (int, type(None)))), "BaseFoldGenerator.__init__: random_state must be an integer or None"
        
        self.k = k
        self.random_state = random_state
        
    @abstractmethod
    def create_fold(self, X, y):
        """ 
        Parameters:
        ----------
        X (np.ndarray): Input data.
        y (np.ndarray): Input labels.
        """
        assert(isinstance(X, np.ndarray)), "BaseFoldGenerator.create_fold: X must be of type np.ndarray"
        assert(isinstance(y, np.ndarray)), "BaseFoldGenerator.create_fold: y must be of type np.ndarray"
        
        assert(X.shape[0] == y.shape[0]), "BaseFoldGenerator.create_fold: x and y must have the same number of rows"
        
        self.X = X.copy()
        self.y = y.copy()

'''
returns an array of tuples, each one of the type
(training set sample adresses, validation set sample adresses)

if k > 1, performs k fold cross validation
if 0 < k < 1, performs hold out. The training set is a fraction of the given dataset with size k,
                                 while the test set is a fraction with size (1 - k)
'''
class StandardKFold(BaseFoldGenerator):
    """ Implements the standard K-Fold procedure. """
    def __init__(self, 
                 k, 
                 random_state = None):
        super().__init__(k, random_state)
        
    def create_fold(self, X, y):
        """ 
        Parameters:
        ----------
        X (np.ndarray): Input data.
        y (np.ndarray): Input labels.

        Returns:
        list: List of tuples <training set sample indexes, validation set sample indexes>
        """
        super().create_fold(X, y)
        dataset_size = self.X.shape[0]
        indexes = np.arange(start = 0, 
                            stop = dataset_size, 
                            step = 1, dtype=int)
        
        if(self.random_state == None):
            Random().shuffle(indexes)
        else:
            Random(self.random_state).shuffle(indexes)
            self.random_state = self.random_state + 1
            
        adr = []
        
        if(self.k > 1): # K-fold
            fold_size = int(dataset_size / self.k)
            
            i = 0
            while i < (self.k - 1):
                vl_fold = indexes[i*fold_size: (i + 1)*fold_size]
                tr_fold = np.setdiff1d(indexes, vl_fold)
                adr.append((tr_fold, vl_fold))
                i = i + 1
            vl_fold = indexes[i*fold_size:]
            tr_fold = np.setdiff1d(indexes, vl_fold)
            adr.append((tr_fold, vl_fold))
        else: # Hold out
            fold_size = int(dataset_size * self.k)
            tr_fold = indexes[:fold_size]
            vl_fold = indexes[fold_size:]
            adr.append((tr_fold, vl_fold))
        
        return adr