# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:16:12 2021

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np
from copy import deepcopy
from BaseFoldGenerator import StandardKFold, BaseFoldGenerator
from GridSearch import GridSearch

class NestedCV():
    def __init__(self,
                 grid_search,
                 fold_generator = StandardKFold(k = 5),
                 ):
        assert(isinstance(grid_search, GridSearch)), "NestedCV.__init__: grid_search must be a member of the class GridSearch"
        assert(isinstance(fold_generator, BaseFoldGenerator)), "GridSearch.__init__: fold_generator must inherit from class BaseFoldGenerator"
        
        
        self.grid_search = grid_search
        self.fold_generator = fold_generator
        
    def cross_validate(self, X, y):
        assert(isinstance(X, (np.ndarray))), "NestedCV.cross_validate: X must be an ndarray!"
        assert(isinstance(y, (np.ndarray))), "NestedCV.cross_validate: y must be an ndarray!"
        assert(X.shape[0] == y.shape[0]), "NestedCV.cross_validate: uncoherent dataset and label matrices rows"
        
        k = self.fold_generator.k
        
        #nested cross validation currently supports only cross validation and no hold out
        if(k <= 1):
            raise Exception("cross_validate: k must be an integer > 1 (only cross validation supported)")
        
        # already done by StandardKFold?
        # assert(isinstance(k, int)), "cross_validate: k must be an integer"
        
        self.best_models = []   #fold's best model obtained by training on the training folds
        self.test_scores = []   #fold's score on the validation fold
        
        i = 0
        folds = self.fold_generator.create_fold(X, y)
        for tr_indexes, vl_indexes in folds:
            print("nested cv iteration ", i)
            i = i + 1
            
            grid_search_copy = deepcopy(self.grid_search)
            grid_search_copy.grid_search(X[tr_indexes].copy(), y[tr_indexes].copy())
            
            result = grid_search_copy.score(X[vl_indexes].copy(), y[vl_indexes].copy())
            self.best_models.append(grid_search_copy.best_model)
            self.test_scores.append(result)