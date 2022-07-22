# -*- coding: utf-8 -*-
"""
This file contains the class implementing the grid search tool.

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np
from copy import deepcopy
import time

import dictionary_unfold
from BaseFoldGenerator import StandardKFold, BaseFoldGenerator
from NeuralNetwork import NeuralNetwork
from Pipeline import Pipeline

class GridSearch():
    """ Implement the grid search tool. """
    
    def __init__(self,
                 pipeline,
                 param_dict,
                 fold_generator = StandardKFold(k = 10),
                 is_regression_task = True):
        """
        Parameters:
        pipeline (Pipeline): The pipeline to be used for the grid search
        param_dict (dict): A dictionary of pairs <hyperparameter name, hyperparameter possible value>
                            See the file dictionary_unfold.py for more details.
        fold_generator (BaseFoldGenerator) optional : The fold generator. The default one is StandardKFold(k = 10).
        is_regression_task (boolean) optional : If True the grid search will select the model
                                                with the lowest score. Otherwise, the greatest one.
                                                The default value is True.
        """
        
        assert(isinstance(pipeline, Pipeline)), "GridSearch.__init__: pipeline must be a member of the class Pipeline"
        assert(isinstance(param_dict, dict)), "GridSearch.__init__: param_dict must be a dictionary!"
        assert(isinstance(fold_generator, BaseFoldGenerator)), "GridSearch.__init__: fold_generator must inherit from class BaseFoldGenerator"
        assert(isinstance(is_regression_task, bool)), "GridSearch.__init__: is_regression_task must be a boolean"
        
        
        self.pipeline = pipeline
        self.param_dict = param_dict        
        self.is_regression_task = is_regression_task        
        self.fold_generator = fold_generator
        
        self.has_done = False
        
    def setup_model(self, pipeline, combination):
        """ Assign the hyperparameters values to the models in the pipeline.

        Parameters:
        pipeline (Pipeline) : The pipeline 
        combination (dict) : Dictionary containing the combination of hyperparameters to
                             assign to the pipeline.
        """
        
        assert(isinstance(pipeline, Pipeline)), "GridSearch.setup_model: pipeline must be a member of the class Pipeline"
        assert(isinstance(combination, (dict))), "GridSearch.setup_model: combination must be a list"
        
        # No need to check if combination's items are tuples of string-value since
        # it will cause an exception anyway in the for below             
        
        unrolled = []
        for key, value in list(combination.items()):
            splitted_key = key.split("__")
            model_name = splitted_key[0]
            unrolled.append((model_name, splitted_key[1:], value))
        
        # unrolled is now an array of triples
        # (name of the model in the pipeline, path to the hyperparam, value of the hyperparam)
        unrolled.sort(key=lambda t: len(t[1]), reverse=False)
        
        for key, path, value in unrolled:
            pipeline.pipeline_dict[key].setup_hyperparam(path, value)
    
    def grid_search(self, X, y, X_ts = None, y_ts = None):
        """ Run the grid search. Extract from param_dict the hyperparameters combinations and
            trains one model for each of them. It selects the model with the highest score and
            save it in the variale best_model. best_model is trained again with the whole input
            data X.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Training labels.
        X_ts (np.ndarray) optional: Testing data. It is used as validation data for the best model 
                                    during the final fit (in our case, used only and exclusively for plots)
        y_ts (np.ndarray) optional: Testing labels.
        """
        
        assert(isinstance(X, (np.ndarray))), "GridSearch.grid_search: X must be an ndarray!"
        assert(isinstance(y, (np.ndarray))), "GridSearch.grid_search: y must be an ndarray!"
        assert(X.shape[0] == y.shape[0]), "GridSearch.grid_search: uncoherent dataset and label matrices rows"
        
        self.params_combinations = dictionary_unfold.unfold_params(self.param_dict)
        
        n_fold = 1
        if(self.fold_generator.k > 1):
            n_fold = self.fold_generator.k
        
        self.vl_scores_matrix = np.zeros(shape = (len(self.params_combinations), n_fold))
        self.tr_scores_matrix = np.zeros(shape = (len(self.params_combinations), n_fold))
        self.tr_epochs = np.zeros(len(self.params_combinations))
        
        folds = self.fold_generator.create_fold(X, y)
        print(len(folds))
        n_iter = 0
        grid_start_time = time.time() 
        for tr_indexes, vl_indexes in folds:
            fold_start_time = time.time() 
            X_tr = X[tr_indexes].copy()
            y_tr = y[tr_indexes].copy()
            X_vl = X[vl_indexes].copy()
            y_vl = y[vl_indexes].copy()
            
            print("params_combinations ", len(self.params_combinations))
            for i, combination in enumerate(self.params_combinations):
                print(combination)
                pipeline_copy = deepcopy(self.pipeline)
                self.setup_model(pipeline_copy, combination)
                
                pipeline_copy.fit(X_tr, y_tr, X_vl, y_vl)
                
                self.vl_scores_matrix[i][n_iter] = pipeline_copy.score(X_vl, y_vl)
                self.tr_scores_matrix[i][n_iter] = pipeline_copy.score(X_tr, y_tr)
                
                if(isinstance(pipeline_copy.final_element, NeuralNetwork)):
                    self.tr_epochs[i] = self.tr_epochs[i] + 1 + pipeline_copy.final_element.tot_epochs
            fold_end_time = time.time()
            n_iter = n_iter + 1
            print("Fold time: ", (fold_end_time - fold_start_time))
        
        grid_end_time = time.time()
        print("Grid search time: ", (grid_end_time - grid_start_time))
        
        self.vl_scores = self.vl_scores_matrix.mean(1)
        self.tr_scores = self.tr_scores_matrix.mean(1)
        
        self.vl_scores_std = self.vl_scores_matrix.std(1)
        self.tr_scores_std = self.tr_scores_matrix.std(1)
        
        if(self.is_regression_task):
            self.best_combination_index = np.argmin(self.vl_scores)
        else:
            self.best_combination_index = np.argmax(self.vl_scores)
        
        self.best_combination = self.params_combinations[self.best_combination_index]
        
        self.best_model_average_vl_acc = self.vl_scores[self.best_combination_index]
        self.best_model_average_tr_acc = self.tr_scores[self.best_combination_index]
        
        self.best_model_std_vl_acc = self.vl_scores_std[self.best_combination_index]
        self.best_model_std_tr_acc = self.tr_scores_std[self.best_combination_index]
        
        self.best_model = deepcopy(self.pipeline)
        self.setup_model(self.best_model, self.best_combination)
        
        # Early stopping: calculates the optimal number of iterations as the average
        # (rounded, of course) number of iterations in the best combination's fold
        # this of course only if the best model used early stopping.
        if(isinstance(self.best_model.final_element, NeuralNetwork)):
            if(self.best_model.final_element.use_early_stopping):
                n_iter_totale = int(round(self.tr_epochs[self.best_combination_index]/n_fold))
                self.best_model.final_element.max_epochs = n_iter_totale
                
                # We don't want to train the final model using early stopping.
                self.best_model.final_element.use_early_stopping = False
        
        self.has_done = True
        self.best_model.fit(X, y, X_ts.copy(), y_ts.copy())
    
    def score(self, X, y):
        """ Call the score() method of the best model.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): Expected output.

        Returns
        float: Accuracy score between the output associated to X and the expected output.
        """
        
        assert(self.has_done == True), "GridSearch.score: call GridSearch.grid_search first!"
        assert(isinstance(X, (np.ndarray))), "GridSearch.score: X must be an ndarray!"
        assert(isinstance(y, (np.ndarray))), "GridSearch.score: y must be an ndarray!"
        assert(X.shape[0] == y.shape[0]), "GridSearch.score: uncoherent dataset and label matrices rows"
        
        return self.best_model.score(X.copy(), y.copy())