# -*- coding: utf-8 -*-
"""
This file contains the class implementing the pipeline tool.

Example of pipeline:
    Suppose we want to train a neural network after standardizing the input data.
    We should create a pipeline with pipeline_dict equal to:
        {"scaler": StandardScaler(),
         "nn": NeuralNetwork()}
    and call the method fit() of the pipeline to train the pipeline.

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np

from BaseModel import FinalModel, BaseTransformer

class Pipeline(FinalModel):
    """ Implement the pipeline tool. """
    
    def __init__(self,
                 pipeline_dict):
        """
        Parameters:
        pipeline_dict (dict) : Dictionary of pairs <model name, model instance>
        """
        
        self.pipeline_dict = pipeline_dict
        
        self.check_pipeline_validity()
        self.final_element = list(self.pipeline_dict.values())[-1]
        self.has_trained = False
        
    def check_pipeline_validity(self):
        """ Check if pipeline_dict is valid, so if each elements inherits from BaseTransformer,
            except the last one which inherits from FinalModel.
        """
        
        assert(isinstance(self.pipeline_dict, (dict))), "check_pipeline_validity: pipeline_dict must be a dictionary!"
        
        for key, model in list(self.pipeline_dict.items())[0:-1]:
            assert(isinstance(model, BaseTransformer)), "check_pipeline_validity: model in pipeline is not of type BaseTransformer"
        
        key, final_element = list(self.pipeline_dict.items())[-1]
        assert(isinstance(final_element, FinalModel)), "check_pipeline_validity: final element in pipeline is not of type FinalModel"
        
    def fit(self, X_tr, y_tr, X_vl = None, y_vl = None):
        """ It takes sequentially each BaseTransformer and calls sequentially their fit() and 
            transform() methods. The output of each transform() is used as input for the following
            BaseTransformer. Finally, it calls the fit() method of the FinalModel with the 
            transformed data.

        Parameters:
        X_tr (np.ndarray): Training data.
        y_tr (np.ndarray): Labels associated to the training data X.
        X_vl (np.ndarray) optional: Validation data.
        y_vl (np.ndarray) optional: Labels associated to the validation data X_vl.
        """
        
        assert(isinstance(X_tr, (np.ndarray))), "Pipeline.fit: X_tr must be an ndarray!"
        assert(isinstance(y_tr, (np.ndarray))), "Pipeline.fit: y_tr must be an ndarray!"
        
        assert(X_tr.shape[0] == y_tr.shape[0]), "Pipeline.fit: X_tr and y_tr must have the same number of rows!"
        
        assert(isinstance(X_vl, (np.ndarray, type(None)))), "Pipeline.fit: X_vl must be an ndarray!"
        assert(isinstance(y_vl, (np.ndarray, type(None)))), "Pipeline.fit: y_vl must be an ndarray!"
        
        X_vl_b = None
        y_vl_b = None
        if(not isinstance(X_vl, type(None))):
            assert(X_vl.shape[0] == y_vl.shape[0]), "Pipeline.fit: X_vl and y_vl must have the same number of rows!"
            X_vl_b = X_vl.copy()
            y_vl_b = y_vl.copy()
        
        X_tr_b = X_tr.copy()
        y_tr_b = y_tr.copy()
        
        for key, model in list(self.pipeline_dict.items())[0:-1]:
            model.fit(X_tr_b, y_tr_b, X_vl_b, y_vl_b)
            X_tr_b, y_tr_b = model.transform(X_tr_b, y_tr_b)
            if((not X_vl_b is None) and (not y_vl_b is None)):
                X_vl_b, y_vl_b = model.transform(X_vl_b, y_vl_b)
            else:
                X_vl_b = None
                y_vl_b = None
        
        self.final_element.fit(X_tr_b, y_tr_b, X_vl_b, y_vl_b)
        self.has_trained = True
    
    def predict(self, X):
        """ It takes sequentially each BaseTransformer and calls sequentially the transform() method
            on the data. The output of each transform() is used as input for the following one.
            Finally, it calls the predict() method of the FinalModel with the transformed data.

        Parameters:
        X (np.ndarray): Input data.
        """
        assert(self.has_trained), "Pipeline.predict: call Pipeline.fit() first!"
        assert(isinstance(X, (np.ndarray))), "Pipeline.predict: X must be an ndarray"
        
        X_b = X.copy()
        
        for key, model in list(self.pipeline_dict.items())[0:-1]:
            X_b, unused = model.transform(X_b)
        
        return self.final_element.predict(X_b)
    
    def score(self, X, y):
        """ It takes sequentially each BaseTransformer and calls sequentially the transform() method
            on the data. The output of each transform() is used as input for the following one.
            Finally, it calls the predict() method of the FinalModel with the transformed data and
            calculates the accuracy score between the actual output and the expected one.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): Expected output associated to X.
        
        Returns:
        float: The accuracy score between the expected output and the actual one.
        """
        
        assert(self.has_trained), "Pipeline.predict: call Pipeline.score() first!"
        assert(isinstance(X, (np.ndarray))), "Pipeline.score: X must be an ndarray!"
        assert(isinstance(y, (np.ndarray))), "Pipeline.score: y must be an ndarray!"
        assert(X.shape[0] == y.shape[0]), "Pipeline.score: X and y should have the same number of elements"
        
        X_b = X.copy()
        y_b = y.copy()
        for key, model in list(self.pipeline_dict.items())[0:-1]:
            X_b, y_b = model.transform(X_b, y_b)
        
        return self.final_element.score(X_b, y_b)
    
    def setup_hyperparam(self, kwargs):
        """ Unimplemented """
        return 0