# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:56:41 2021

@author: 
    Ninniri Matteo (m.ninniri1@studenti.unipi.it)
    Davide Amadei (d.amadei@studenti.unipi.it)    
"""

import numpy as np
from numpy import linalg
import time

from BaseModel import FinalModel

class QR_Regressor(FinalModel):
    def __init__(self,
                 l2_reg = 0):
        super().__init__()
        
        self.l2_reg = l2_reg
        
        self.has_initialized = False
        self.valid_hyperparams = ['l2_reg']
        
    def fit(self, X, y, X_vl = None, y_vl = None):
        """ Train the regressor.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Labels associated to the training data X.
        X_vl (np.ndarray) optional: Validation data.
        y_vl (np.ndarray) optional: Labels associated to the validation data X_vl.

        Raises:
        Exception: If the early stopping criterion is not valid.
        """
        
        X_copy = X.copy()
        y_copy = y.copy()
        
        X_copy = np.hstack((X_copy, np.ones((X_copy.shape[0], 1))))
        
        n_features  = X_copy.shape[1]
        n_targets   = y_copy.shape[1]
        
        if(self.l2_reg > 0):
            identity = self.l2_reg*np.eye(n_features)
            zeros = np.zeros(shape = (n_features, y.shape[1]))
            
            X_copy = np.vstack((X_copy, identity))
            y_copy = np.vstack((y_copy, zeros))
        
        (self.Q1, self.R1) = self.thin_qr(X_copy)
        
        
        self.w = np.zeros(shape = (n_features, n_targets))
        
        for i in range(0, n_targets):
            curr_tgt = y_copy[:, i]
            
            res = np.dot(self.Q1.T, curr_tgt)
            x = self.backward_substitution(self.R1, res)
            
            self.w[:, i] = x.T
            
        self.w = self.w.T
        self.has_initialized = True
        
    def predict(self, X):
        """ Calculate the output associated to the input data X using feed_forward_test().        

        Parameters
        ----------
        X (np.ndarray): Input data.
        training (boolean) optional: Used only in QR_RegressorClassifier.

        Returns
        np.ndarray: The output labels associated to the input data.

        """
        
        assert(isinstance(X, np.ndarray)), "QR_Regressor.predict: X should be an ndarray!"
        assert(X.shape[1] + 1 == self.w.shape[1]), "QR_Regressor.predict: X features does not match first layer weight matrix!"        
        
        X_copy = X.copy()
        X_copy = np.hstack((X_copy, np.ones((X_copy.shape[0], 1))))
        
        return np.dot(self.w, X_copy.T).T
    
    def score(self, X, y):
        """ Calculate the accuracy score between the output associated to X and the expected output.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): Expected output.

        Returns
        float: Accuracy score between the output associated to X and the expected output.

        """
        
        
        assert(self.has_initialized), "QR_Regressor.score: call QR_Regressor.fit() first!"
        assert(isinstance(X, np.ndarray)), "QR_Regressor.score: X should be an ndarray!"
        assert(isinstance(y, np.ndarray)), "QR_Regressor.score: y should be an ndarray!"
        assert(X.shape[0] == y.shape[0]), "QR_Regressor.score: X and y should have the same number of samples!"
        assert(X.shape[1] + 1 == self.w.shape[1]), "QR_Regressor.predict: X features does not match first layer weight matrix!"
        
        o = self.predict(X)
        return np.linalg.norm(o - y)/o.shape[0]
    
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        assert(hyperparam_path[0] in self.valid_hyperparams), "QR_Regressor.setup_hyperparam: hyperparam_path[0] is not a valid hyperparam"
        setattr(self, hyperparam_path[0], hyperparam_value)
        
    def thin_qr(self, A):
        assert(isinstance(A, (np.ndarray, np.matrix))), "thin_qr: A must be a matrix!"
        
        
        m = A.shape[0]
        n = A.shape[1]
        assert(m >= n), "thin_qr: A should have more rows than columns! (more samples than features)"
        
        R = A.copy()
        Q = np.eye(m, n)
        
        self.h = []
        
        start_time = time.time()
        for j in range(n):
            (uj ,sj) = self.householder_vector(R[j:, j])
            self.h.append(uj)
            
            R[j, j] = sj
            R[(j+1):, j] = 0
            R[j:, (j+1):] =  R[j:, (j+1):] - 2*np.dot(uj, np.dot(uj.T, R[j:, (j+1):]))
        
        
        for j in range(n-1, -1, -1):
            uj = self.h[j]
            Q = np.block([
                [Q[:j, :j], Q[:j, j:]                                       ],
                [Q[j:, :j], Q[j:, j:] - 2*np.dot(uj, np.dot(uj.T, Q[j:,j:]))]
            ])
        
        
        end_time = time.time()
        self.factorization_time = end_time - start_time
        
        return (Q, R[:n, :])


    def backward_substitution(self, R1, y):
        n = y.shape[0]
        x = np.zeros(n)
        
        assert(R1[n-1, n-1] != 0), "backward_substitution: division by zero!"
        
        x[n-1] = y[n-1]/R1[n-1, n-1]
        for k in range (n-2, -1, -1):
            s = 0
            for j in range(k+1, n):
                s = s + (R1[k, j]*x[j])
                
            x[k] = (y[k] - s)/R1[k, k]
            
        return x
    
    def householder_vector(self, x, order = 2):
        assert(isinstance(x, np.ndarray)), "householder_vector: x must be a vector!"
        assert(np.any(x)), "householder_vector: x must be a nonzero vector!"
        
        v = x.copy()
        s = linalg.norm(v, order)
        if(v[0] >= 0):
            s = -s
        v[0] = v[0] - s
        u = v/linalg.norm(v, order)

        return (u.reshape((-1,1)), s)