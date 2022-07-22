# -*- coding: utf-8 -*-
"""
Classes implenting the optimizer of our machine learning models

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
         Amadei Davide (d.amadei@studenti.unipi.it)    
"""

from abc import ABC, abstractmethod
import numpy as np
import math as mt
import copy as cp
from os import sys
import collections

from numpy.linalg import norm
from scipy import sparse


from Layer import Layer


class BaseOptimizer(ABC):
    """ Abstract class which all optimizers inherit from """
    
    def __init__(self):
        super().__init__()
        self.n_iter = 0
    
    @abstractmethod
    def optimize(self, d, o, layers, loss, last_layer_to_train):
        """ Optimize the weights of the layers with respect to the
            expected output d and the actual output o, based on
            the input loss function.
        
        Parameters:
        d (np.ndarray): Expected output.
        o (np.ndarray): Actual output.
        layers (list): List of the layers to update
        loss (BaseLoss): loss function
        last_layer_to_train (int): Index of the bottommost layer to update,
                                    in case of extreme learning.
                                    
        Return: 
        float: Step loss.
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
    
    def backpropagate(self, d, out, layers, loss, last_layer_to_train):
        """
        Sets the layers' gradients, minimizing the loss according to d and out
        
        Parameters:
        d: targets
        o: predicted output
        layers: layers array
        loss: which loss to minimize
        last_layer_to_train: index of the last layer to train (used in extreme learningZ)
        """
        output_index = len(layers) - 1
        NET = layers[output_index].wx.copy()
		
        """ Calculate the gradient with the formula described in chapter 6.4
            of the book Deep learning by Goodfellow et al.
        """
        gradient = loss.calculate_de_do(d, out.copy())
        iteration_loss = loss.calculate_loss_value(d, out.copy())
        for i in range (output_index, last_layer_to_train - 1, -1):
            f1 = layers[i].activation_func.bprop(NET)
            gradient = np.multiply(gradient, f1)
            
            # Apply the dropout mask (if necessary)
            if(layers[i].dropout_p != 1):
                gradient = layers[i].dropout_mask.multiply(sparse.csr_matrix(gradient)).toarray()
            
            layers[i].gradient_b = np.dot(gradient, np.ones(shape = (gradient.shape[1], 1)))
            layers[i].gradient_w = np.dot(gradient, layers[i].last_input.T) + \
                 + (2*layers[i].l2_reg*layers[i].w + layers[i].l1_reg*np.sign(layers[i].w))
            
            # Gradient clipping (if necessary)
            if(self.gradient_clipping_threshold != None):
                tmp = layers[i].gradient_w.copy()
                tmp[:,:-1] = layers[i].gradient_b.copy()
                gradient_norm = norm(tmp, 'fro')
                if(gradient_norm >= self.gradient_clipping_threshold):
                    layers[i].gradient_b = (layers[i].gradient_b*self.gradient_clipping_threshold)/gradient_norm
                    layers[i].gradient_w = (layers[i].gradient_w*self.gradient_clipping_threshold)/gradient_norm
            
            
            # Calculate gradient for the next layer
            if(i > last_layer_to_train):
                gradient = np.dot(layers[i].w.T, gradient)
                NET = layers[i-1].wx.copy()
        
        return iteration_loss
    
    def flatten_layers(self, layers, gradients = False):
        flatted = []
        
        for layer in layers:
            if(gradients is False):
                flatted_w = layer.w.flatten()
                flatted_b = layer.b.flatten()
            else:
                flatted_w = layer.gradient_w.flatten()
                flatted_b = layer.gradient_b.flatten()
            
            joined  = np.concatenate((flatted_w, flatted_b))
            flatted = np.concatenate((flatted, joined))
        
        return flatted.T
        
    def unflatten_layers(self, destination_layers, flatted_layers):
        start_idx = 0
        
        for layer in destination_layers:
            tgt_w_rows = layer.w.shape[0]
            tgt_w_cols = layer.w.shape[1]
            
            tgt_b_rows = layer.b.shape[0]
            
            H_end = start_idx + tgt_w_rows*tgt_w_cols - 1
            layer.w = flatted_layers[start_idx:H_end+1].reshape(layer.w.shape)
            
            b_start = H_end + 1
            b_end = b_start + tgt_b_rows - 1
            layer.b = flatted_layers[b_start:b_end+1].reshape(layer.b.shape)
            
            start_idx = b_end + 1
        
    
class SGD(BaseOptimizer) :
    """ Implement the Stochastic Gradient Descent """
    
    def __init__(self,
                 learning_rate = .1,
                 momentum = 0,
                 nesterov = False,
                 eta_decay_fraction = 0.01,
                 eta_decay_niter = 0,
                 gradient_clipping_threshold = None
        ):
        """
        Parameters:
        learning_rate (float) optional: Learning rate. Must be greater than or equal to 0.
                                Default value is 0,1.
        momentum (float) optional: Momentum. Must be greater than or eaqual to 0.
                                Default value is 0.
        nesterov (boolean) optional: Whether to apply Yoshua Bengio simplified nesterov momentum.
                                Default value is False.
        eta_decay_fraction (float), optional : If eta_decay_niter is greater than 0, after eta_decay_niter
                                               iterations, the learning rate will amount to its initial
                                               value times this value.
                                Default value os 0,01.
        eta_decay_niter (int) optional : Number of steps where we will decay the learning rate.
                                Default value is 0 (no learning rate decay).
        gradient_clipping_threshold (float, None) optional : If not None, it represents the gradient clipping threshold.
                                Default value is None (no gradient clipping).
        """
        
        assert(isinstance(learning_rate, (float)) and learning_rate > 0), "SGD.__init__: learning_rate must be a real number > 0"
        assert(isinstance(momentum, (float, int)) and momentum >= 0), "SGD.__init__: momentum must be a real number >= 0"
        assert(isinstance(nesterov, (bool))), "SGD.__init__: nesterov must be a boolean"
        assert(isinstance(eta_decay_fraction, (float, int)) and eta_decay_fraction > 0), "SGD.__init__: eta_decay_fraction must be a real number > 0"
        assert(isinstance(eta_decay_niter, (int)) and eta_decay_niter >= 0), "SGD.__init__: eta_decay_niter must be an integer >= 0"
        assert(isinstance(gradient_clipping_threshold, (float, type(None), int))), "SGD.__init__: gradient_clipping_threshold must be a real number > 0"
        if(gradient_clipping_threshold != None):
            assert(gradient_clipping_threshold > 0), "SGD.__init__: gradient_clipping_threshold must be a real number > 0"
        
        super().__init__()
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.eta_decay_fraction = eta_decay_fraction
        self.eta_decay_niter = eta_decay_niter
        
        self.gradient_clipping_threshold = gradient_clipping_threshold
        
        self.final_learning_rate = self.eta_decay_fraction * self.learning_rate
        
        self.valid_hyperparams = ['learning_rate', 'momentum', 'nesterov',
                                  'eta_decay_fraction', 'eta_decay_niter', 'final_learning_rate',
                                  'gradient_clipping_threshold']
    
    def optimize(self, d, o, layers, loss, last_layer_to_train): 
        assert(isinstance(d, (np.ndarray))), "SGD.optimize: d must be a ndarray"
        assert(isinstance(o, (np.ndarray))), "SGD.optimize: o must be a ndarray"
        assert(d.shape == o.shape), "SGD.optimize: d and o should have the same shape"
        
        assert(isinstance(layers, list)), "SGD.optimize: layers must be a layer"
        for layer in layers:
            assert(isinstance(layer, Layer)), "SGD.optimize: one of the layer in the layers array is not a layer"
        assert(isinstance(last_layer_to_train, (int))
               and last_layer_to_train >= 0
               and last_layer_to_train < len(layers)), "SGD.optimize: last_layer_to_train must be an integer >= 0, and lower than len(layers)"
        
        #batch_size = o.shape[1]
        out = o.copy()
        
        # Calculate actual learning rate according to learning rate decay (if necessary)
        if(self.eta_decay_niter > 0):
            if(self.n_iter <= self.eta_decay_niter):
                alpha = self.n_iter / self.eta_decay_niter
                lr = (1 - alpha)*self.learning_rate + alpha*self.final_learning_rate
            else:
                lr = self.final_learning_rate
        else:
            lr = self.learning_rate
        # actual_lr = (lr/batch_size)
        actual_lr = lr
        
        iteration_loss = self.backpropagate(d, out, layers, loss, last_layer_to_train)
                
        # Update the weights
        for layer in layers:
            if(not layer.freeze_layer):
                layer.delta_b = self.momentum*layer.old_deltab - actual_lr*layer.gradient_b
                layer.delta_w = self.momentum*layer.old_deltaw - actual_lr*layer.gradient_w
                
                layer.old_deltab = layer.delta_b.copy()
                layer.old_deltaw = layer.delta_w.copy()
                
                if(self.nesterov):
                    ''' For more details about the implementation, see the report '''
                    layer.b = layer.b + self.momentum*layer.delta_b - actual_lr*layer.gradient_b
                    layer.w = layer.w + self.momentum*layer.delta_w - actual_lr*layer.gradient_w
                else:
                    layer.b = layer.b + layer.delta_b
                    layer.w = layer.w + layer.delta_w
        
        self.n_iter = self.n_iter + 1 
        return iteration_loss
    
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        assert(hyperparam_path[0] in self.valid_hyperparams), "SGD.setup_hyperparam: hyperparam_path[0] is not a valid hyperparam"
        setattr(self, hyperparam_path[0], hyperparam_value)
    

class Adam(BaseOptimizer) :
    def __init__(self,
                 learning_rate = .001,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 gradient_clipping_threshold = None
        ):
        super().__init__()
        
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        self.gradient_clipping_threshold = gradient_clipping_threshold
        
        self.s = None
        self.r = None
        
        self.delta_const = pow(10, -8)
        
        self.valid_hyperparams = ['learning_rate', 'beta_1', 'beta_2',
                                  'gradient_clipping_threshold']
    
    def optimize(self, d, o, layers, loss, last_layer_to_train): 
        out = o.copy()
        
        iteration_loss = self.backpropagate(d, out, layers, loss, last_layer_to_train)
        self.n_iter = self.n_iter + 1
        
        
        flatten_w = self.flatten_layers(layers)
        flatten_g = self.flatten_layers(layers, True)
        
        if(self.n_iter == 1):
            self.s = np.zeros(shape=flatten_w.shape)
            self.r = np.zeros(shape=flatten_w.shape)
        
        self.s = self.beta_1*self.s + (1 - self.beta_1)*flatten_g
        self.r = self.beta_2*self.r + (1 - self.beta_2)*np.multiply(flatten_g, flatten_g)
        
        at = self.learning_rate*np.sqrt(1 - pow(self.beta_2, self.n_iter))/ \
                                       (1 - pow(self.beta_1, self.n_iter))
        flatten_w = flatten_w - at*self.s/(np.sqrt(self.r) + self.delta_const)
        
        self.unflatten_layers(layers, flatten_w)
        
        return iteration_loss
    
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        assert(hyperparam_path[0] in self.valid_hyperparams), "Adam.setup_hyperparam: hyperparam_path[0] is not a valid hyperparam"
        setattr(self, hyperparam_path[0], hyperparam_value)
        

class BFGS_L(BaseOptimizer):
    def __init__(self,
                 m = 10,       #max number of s-y pairs storeable
                 a_start = 1,  #initial estimate for alpha
                 c1 = .0001,   #parameters for AWLS
                 c2 = .9,
                 MaxEval = 10,
                 
                 gradient_clipping_threshold = None
        ):
        super().__init__()
        self.m = m
        self.a_start = a_start
        self.c1 = c1
        self.c2 = c2
        self.MaxEval = MaxEval
        self.gradient_clipping_threshold = gradient_clipping_threshold
        
        self.ys_deque = collections.deque(maxlen = self.m)
        self.rho = collections.deque(maxlen = self.m)
        self.alpha = 0
        self.pk = []
        
        self.valid_hyperparams = ['m', 'a_start', 'c1', 'c2', 'delta', 'MaxEval',
                                  'gradient_clipping_threshold']
        
    def optimize(self, d, o, layers, loss, last_layer_to_train): 
        assert(isinstance(d, (np.ndarray))), "BFGS_L.optimize: d must be a ndarray"
        assert(isinstance(o, (np.ndarray))), "BFGS_L.optimize: o must be a ndarray"
        assert(d.shape == o.shape), "BFGS_L.optimize: d and o should have the same shape"
        
        assert(isinstance(layers, list)), "BFGS_L.optimize: layers must be a layer"
        for layer in layers:
            assert(isinstance(layer, Layer)), "BFGS_L.optimize: one of the layer in the layers array is not a layer"
        assert(isinstance(last_layer_to_train, (int))
               and last_layer_to_train >= 0
               and last_layer_to_train < len(layers)), "BFGS_L.optimize: last_layer_to_train must be an integer >= 0, and lower than len(layers)"
        
        out = o.copy()
        
        iteration_loss = self.backpropagate(d, out, layers, loss, last_layer_to_train)
        
        flatten_w = self.flatten_layers(layers)
        flatten_g = self.flatten_layers(layers, True)
        
        #######################################################################
        if(self.n_iter > 0):
            s_km1 = self.alpha*self.pk
            y_km1 = flatten_g - self.old_flatten_g
            
            self.ys_deque.append((s_km1, y_km1))
            self.rho.append(np.dot(y_km1.T, s_km1))
            
            dot = np.dot(s_km1.T, y_km1)
            if(dot < 0):
                print(self.n_iter, ": dot product error. ", dot)
                # sys.exit()
        
        self.pk = -self.BFGS_L_2_Loop(self.ys_deque, flatten_g)
        
        phi_0 = iteration_loss
        phip_0 = np.dot(flatten_g.T, self.pk)
        
        self.old_flatten_g = cp.deepcopy(flatten_g)
        self.alpha = self.ASWLS(layers, self.pk, phi_0, phip_0, loss, d, self.MaxEval)
        
        flatten_w = flatten_w + self.alpha*self.pk
        
        self.unflatten_layers(layers, flatten_w)
        
        #######################################################################
        
        self.n_iter = self.n_iter + 1
        return iteration_loss
    
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        assert(hyperparam_path[0] in self.valid_hyperparams), "BaseOptimizer.setup_hyperparam: hyperparam_path[0] is not a valid hyperparam"
        setattr(self, hyperparam_path[0], hyperparam_value)
        
        # TODO: validity check
        # non particolarmente elegante, ma setup_hyperparam è eseguito sempre prima
        # della fit e perciò resettare ys_deque prima di utilizzarlo non da problemi
        if(hyperparam_path[0] == 'm'):
            self.ys_deque = collections.deque(maxlen = hyperparam_value)
            self.rho = collections.deque(maxlen = hyperparam_value)
    
    def BFGS_L_2_Loop(self, ys_deque_orig, flatten_grad):
        q = cp.deepcopy(flatten_grad)
        
        ys_deque = cp.deepcopy(ys_deque_orig)
        
        alphas = np.zeros(len(ys_deque))
        
        for i in reversed(range(len(ys_deque))):
            s_i, y_i = ys_deque[i]
            
            alphas[i] = np.dot(s_i.T, q)/self.rho[i]
            q -= alphas[i]*y_i
        
        gamma_k = 1
        if(len(ys_deque) > 0):
            sk, yk = ys_deque[-1]
            gamma_k = np.dot(sk.T, yk)/np.dot(yk.T, yk)
        
        r = q*gamma_k
        
        for i in range(len(ys_deque)):
            s_i, y_i = ys_deque[i]
            beta = np.dot(y_i.T, r)/self.rho[i]
            r += s_i*(alphas[i] - beta)
        
        return r
    
    def ASWLS(self, layers, pk, phi_0, phip_0, loss, d, 
             maxiters = 10, a_start = 1, tau = 0.9, mina = 1e-16, sfgrd = 0.01):
        i = 0
        alpha = a_start
        interpolate = False
		
        while i < maxiters:
            phi_a, phip_a = self.f2phi(layers, alpha, pk, d, loss)
            
            if ((phi_a <= phi_0 + self.c1*alpha*phip_0) and (phip_a >= self.c2*phip_0)):
            # if ((phi_a <= phi_0 + self.c1*alpha*phip_0) and (abs(phip_a) <= -self.c2*phip_0)):
                return alpha
            
            if(phip_a > 0):
                interpolate = True
                break
            
            alpha = alpha / tau
            i = i + 1
            
        if(not interpolate):
            print(self.n_iter, "-", i,": failed ASWLS Step 1.")
            return 1
            
        sx = 0
        dx = alpha

        
        phip_sx = cp.deepcopy(phip_0)
        phip_dx = cp.deepcopy(phip_a)
        
        i = 0
        while( i < maxiters and (dx - sx) > mina and phip_dx > 1e-12):
        # while( i < maxiters ):
            alpha = (sx*phip_dx - dx*phip_sx)/(phip_dx - phip_sx)
            
            alpha = max([sx+(dx-sx)*sfgrd, 
                         min([dx-(dx-sx)*sfgrd, alpha])]);
            
            phi_a, phip_a = self.f2phi(layers, alpha, pk, d, loss)
            
            if ((phi_a <= phi_0 + self.c1*alpha*phip_0) and (phip_a >= self.c2*phip_0)):
            # if ((phi_a <= phi_0 + self.c1*alpha*phip_0) and (abs(phip_a) <= -self.c2*phip_0)):
                return alpha
            
            if(phip_a < 0):
                sx = alpha
                phip_sx = cp.deepcopy(phip_a)
            else:
                dx = alpha
                if(dx <= mina): #too close to zero > it's basically the original point x
                    break
                phip_dx = cp.deepcopy(phip_a)
			
            i = i + 1
        
        print(self.n_iter, "-", i,": failed ASWLS interpolation. Returning ", alpha)
        return alpha
    
    def f2phi(self, layers, alpha, pk, d, loss):
        flatten_wb_alpha = self.flatten_layers(layers) + alpha*pk
        
        layers_alpha = cp.deepcopy(layers)
        self.unflatten_layers(layers_alpha, flatten_wb_alpha)
        
        out = layers_alpha[0].last_input.copy()
        for layer in layers_alpha:
            out = layer.feed_forward(layer.w, layer.b, out).copy()
        
        phi_a = self.backpropagate(d, out, layers_alpha, loss, 0)
        
        flatten_grads_wa = self.flatten_layers(layers_alpha, True)
        phip_a = np.dot(flatten_grads_wa.T, pk)
        
        return (phi_a, phip_a)