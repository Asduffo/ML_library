# -*- coding: utf-8 -*-
"""
This file contains the class implementing the neural network.

@author: Lepri Elena (e.lepri@studenti.unipi.it)
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt

from BatchGenerator import BatchGenerator
from BaseModel import FinalModel
from BaseLoss import BaseLoss, MEE, Accuracy
from BaseOptimizer import BaseOptimizer, SGD
from Layer import Layer

class NeuralNetwork(FinalModel):
    """ Implement the neural network. The class itself is used only as regressor, but
        it takes into account also the classification tasks. """
    
    def __init__(self,
                 loss = MEE(),
                 accuracy = MEE(),
                 optimizer = SGD(),
                 max_epochs = 100,
                 batch_size = 0,
                 use_early_stopping = False,
                 early_stopping_maxiter = 5,
                 early_stopping_criterion = 'vl_loss',
                 input_layer_dropout_rate = 1,
                 mode = 'standard',
                 delta_threshold = None,
                 random_state = None,
                 verbose = 1):
        """
        Parameters:
        loss (BaseLoss) optional : The loss funcion. The default one is MEE().
        accuracy(BaseLoss) optional : The activation funcion. The default one is MEE().
        optimizer (BaseOptimizer) optional : The optimizer algorithm. The default one is SGD().   
        max_epochs (int) optional : The maximum number of epochs. Default value is 100.
        batch_size (int) optional : The size of each batch. With the value 0 we use the whole
                                    dataset. Default value is 0.        
        use_early_stopping (boolean) optional : If True, we use early stopping. Default value is False.
        early_stopping_maxiter (int) optional : The patience of early stopping. Default value is 5.
        early_stopping_criterion (string) optional : The metric to monitor for early stopping.
                                        Values accepted:
                                            'tr_loss' -> Training loss,
                                            'vl_loss' -> Validation loss, 
                                            'tr_acc'  -> Training accuracy, 
                                            'vl_acc'  -> Validation accuracy.         
        input_layer_dropout_rate (float) opional : Density of the dropout mask for the input layer.
        random_state (int, None) : Seed for the rng. If not None, the results are replicable.
        verbose (int) optional : If the value is 0, the evaluation metrics are printed at the end of 
                                 each epoch. It must be greater than or equal to 0.
        """        
    
        super().__init__()

        assert(isinstance(loss, BaseLoss)), "NeuralNetwork.__init__: loss should inherit from BaseLoss"
        assert(isinstance(accuracy, BaseLoss)), "NeuralNetwork.__init__: accuracy should inherit from BaseLoss"
        assert(isinstance(optimizer, BaseOptimizer)), "NeuralNetwork.__init__: optimizer should inherit from BaseOptimizer"
        
        assert(isinstance(max_epochs, (int)) and max_epochs > 0), "NeuralNetwork.__init__: max_epochs should be an integer > 0"
        assert(isinstance(batch_size, (int))), "NeuralNetwork.__init__: batch_size should be an integer"
        
        assert(isinstance(use_early_stopping, (bool))), "NeuralNetwork.__init__: use_early_stopping should be a boolean"
        assert(isinstance(early_stopping_maxiter, (int)) and early_stopping_maxiter > 0), "NeuralNetwork.__init__: early_stopping_maxiter should be an integer > 0"
        valid_early_stropping_criterions = ['vl_loss', 'tr_loss', 'vl_acc', 'tr_acc']
        assert(isinstance(early_stopping_criterion, (str)) and 
               early_stopping_criterion in valid_early_stropping_criterions), "NeuralNetwork.__init__: early_stopping_criterion should be one of vl_loss, tr_loss, vl_acc or tr_acc"

        assert(isinstance(input_layer_dropout_rate, (float, int)) 
               and input_layer_dropout_rate >= 0 
               and input_layer_dropout_rate <= 1), "NeuralNetwork.__init__: input_layer_dropout_rate must be a real number between 0 and 1" 

        assert(isinstance(random_state, (int, type(None)))), "NeuralNetwork.__init__: random_state must be an integer or None"
        assert(isinstance(verbose, (int)) and verbose >= 0), "NeuralNetwork.__init__: verbose should be an integer >= 0"
        
        valid_modes = ['standard', 'gradient', 'loss']
        assert(isinstance(mode, (str)) and mode in valid_modes), "NeuralNetwork.__init__: illegal stop mode"
        assert(isinstance(delta_threshold, (float, int, type(None)))), "NeuralNetwork.__init__: illegal delta_threshold"
        
        if(not isinstance(delta_threshold, type(None))):
            assert(delta_threshold > 0), "NeuralNetwork.__init__: delta_threshold must be > 0"
            
        
        #mode != std => delta_threshold != None <=> mode == std or delta_threshold != None
        assert(mode == 'standard' or not isinstance(delta_threshold, type(None))), "NeuralNetwork.__init__: if 'mode' is not 'standard' then delta_threshold can't be None"


        self.loss = loss
        self.accuracy = accuracy        
        self.optimizer = optimizer        
        self.max_epochs = max_epochs
        self.batch_size = batch_size       
        self.use_early_stopping = use_early_stopping
        self.early_stopping_maxiter = early_stopping_maxiter
        self.early_stopping_criterion = early_stopping_criterion
        self.input_layer_dropout_rate = input_layer_dropout_rate
        self.mode = mode
        self.delta_threshold = delta_threshold
        
        
        # Initialize neural network variables
        self.layers = []
        self.output_layer = Layer()       
        self.tr_loss = []
        self.tr_acc = []
        self.vl_loss = []
        self.vl_acc = []       
        self.grad_norm = []
        self.random_state = random_state
        self.verbose = verbose  
        self.has_initialized = False
        
        self.valid_hyperparams = ['loss', 'optimizer', 'max_epochs', 'batch_size',
                                  'use_early_stopping', 'early_stopping_maxiter',
                                  'early_stopping_criterion', 'input_layer_dropout_rate',
                                  'layers', 'output_layer', 'mode','delta_threshold']
        
    def add_layer(self, layer):
        """ Add a layer to the neural network.

        Parameters:
        layer (Layer): Layer instance to be added to the layers list.
        """
        assert(isinstance(layer, (Layer))), "NeuralNetwork.add_layer: layer should be of type Layer"
        
        layer.random_state = self.random_state
        if(self.random_state != None):
            layer.random_state = layer.random_state + len(self.layers)
        
        self.layers.append(layer)
        
    def fit(self, X, y, X_vl = None, y_vl = None):
        """ Train the neural network.

        Parameters:
        X (np.ndarray): Training data.
        y (np.ndarray): Labels associated to the training data X.
        X_vl (np.ndarray) optional: Validation data.
        y_vl (np.ndarray) optional: Labels associated to the validation data X_vl.

        Raises:
        Exception: If the early stopping criterion is not valid.
        """
        
        # print(X_vl.shape[0], ", ", y_vl.shape[0])
        
        assert(isinstance(X, (np.ndarray))), "NeuralNetwork.fit: X must be an ndarray!"
        assert(isinstance(y, (np.ndarray))), "NeuralNetwork.fit: y must be an ndarray!"
        assert(X.shape[0] == y.shape[0]), "NeuralNetwork.fit: uncoherent dataset and label matrices rows"
        assert(isinstance(X_vl, (np.ndarray, type(None)))), "NeuralNetwork.fit: X_vl must be an ndarray!"
        assert(isinstance(y_vl, (np.ndarray, type(None)))), "NeuralNetwork.fit: y_vl must be an ndarray!"
        if(not isinstance(X_vl, type(None))):
            assert(X_vl.shape[0] == y_vl.shape[0]), "NeuralNetwork.fit: X_vl and y_vl must have the same number of rows!"
        assert(self.layers != []), "NeuralNetwork.fit: add at least 1 layer!"
        
        n_samples = X.shape[0]
        n_input_features = X.shape[1]
        n_outputs = y.shape[1]
        
        # Sets the output layer
        self.output_layer.n_units = n_outputs
        
        # Adds at the end of layers[] the output_layer
        self.add_layer(self.output_layer)
        
        # Initialize the weights matrix of each layer
        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i].set_w(n_features = self.layers[i-1].n_units)
        self.layers[0].set_w(n_features = n_input_features)
        
        self.tot_epochs = 0
        
        # Generate the batches
        batch_generator = BatchGenerator(n_samples, self.batch_size,
                                         random_state = self.random_state)
        
        # Set up early stopping (if necessary)
        if(self.use_early_stopping):
            if(self.early_stopping_criterion == 'tr_loss' or
                self.early_stopping_criterion == 'vl_loss'):
                previous_value = np.Inf
            else:
                previous_value = -np.Inf
            
            self.early_stopping_iters = 0
            self.best_layer = []
            self.best_iter = 0
        
        # Set up extreme learning (if necessary)
        self.output_layer.freeze_layer = False
        output_layer_index = len(self.layers)
        self.last_layer_to_train = output_layer_index
        c = 0
        done = False
        while(c < output_layer_index and not done):
            if(not self.layers[c].freeze_layer):
                self.last_layer_to_train = c
                done = True
            else:
                c = c + 1
        
        self.has_initialized = True
        # Main loop
        while(self.check_stop()):
            adr = batch_generator.get_batches()
            
            total_tr_loss = 0
            total_tr_acc = 0
            total_vl_loss = 0
            total_vl_acc = 0
            
            for batch in adr:
                X_b = X[batch].T.copy()
                y_b = y[batch].T.copy()
                
                if(self.input_layer_dropout_rate < 1):
                    self.input_dropout_mask = random(m = X_b.shape[0],
                                                     n = X_b.shape[1], 
                                                     format = "csr", 
                                                     density = self.input_layer_dropout_rate)
                    self.input_dropout_mask.data[:] = 1
                    X_b = self.input_dropout_mask.multiply(X_b).toarray()
                
                for layer in self.layers:
                    X_b = layer.feed_forward_training(X_b).copy()
                
                curr_tr_loss = self.optimizer.optimize(d = y_b, o = X_b, layers = self.layers,
                                                       loss = self.loss, 
                                                       last_layer_to_train = self.last_layer_to_train)
                
                y_acc = self.predict(X[batch].copy(), training = False)
                curr_tr_acc = self.accuracy.calculate_loss_value(d = y_b.copy(), o = y_acc.copy().T)
                
                total_tr_loss = total_tr_loss + curr_tr_loss
                total_tr_acc = total_tr_acc + curr_tr_acc
                
                if(not X_vl is None):
                    #when we calculate the validation loss, we must have raw, non rounded
                    #values. For the accuracy, we need them rounded (but only in case of classification)
                    o_vl_loss = self.predict(X_vl, training = True)
                    o_vl_acc = self.predict(X_vl, training = False)
                    
                    total_vl_loss = total_vl_loss + self.loss.calculate_loss_value(d = y_vl.T, o = o_vl_loss.T)
                    total_vl_acc = total_vl_acc +self.accuracy.calculate_loss_value(d = y_vl.T, o = o_vl_acc.T)
                    
            total_tr_loss = total_tr_loss/len(adr)
            total_tr_acc = total_tr_acc/len(adr)
            total_vl_loss = total_vl_loss/len(adr)
            total_vl_acc = total_vl_acc/len(adr)
            
            #calculates the norm of the gradient at iteration zero, useful
            #when we want to stop when the RELATIVE norm of the gradient 
            #||df(x_i)||/||df(x_0)|| < delta_threshold
            if(self.mode == 'gradient'):
                tmp_grad = self.optimizer.flatten_layers(self.layers, gradients = True)
                
                if(self.tot_epochs == 0):
                    self.grad_norm_epoch_zero = np.linalg.norm(tmp_grad)
                    
                norm = np.linalg.norm(tmp_grad)/self.grad_norm_epoch_zero
                self.grad_norm.append(norm)
            
            # Print metrics (if necessary)
            if(self.verbose < 1):
                to_print = "Iteration {:d} loss = {:.7f}; acc: {:.7f}".format(self.tot_epochs, total_tr_loss, total_tr_acc)
                if(not X_vl is None):
                    to_append = "; vl loss = {:.7f}; vl acc: {:.7f}".format(total_vl_loss, total_vl_acc)
                    to_print = to_print + to_append
                
                if(self.mode == 'gradient'):
                    to_append = "; ||g|| = {:.7f}".format(self.grad_norm[-1])
                    to_print = to_print + to_append
                
                print(to_print)
            
            # Check early stopping (if necessary)
            if(self.use_early_stopping):
                if(self.early_stopping_criterion == 'tr_loss'):
                    to_monitor = total_tr_loss
                    has_improved = (total_tr_loss < previous_value)
                elif(self.early_stopping_criterion == 'tr_acc'):
                    to_monitor = total_tr_acc
                    has_improved = (total_tr_acc > previous_value)
                elif(self.early_stopping_criterion == 'vl_loss'):
                    to_monitor = total_vl_loss
                    has_improved = (total_vl_loss < previous_value)
                elif(self.early_stopping_criterion == 'vl_acc'):
                    to_monitor = total_vl_acc
                    has_improved = (total_vl_acc > previous_value)
                else:
                    raise Exception("unrecognized self.early_stopping_criterion")
                
                if(not has_improved):
                    self.early_stopping_iters = self.early_stopping_iters + 1
                else:
                    self.early_stopping_iters = 0
                    self.best_layer = self.layers.copy()
                    self.best_iter = self.tot_epochs
                    previous_value = to_monitor

            self.tr_loss.append(total_tr_loss)
            self.tr_acc.append(total_tr_acc)
            self.vl_loss.append(total_vl_loss)
            self.vl_acc.append(total_vl_acc)
            
            if(self.tot_epochs >= 2 and self.tr_loss[-2] - self.tr_loss[-1] < 0):
                print("Iteration ", self.tot_epochs, " WARNING (remove me in the future): ",
                      "the last loss was worse than the one obtained two epochs ago")
             
            self.tot_epochs = self.tot_epochs + 1
          
    def check_stop(self):
        """ Internal method. It checks the stopping criterion for the training phase.
            The criterions are the number of epochs and the early stopping criterion.        

        Returns:
        boolean: True if the training phase can continue; False otherwise
        
        """
        condition_1 =  self.tot_epochs < self.max_epochs
        condition_2 = True
        
        if(self.use_early_stopping and
           self.early_stopping_iters >= self.early_stopping_maxiter):
            
            self.layers = self.best_layer.copy()
            self.tot_epochs = self.best_iter
            # print(self.best_layer[0].w)
            condition_2 = False
        
        condition_3 = True
        #checks if the norm of the gradient is less or equal than the delta.
        #if that's so, it stops.
        if(self.tot_epochs > 1 and self.mode == 'gradient'):
            if(self.grad_norm[-1] <= self.delta_threshold):
                condition_3 = False
        elif(self.tot_epochs >= 2 and self.mode == 'loss'):
            diff = abs(self.tr_loss[-1] - self.tr_loss[-2])
            # print(diff)
            if(diff <= self.delta_threshold):
                condition_3 = False
        
        return condition_1 and condition_2 and condition_3
    
    def predict(self, X, training = False):
        """ Calculate the output associated to the input data X using feed_forward_test().        

        Parameters
        ----------
        X (np.ndarray): Input data.
        training (boolean) optional: Used only in NeuralNetworkClassifier.

        Returns
        np.ndarray: The output labels associated to the input data.

        """
        assert(not training or self.has_initialized), "NeuralNetwork.predict: call NeuralNetwork.fit() first!"
        assert(isinstance(X, np.ndarray)), "NeuralNetwork.predict: X should be an ndarray!"
        assert(X.shape[1] == self.layers[0].w.shape[1]), "NeuralNetwork.predict: X features does not match first layer weight matrix!"        
        assert(isinstance(training, (bool))), "NeuralNetwork.predict: training should be a boolean"

        X_b = X.T.copy()        
        for layer in self.layers:
            X_b = layer.feed_forward_test(X_b)
        
        return X_b.T
    
    def score(self, X, y):
        """ Calculate the accuracy score between the output associated to X and the expected output.

        Parameters:
        X (np.ndarray): Input data.
        y (np.ndarray): Expected output.

        Returns
        float: Accuracy score between the output associated to X and the expected output.

        """
        assert(self.has_initialized), "NeuralNetwork.score: call NeuralNetwork.fit() first!"
        assert(isinstance(X, np.ndarray)), "NeuralNetwork.score: X should be an ndarray!"
        assert(isinstance(y, np.ndarray)), "NeuralNetwork.score: y should be an ndarray!"
        assert(X.shape[0] == y.shape[0]), "NeuralNetwork.score: X and y should have the same number of samples!"
        assert(X.shape[1] == self.layers[0].w.shape[1]), "NeuralNetwork.predict: X features does not match first layer weight matrix!"
        
        o = self.predict(X)
        return self.accuracy.calculate_loss_value(y.T, o.T)
        
    def setup_hyperparam(self, hyperparam_path, hyperparam_value):
        """ Set an hyperparameter identified by hyperparam_name to
            the value specified by hyperparam_value.
        
        Parameters:
        hyperparam_name (list of string): Path to the hyperparameter to be set.
        hyperparam_value (any): Value to assign to the hyperparameter
        """
        
        assert(hyperparam_path[0] in self.valid_hyperparams), "NeuralNetwork.setup_hyperparam: hyperparam_path[0] is not a valid hyperparam"
        hyperparam_name_len = len(hyperparam_path)
        
        if(hyperparam_name_len == 1):
            setattr(self, hyperparam_path[0], hyperparam_value)
        else:
            if(hyperparam_path[0] == 'layers'):
                #TODO: controllare se esiste hyperparam_path[1]
                layer_index = int(hyperparam_path[1])
                if(len(self.layers) <= layer_index):
                    for i in range(len(self.layers), layer_index + 1):
                        self.add_layer(Layer())
                self.layers[layer_index].setup_hyperparam(hyperparam_path[2:], hyperparam_value)
            else:
                att_to_set = getattr(self, hyperparam_path[0])
                att_to_set.setup_hyperparam(hyperparam_path[1:], hyperparam_value)
    
    def plot(self, name, ylim = None):
        """ Plot the results """
        plt.figure(dpi=500)
        plt.xlabel('epoch')
        
        assert(len(name) <= 2), "NeuralNetwork.plot: max 2 losses!"
        
        
        plot_colors = ['b-', 'r--']
        c = 0
        
        for line in name:
            if(line == 'tr_loss'):
                plt.plot(self.tr_loss, plot_colors[c], label='Training loss')
                plt.ylabel("Loss")
            elif(line == 'tr_acc'):
                plt.plot(self.tr_acc, plot_colors[c], label='Training accuracy')
                plt.ylabel("Accuracy")
            elif(line == 'vl_loss'):
                plt.plot(self.vl_loss, plot_colors[c], label='Validation loss')
                plt.ylabel("Loss")
            elif(line == 'vl_acc'):
                plt.plot(self.vl_acc, plot_colors[c], label='Validation accuracy')
                plt.ylabel("Accuracy")
            elif(line == 'gradient'):
                plt.plot(self.grad_norm, plot_colors[c], label='Gradient norm')
                plt.ylabel("Gradient norm")
            else:
                raise Exception("Unrecognized loss name")
            c = c + 1
        
        if(not ylim is None):
            plt.ylim(ylim)
        plt.yscale("log")    
        
        plt.legend(fontsize  = 'large')
        plt.show()
        
class NeuralNetworkClassifier(NeuralNetwork):
    """ Extend NeuralNetwork in order to use it as classifier. """
    def __init__(self,
                 loss = MEE(),
                 accuracy = Accuracy(),                 
                 optimizer = SGD(),                 
                 max_epochs = 100,
                 batch_size = 0,                 
                 use_early_stopping = False,
                 early_stopping_maxiter = 5,
                 early_stopping_criterion = 'vl_loss',                 
                 input_layer_dropout_rate = 1,   
                 mode = 'standard',
                 delta_threshold = None,
                 random_state = None,
                 verbose = 1):
        """
        Parameters:
        loss (BaseLoss) optional : The loss funcion. The default one is MEE().
        accuracy(BaseLoss) optional : The activation funcion. The default one is Accuracy().
        optimizer (BaseOptimizer) optional : The optimizer algorithm. The default one is SGD().   
        max_epochs (int) optional : The maximum number of epochs. Default value is 100.
        batch_size (int) optional : The size of each batch. With the value 0 we use the whole
                                    dataset. Default value is 0.        
        use_early_stopping (boolean) optional : If True, we use early stopping. Default value is False.
        early_stopping_maxiter (int) optional : The patience of early stopping. Default value is 5.
        early_stopping_criterion (string) optional : The metric to monitor for early stopping.
                                        Values accepted:
                                            'tr_loss' -> Training loss,
                                            'vl_loss' -> Validation loss, 
                                            'tr_acc'  -> Training accuracy, 
                                            'vl_acc'  -> Validation accuracy.         
        input_layer_dropout_rate (float) opional : Density of the dropout mask for the input layer.
        random_state (int, None) : Seed for the rng. If not None, the results are replicable.
        verbose (int) optional : If the value is 0, the evaluation metrics are printed at the end of 
                                 each epoch. It must be greater than or equal to 0.
        """ 
        
        super().__init__(loss,
                         accuracy,
                         optimizer,
                         max_epochs,
                         batch_size,
                         use_early_stopping,
                         early_stopping_maxiter,
                         early_stopping_criterion,
                         input_layer_dropout_rate,
                         mode,
                         delta_threshold,
                         random_state,
                         verbose)
        
    def predict(self, X, training = False):
        """ Calculate the output associated to the input data X using feed_forward_test().
            If necessary, the outputs are rounded to 0 or 1.        

        Parameters
        ----------
        X (np.ndarray): Input data.
        training (boolean) optional: If false, the output associated to the input data
                                     are rounded to 0 or 1.

        Returns
        np.ndarray: The output labels associated to the input data.

        """
        
        assert(not training or self.has_initialized), "NeuralNetwork.predict: call NeuralNetwork.fit() first!"
        assert(isinstance(X, np.ndarray)), "NeuralNetwork.predict: X should be an ndarray!"
        assert(X.shape[1] == self.layers[0].w.shape[1]), "NeuralNetwork.predict: X features does not match first layer weight matrix!"
        
        assert(isinstance(training, (bool))), "NeuralNetwork.predict: training should be a boolean"

        X_b = X.T.copy()
        
        for layer in self.layers:
            X_b = layer.feed_forward_test(X_b)
        
        if not training:
            X_b = self.output_layer.activation_func.convert_output(X_b)

        return X_b.T