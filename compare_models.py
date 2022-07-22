# -*- coding: utf-8 -*-
"""
Runs the experiment performed in chapter 4.4.1 and plots figure 4.3.

@author: Amadei Davide (d.amadei@studenti.unipi.it)    
         Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

from NeuralNetwork import NeuralNetwork
from BaseLoss import MEE, MSE
from Layer import Layer
from ActivationFunction import Linear, Sigmoid
from BaseOptimizer import BFGS_L, Adam

import time
import pandas as pd

from BaseFoldGenerator import StandardKFold

###############################################################################
#loads the dataset and the isolated test set
folder = "Datasets/"
train = "ML-CUP20-TR.csv"
test = "ML-CUP20-TS.csv"

training_set_initial = pd.read_csv(folder + train, sep = ',', header = None,
                                   error_bad_lines = False, comment='#', index_col = 0)

training_labels = training_set_initial[[11, 12]].to_numpy()
training_set = training_set_initial[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy()

foldgen = StandardKFold(k = .9, random_state = 0)
folds = foldgen.create_fold(X = training_set, y = training_labels)[0]
tr_set_index = folds[0]
ts_set_index = folds[1]

blind_test_set = pd.read_csv(folder + test, sep = ',', header = None,
                                   error_bad_lines = False, comment='#', index_col = 0).to_numpy()

#final training test and test set
tr_set = training_set[tr_set_index]
tr_set_labels = training_labels[tr_set_index]
ts_set = training_set[ts_set_index]
ts_set_labels = training_labels[ts_set_index]

mode = 'gradient'

max_epochs = 9000000
delta_threshold = 10e-5    #mode = 'gradient' threshold

###############################################################################
# Adam
nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), verbose = 0, random_state = 0, 
                   batch_size = 0, max_epochs = max_epochs, mode = mode, delta_threshold = delta_threshold,
                   optimizer = Adam(learning_rate = .075))

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001
nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))

start_time = time.time()
nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)
end_time = time.time()
training_timeAdam = end_time - start_time

adam_loss = nn.tr_loss
adam_gradients = nn.grad_norm
tot_epochs_adam = nn.tot_epochs
loss_adam = nn.tr_loss[-1]

###############################################################################
#BFGS-L (m = 20)
nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), verbose = 0, random_state = 0, 
                   batch_size = 0, max_epochs = max_epochs, mode = mode, delta_threshold = delta_threshold,
                   optimizer = BFGS_L(m = 20))

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001
nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))

start_time = time.time()
nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)
end_time = time.time()
training_time20 = end_time - start_time

bfgs20_loss = nn.tr_loss
bfgs20_gradients = nn.grad_norm
tot_epochs_bfgs20 = nn.tot_epochs
loss_20 = nn.tr_loss[-1]

###############################################################################
#BFGS-L (m = 50)
nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), verbose = 0, random_state = 0, 
                   batch_size = 0, max_epochs = max_epochs, mode = mode, delta_threshold = delta_threshold,
                   optimizer = BFGS_L(m = 50))

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001
nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))

start_time = time.time()
nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)
end_time = time.time()
training_time50 = end_time - start_time

bfgs50_loss = nn.tr_loss
bfgs50_gradients = nn.grad_norm
tot_epochs_bfgs50 = nn.tot_epochs
loss_50 = nn.tr_loss[-1]

###############################################################################
#BFGS-L (m = 100)
nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), verbose = 0, random_state = 0, 
                   batch_size = 0, max_epochs = max_epochs, mode = mode, delta_threshold = delta_threshold,
                   optimizer = BFGS_L(m = 100))

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001
nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))

start_time = time.time()
nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)
end_time = time.time()
training_time100 = end_time - start_time

bfgs100_loss = nn.tr_loss
bfgs100_gradients = nn.grad_norm
tot_epochs_bfgs100 = nn.tot_epochs
loss_100 = nn.tr_loss[-1]

###############################################################################
#BFGS-L (m = 500)


nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), verbose = 0, random_state = 0, 
                    batch_size = 0, max_epochs = max_epochs, mode = mode, delta_threshold = delta_threshold,
                    optimizer = BFGS_L(m = 500))

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001
nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))

start_time = time.time()
nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)
end_time = time.time()
training_time500 = end_time - start_time

bfgs500_loss = nn.tr_loss
bfgs500_gradients = nn.grad_norm
tot_epochs_bfgs500 = nn.tot_epochs
loss_500 = nn.tr_loss[-1]

###############################################################################

#plots
import matplotlib.pyplot as plt
plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gradient norm")
plt.plot(bfgs20_gradients, 'g-', label='BGFS-L (m = 20)')
plt.plot(bfgs50_gradients, 'k-', label='BGFS-L (m = 50)')
plt.plot(bfgs100_gradients, 'c-', label='BGFS-L (m = 100)')
plt.plot(bfgs500_gradients, 'y-', label='BGFS-L (m = 500)')
plt.plot(adam_gradients, 'r-', label='Adam')
plt.yscale("log")
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gradient norm")
plt.plot(bfgs20_gradients, 'g-', label='BGFS-L (m = 20)')
plt.plot(bfgs50_gradients, 'k-', label='BGFS-L (m = 50)')
plt.plot(bfgs100_gradients, 'c-', label='BGFS-L (m = 100)')
plt.plot(bfgs500_gradients, 'y-', label='BGFS-L (m = 500)')
plt.yscale("log")
plt.legend(fontsize  = 'large')
plt.show()



#0.23034336080122753