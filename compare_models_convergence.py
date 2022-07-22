# -*- coding: utf-8 -*-
"""
Runs the experiment in section 4.6 and plots the various graphs present in that
chapter.

WARNING: this script is insanely slow (we are talking about at least 5 hours of
runtime).

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
import numpy as np
import matplotlib.pyplot as plt

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


mode = 'loss'
delta_threshold = 10e-12
max_epochs = 37922

#m = 500
# f_star = 0.2208285018379867

#m = 20:
f_star = 0.1899490809021313
###############################################################################
#BFGS-L (m = 500)

nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), verbose = 0, random_state = 0, 
                    batch_size = 0, max_epochs = max_epochs, mode = mode, delta_threshold = delta_threshold,
                    optimizer = BFGS_L(m = 20))

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001
nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))

start_time = time.time()
nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)
end_time = time.time()
training_time_bfgs = end_time - start_time

tot_epochs_bfgs = nn.tot_epochs
bfgs_loss = nn.tr_loss
bfgs_gradients = nn.grad_norm
f_star = min(bfgs_loss) # f_star = 0.2208285018379867

diff_bfgs_fstar = np.abs(np.subtract(bfgs_loss, f_star))

q_conv_bfgs = [(diff_bfgs_fstar[i]/diff_bfgs_fstar[i-1]) for i in range(1, len(diff_bfgs_fstar) - 1)]
gap_bfgs_fstar = [(diff_bfgs_fstar[i]/f_star)    for i in range(1, len(diff_bfgs_fstar) - 1)]

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence")
plt.plot(q_conv_bfgs, 'y-', label='BGFS-L (m = 500)')
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence")
plt.plot(q_conv_bfgs, 'y-', label='BGFS-L (m = 500)')
plt.legend(fontsize  = 'large')
plt.ylim([-.1, 1.2])
plt.xlim([tot_epochs_bfgs - 100, tot_epochs_bfgs])
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap")
plt.plot(gap_bfgs_fstar, 'r-', label='BGFS-L (m = 500)')
plt.legend(fontsize  = 'large')
plt.yscale('log')
plt.show()



###############################################################################
# Adam

max_epochs = 100000

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
loss_adam = min(adam_loss) #0.5232093299015486

diff_adam_local_abs = np.abs(np.subtract(adam_loss, loss_adam))
diff_adam_fstar_abs = np.abs(np.subtract(adam_loss, f_star))

diff_adam_local = np.subtract(adam_loss, loss_adam)
diff_adam_fstar = np.subtract(adam_loss, f_star)

q_conv_adam_fstar = [(diff_adam_fstar_abs[i]/diff_adam_fstar_abs[i-1]) for i in range(1, len(adam_loss) - 1)]
q_conv_adam_local = [(diff_adam_local_abs[i]/diff_adam_local_abs[i-1]) for i in range(1, len(adam_loss) - 1)]

gap_adam_local = [(diff_adam_local[i]/loss_adam) for i in range(1, len(diff_adam_local) - 1)]
gap_adam_fstar = [(diff_adam_fstar[i]/f_star)    for i in range(1, len(diff_adam_fstar) - 1)]


plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence")
plt.plot(q_conv_adam_fstar, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence")
plt.plot(q_conv_adam_fstar, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.ylim([-.1, 1.2])
plt.xlim([tot_epochs_adam - 100, tot_epochs_adam])
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence (using Adam's last iteration as optimal x)")
plt.plot(q_conv_adam_local, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence (using Adam's last iteration as optimal x)")
plt.plot(q_conv_adam_local, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.ylim([-.1, 10])
plt.xlim([tot_epochs_adam - 100, tot_epochs_adam])
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap")
plt.plot(gap_adam_fstar, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap")
plt.plot(gap_adam_fstar, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.ylim([-.2, 3])
plt.xlim([tot_epochs_adam - 100, tot_epochs_adam])
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap (using Adam's last iteration as optimal x)")
plt.plot(gap_adam_local, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.ylim([-.2, 3])
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap (using Adam's last iteration as optimal x)")
plt.plot(gap_adam_local, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.ylim([-.2, 3])
plt.xlim([tot_epochs_adam - 100, tot_epochs_adam])
plt.show()

###############################################################################
print("f_star = ", f_star)


#plots

"""
plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence")
plt.plot(q_conv_bfgs, 'y-', label='BGFS-L (m = 500)')
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence")
plt.plot(q_conv_adam_fstar, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence (using Adam's last iteration as optimal x)")
plt.plot(q_conv_adam_local, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence")
plt.plot(q_conv_bfgs, 'y-', label='BGFS-L (m = 500)')
plt.legend(fontsize  = 'large')
plt.ylim([-.1, 1.2])
plt.xlim([tot_epochs_bfgs - 100, tot_epochs_bfgs])
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Q-convergence (using Adam's last iteration as optimal x)")
plt.plot(q_conv_adam_local, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.ylim([-.1, 1.2])
plt.xlim([tot_epochs_adam - 100, tot_epochs_adam])
plt.show()



plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap")
plt.plot(gap_bfgs_fstar, 'r-', label='BGFS-L (m = 500)')
plt.legend(fontsize  = 'large')
plt.yscale('log')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap")
plt.plot(gap_adam_fstar, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.yscale('log')
plt.show()

plt.figure(dpi=500)
plt.xlabel('Epoch')
plt.ylabel("Gap")
plt.plot(gap_adam_fstar, 'r-', label='Adam')
plt.legend(fontsize  = 'large')
plt.yscale('log')
plt.show()
"""