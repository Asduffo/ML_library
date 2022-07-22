# -*- coding: utf-8 -*-
"""
Runs the experiments in chapter 5.6.3

@author: 
    Amadei Davide (d.amadei@studenti.unipi.it)    
    Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import matplotlib.pyplot as plt
import numpy as np
import random

from QR_Regressor import QR_Regressor

###############################################################################

tr_set = np.matrix([[ 0],
                    [ 1]])
tr_set_labels = np.matrix([[ 0],
                           [ 3]])

ts_set = np.matrix([0])
ts_set_labels = np.matrix([0])

random.seed(42)

##############################################################################
rg = QR_Regressor(l2_reg = 10)
rg.fit(tr_set[0:2, :], tr_set_labels[0:2, :], ts_set[0:2, :], ts_set_labels[0:2, :])
tr_acc = rg.score(tr_set[0:1, :], tr_set_labels[0:1, :])
ts_acc = rg.score(ts_set[0:1, :], ts_set_labels[0:1, :])
print("2 training samples, L2 regularization term = 10 => w =  ", rg.w)

linsp = np.linspace(-9999999,9999999,100000)
axes = linsp*rg.w[0, 0]
plt.plot(linsp, axes, '-r', label = "w")
plt.scatter([tr_set[:,0]], [tr_set_labels[:, 0]], label = "target")
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title("2 training samples L2 regularization term = 10")
plt.show()

rg = QR_Regressor(l2_reg = 0)
rg.fit(tr_set[0:2, :], tr_set_labels[0:2, :], ts_set[0:2, :], ts_set_labels[0:2, :])
tr_acc = rg.score(tr_set[0:1, :], tr_set_labels[0:1, :])
ts_acc = rg.score(ts_set[0:1, :], ts_set_labels[0:1, :])
print("2 training samples, L2 regularization term = 0 => w =  ", rg.w)

linsp = np.linspace(-9999999,9999999,100000)
axes = linsp*rg.w[0, 0]
plt.plot(linsp, axes, '-r', label = "w")
plt.scatter([tr_set[:,0]], [tr_set_labels[:, 0]], label = "target")
plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.title("2 training samples L2 regularization term = 0")
plt.show()
