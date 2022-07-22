# -*- coding: utf-8 -*-
"""
Runs the experiments in chapter 5.7 and plots the figures shown in that chapter

@author: 
    Amadei Davide (d.amadei@studenti.unipi.it)    
    Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

import matplotlib.pyplot as plt
import numpy as np

from QR_Regressor import QR_Regressor
from GridSearch import GridSearch
import pandas as pd
from Pipeline import Pipeline
from BaseFoldGenerator import StandardKFold

folder = "Datasets/"
train = "ML-CUP20-TR.csv"
test = "ML-CUP20-TS.csv"

training_set_initial = pd.read_csv(folder + train, sep = ',', header = None,
                                   error_bad_lines = False, comment='#', index_col = 0)

training_labels = training_set_initial[[11, 12]].to_numpy()
training_set = training_set_initial[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy()

#random_state = 0 ensures replicability 
foldgen = StandardKFold(k = .9, random_state = 0)
folds = foldgen.create_fold(X = training_set, y = training_labels)[0]
tr_set_index = folds[0]
ts_set_index = folds[1]

blind_test_set = pd.read_csv(folder + test, sep = ',', header = None,
                                   error_bad_lines = False, comment='#', index_col = 0).to_numpy()

#training set
tr_set = training_set[tr_set_index]
tr_set_labels = training_labels[tr_set_index]

#isolated test set
ts_set = training_set[ts_set_index]
ts_set_labels = training_labels[ts_set_index]

rg = QR_Regressor()

###############################################################################
# In case you want to try the grid searches performed in round 1 and 2, comment
# the "params" dictionary at the bottom and de-comment the corrispondent dictionary
# for round 1 or 2.

###############################################################################
#round 1
params = {
    'fixed': [
        {
            'rg__l2_reg' : [0, 10**8, 10**7, 10**6, 10**5, 10**4, 10**3, 10**2, 10, 1],
        }
    ],
    'variable': []
}


#round 2
params = {
    'fixed': [
        {
            'rg__l2_reg' : np.linspace(0, 2, 100),
        }
    ],
    'variable': []
}

#round 3
params = {
    'fixed': [
        {
            'rg__l2_reg' : np.linspace(0.38, 0.48, 25),
        }
    ],
    'variable': []
}

#creates the pipeline
pip_d = {"rg": rg}
pip = Pipeline(pip_d)

#starts the grid search
gs = GridSearch(pipeline = pip,
                param_dict = params,
                is_regression_task = True)
gs.fold_generator.k = 5
gs.fold_generator.random_state = 0

##############################################################################

gs.grid_search(tr_set, tr_set_labels, ts_set, ts_set_labels)
tr_acc = gs.best_model.score(tr_set, tr_set_labels)
ts_acc = gs.best_model.score(ts_set, ts_set_labels)


#prints the best combination of hyperparameters
print("best combination:", gs.best_combination)

#final training and test accuracy on the isolated test set

#output on the test set and the blind test set
out_isolated = gs.best_model.predict(ts_set)

target_norm = np.linalg.norm(ts_set_labels - out_isolated)/np.linalg.norm(ts_set_labels)

#various plots

plt.figure()
x = ts_set_labels.T[0]
y = ts_set_labels.T[1]
plt.title("Target on the isolated test set")
plt.scatter(x, y, alpha=1, s = .05)
plt.show()

plt.figure()
x = out_isolated.T[0]
y = out_isolated.T[1]
plt.title("Predicted from the isolated test set")
plt.scatter(x, y, alpha=1, s = .05)
plt.show()


best_model = gs.best_model.final_element

#saves the first 6 best models
best_combinations_sorted_indexes = gs.vl_scores.argsort()
best_model0 = gs.params_combinations[best_combinations_sorted_indexes[0]]
best_model1 = gs.params_combinations[best_combinations_sorted_indexes[1]]
best_model2 = gs.params_combinations[best_combinations_sorted_indexes[2]]
best_model3 = gs.params_combinations[best_combinations_sorted_indexes[3]]
best_model4 = gs.params_combinations[best_combinations_sorted_indexes[4]]
best_model5 = gs.params_combinations[best_combinations_sorted_indexes[5]]

#prints the training and test accuracy, plus the average validation accuracy
print("training accuracy: ", tr_acc)
print("test accuracy: ", ts_acc)
print("model's average validation accuracy", gs.vl_scores[best_combinations_sorted_indexes[0]])

print("best combinations indexes sorted: ", best_combinations_sorted_indexes)


#prints the 20 best performing models alongside their average training and
#validation accuracies during the 5 fold cv grid search.
for i in range(0, 10):
    print(gs.params_combinations[best_combinations_sorted_indexes[i]], "; vl score: ", 
          gs.vl_scores[best_combinations_sorted_indexes[i]], " +- ", 
          gs.vl_scores_std[best_combinations_sorted_indexes[i]], "; tr score: ",
          gs.tr_scores[best_combinations_sorted_indexes[i]], " +- ", 
          gs.tr_scores_std[best_combinations_sorted_indexes[i]])


tr_out_isolated = gs.best_model.predict(tr_set)

from BaseLoss import MEE
acc = MEE()

mee_tr = acc.calculate_loss_value(tr_set_labels, tr_out_isolated)
mee_ts = acc.calculate_loss_value(ts_set_labels, out_isolated)

print("MEE on the test set: ", mee_ts)
print("Relative norm of the difference between the targets and the output = ", target_norm)