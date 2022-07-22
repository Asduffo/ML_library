# -*- coding: utf-8 -*-
"""
Runs the test described in section 4.3 and plots figure 4.2

@author: 
    Amadei Davide (d.amadei@studenti.unipi.it)    
    Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

from NeuralNetwork import NeuralNetwork
from BaseLoss import MEE, MSE
from Layer import Layer
from ActivationFunction import Linear, Tanh, Sigmoid, ReLU
from BaseOptimizer import Adam, SGD, BFGS_L
from GridSearchExperimental import GridSearch
import pandas as pd
from Pipeline import Pipeline
from BaseFoldGenerator import StandardKFold

folder = "Datasets/"
train = "ML-CUP20-TR.csv"
test = "ML-CUP20-TS.csv"

training_set_initial = pd.read_csv(folder + train, sep = ',', header = None,
                                   on_bad_lines = 'skip', comment='#', index_col = 0)

training_labels = training_set_initial[[11, 12]].to_numpy()
training_set = training_set_initial[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy()

#random_state = 0 ensures replicability 
foldgen = StandardKFold(k = .9, random_state = 0)
folds = foldgen.create_fold(X = training_set, y = training_labels)[0]
tr_set_index = folds[0]
ts_set_index = folds[1]

blind_test_set = pd.read_csv(folder + test, sep = ',', header = None,
                             on_bad_lines = 'skip', comment='#', index_col = 0).to_numpy()

#training set
tr_set = training_set[tr_set_index]
tr_set_labels = training_labels[tr_set_index]

#isolated test set
ts_set = training_set[ts_set_index]
ts_set_labels = training_labels[ts_set_index]

###############################################################################
nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), 
                   verbose = 1, random_state = 0, use_early_stopping = True,
                   early_stopping_maxiter = 100, early_stopping_criterion = 'vl_loss',
                   max_epochs = 400, batch_size = 0,
                   optimizer = BFGS_L())

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001

nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))
# One shot with our current best hyperparameters

nn.use_early_stopping = False
nn.verbose = 0
nn.max_epochs = 83
nn.optimizer.setup_hyperparam(['m'], 20)
nn.optimizer.MaxEval = 10

nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)

tr_acc = nn.score(tr_set, tr_set_labels)
ts_acc = nn.score(ts_set, ts_set_labels)
print("tr_acc = ", tr_acc)
print("ts_acc = ", ts_acc)

nn.plot(['tr_loss', 'vl_loss'])
nn.plot(['tr_acc', 'vl_acc'])
###############################################################################
nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), 
                   verbose = 1, random_state = 0, use_early_stopping = True,
                   early_stopping_maxiter = 100, early_stopping_criterion = 'vl_loss',
                   max_epochs = 400, batch_size = 0,
                   optimizer = BFGS_L())

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001

nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))
# One shot with our current best hyperparameters

nn.use_early_stopping = False
nn.verbose = 0
nn.optimizer.MaxEval = 10

nn.optimizer.setup_hyperparam(['m'], 20)
nn.mode = 'gradient'
nn.delta_threshold = 10e-5
nn.max_epochs = 100000

nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)

tr_acc1 = nn.score(tr_set, tr_set_labels)
ts_acc1 = nn.score(ts_set, ts_set_labels)
print("tr_acc = ", tr_acc1)
print("ts_acc = ", ts_acc1)

nn.plot(['gradient'])
###############################################################################
nn = NeuralNetwork(loss = MSE(), accuracy = MEE(), 
                   verbose = 1, random_state = 0, use_early_stopping = True,
                   early_stopping_maxiter = 100, early_stopping_criterion = 'vl_loss',
                   max_epochs = 400, batch_size = 0,
                   optimizer = BFGS_L())

nn.output_layer.activation_func = Linear()
nn.output_layer.l2_reg = .00001

nn.add_layer(Layer(activation_func = Sigmoid(), n_units = 120, l2_reg = .00001))
# One shot with our current best hyperparameters

nn.use_early_stopping = False
nn.verbose = 0
nn.optimizer.MaxEval = 10

nn.optimizer.setup_hyperparam(['m'], 500)
nn.mode = 'gradient'
nn.delta_threshold = 10e-5
nn.max_epochs = 100000

nn.fit(tr_set, tr_set_labels, ts_set, ts_set_labels)

tr_acc2 = nn.score(tr_set, tr_set_labels)
ts_acc2 = nn.score(ts_set, ts_set_labels)

nn.plot(['gradient'])

print("tr_acc = ", tr_acc)
print("ts_acc = ", ts_acc)

###############################################################################
#full grid search (decomment to run)
"""

params = {
    'fixed': [
        {
            'nn__optimizer__m' : [30, 20, 10, 3],
        }
    ],
    'variable': []
}

###############################################################################


#creates the pipeline
pip_d = {"nn": nn}
pip = Pipeline(pip_d)

#starts the grid search
gs = GridSearch(pipeline = pip,
                param_dict = params,
                is_regression_task = True)
gs.fold_generator.k = .9
gs.fold_generator.random_state = 0
gs.grid_search(tr_set, tr_set_labels, ts_set, ts_set_labels)

#prints the best combination of hyperparameters
print("best combination:", gs.best_combination)
print("best model epochs: ", gs.best_model.final_element.max_epochs)

#final learning curves (zoomed and unzoomed)
gs.best_model.pipeline_dict["nn"].plot(['tr_loss', 'vl_loss'])
gs.best_model.pipeline_dict["nn"].plot(['tr_acc', 'vl_acc'])
gs.best_model.pipeline_dict["nn"].plot(['tr_loss', 'vl_loss'], ylim = [0, 20])
gs.best_model.pipeline_dict["nn"].plot(['tr_acc', 'vl_acc'], ylim = [0, 4])

#final training and test accuracy on the isolated test set
tr_acc = gs.best_model.score(tr_set, tr_set_labels)
ts_acc = gs.best_model.score(ts_set, ts_set_labels)

#output on the test set and the blind test set
out_isolated = gs.best_model.predict(ts_set)
out_blind = gs.best_model.predict(blind_test_set)

#various plots
import matplotlib.pyplot as plt
plt.figure()
x = ts_set_labels.T[0]
y = ts_set_labels.T[1]
plt.title("Target on the isolated test set")
plt.scatter(x, y, alpha=1, s = .05)
plt.xlim([25, 80])
plt.ylim([-45, 0])
plt.show()

plt.figure()
x = out_isolated.T[0]
y = out_isolated.T[1]
plt.title("Predicted from the isolated test set")
plt.scatter(x, y, alpha=1, s = .05)
plt.xlim([25, 80])
plt.ylim([-45, 0])
plt.show()

plt.figure()
x = out_blind.T[0]
y = out_blind.T[1]
plt.title("Predicted from the blind test set")
plt.scatter(x, y, alpha=1, s = .05)
plt.xlim([25, 80])
plt.ylim([-45, 0])
plt.show()

best_combinations_sorted_indexes = gs.vl_scores.argsort()

#prints the training and test accuracy, plus the average validation accuracy
print("training accuracy: ", tr_acc)
print("test accuracy: ", ts_acc)
print("model's average validation accuracy", gs.vl_scores[best_combinations_sorted_indexes[0]])

print("best combinations indexes sorted: ", best_combinations_sorted_indexes)

# print("out_blind:", out_blind)

#prints the 20 best performing models alongside their average training and
#validation accuracies during the 5 fold cv grid search.
threshold = 20
if(len(gs.params_combinations) < 20):
    threshold = len(gs.params_combinations)

for i in range(0, threshold):
    print(gs.params_combinations[best_combinations_sorted_indexes[i]], "; vl score: ", 
          gs.vl_scores[best_combinations_sorted_indexes[i]], " +- ", 
          gs.vl_scores_std[best_combinations_sorted_indexes[i]], "; tr score: ",
          gs.tr_scores[best_combinations_sorted_indexes[i]], " +- ", 
          gs.tr_scores_std[best_combinations_sorted_indexes[i]])
    
#saves the first 6 best models

best_model0 = gs.params_combinations[best_combinations_sorted_indexes[0]]
best_model1 = gs.params_combinations[best_combinations_sorted_indexes[1]]
best_model2 = gs.params_combinations[best_combinations_sorted_indexes[2]]
best_model3 = gs.params_combinations[best_combinations_sorted_indexes[3]]
best_model4 = gs.params_combinations[best_combinations_sorted_indexes[4]]
best_model5 = gs.params_combinations[best_combinations_sorted_indexes[5]]
"""