# -*- coding: utf-8 -*-
"""
This test will perform the same experiment shown in test_adam.py by using the
Keras library, in order to show how the fluctuations are a problem in Keras as well.

The hyperparameters are the same as in test_adam.py

@author: 
    Amadei Davide (d.amadei@studenti.unipi.it)    
    Ninniri Matteo (m.ninniri1@studenti.unipi.it)
"""

from keras import backend as K
import pandas as pd
import tensorflow as tf


from keras import activations, regularizers
from keras.models import Sequential
from keras.layers import Dense
    
from tensorflow.keras import optimizers

folder = "Datasets/"
train = "ML-CUP20-TR.csv"
test = "ML-CUP20-TS.csv"

training_set_initial = pd.read_csv(folder + train, sep = ',', header = None,
                                   error_bad_lines = False, comment='#', index_col = 0)

training_labels = training_set_initial[[11, 12]].to_numpy()
training_set = training_set_initial[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].to_numpy()


from BaseFoldGenerator import StandardKFold
foldgen = StandardKFold(k = .9, random_state = 0)
folds = foldgen.create_fold(X = training_set, y = training_labels)[0]
tr_set_index = folds[0]
ts_set_index = folds[1]

blind_test_set = pd.read_csv(folder + test, sep = ',', header = None,
                                   error_bad_lines = False, comment='#', index_col = 0).to_numpy()

tr_set = training_set[tr_set_index]
tr_set_labels = training_labels[tr_set_index]
ts_set = training_set[ts_set_index]
ts_set_labels = training_labels[ts_set_index]

batch_size = tr_set.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((tr_set, tr_set_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((ts_set, ts_set_labels))
val_dataset = val_dataset.batch(batch_size)


###############################################################################
print("======================================================================")
opt = optimizers.Adam(learning_rate = .0025)
loss_fn = tf.keras.losses.MeanSquaredError()

def euclidean_distance_loss(y_true, y_pred):
    diff = y_pred - y_true
    result = tf.reduce_mean(K.sqrt(K.sum(K.square(diff), axis=-1)))
    return result

def get_gradient_norm(model):
    with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
    return norm

model = Sequential()

model.add(Dense(120, 
                activation = activations.tanh, 
                kernel_initializer = 'glorot_uniform',
                kernel_regularizer = regularizers.l2(.00001),
                input_shape = (tr_set.shape[1],)))

model.add(Dense(tr_set_labels.shape[1], 
                activation = activations.linear, 
                kernel_regularizer = regularizers.l2(.00001),
                kernel_initializer = 'glorot_uniform'))

model.compile(loss = loss_fn, 
              optimizer = opt,
              metrics = [euclidean_distance_loss])

mse_losses = []
mee_losses = []
norms = []
epochs = 8000


for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        #flatten the tensors
        flattenedList = [K.flatten(x) for x in grads]

        #concatenate them
        concatenated = K.concatenate(flattenedList)
        norm = tf.norm(concatenated)
        
        if(epoch == 0):
            iter_0_norm = norm
            
        actual_norm = float(norm/iter_0_norm)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        opt.apply_gradients(zip(grads, model.trainable_weights))

        mse_losses.append(loss_value)
        norms.append(actual_norm)
        
        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f, gradient norm = %.10f"
                % (step, float(loss_value), actual_norm)
            )


out_blind = model.predict(tr_set)
out_ts = model.predict(ts_set)

from BaseLoss import MEE
acc_mee = MEE().calculate_loss_value(out_ts.T, ts_set_labels.T)

from matplotlib import pyplot as plt
plt.figure(dpi = 500)
plt.plot(mse_losses)
plt.title('keras model loss')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.ylim([0, 15])
plt.show()

plt.figure(dpi = 500)
plt.plot(norms)
plt.title('gradient norms')
plt.ylabel('norms')
plt.xlabel('epoch')
plt.ylim([0, .01])
plt.show()