"""
This file contains the run code for a simple feedforward neural network to 
classify different event types.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import models.sequential_models as sequential_models
import utilities.plotlib as plotlib

# Python libraries
import pickle
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from math import sqrt

# -------------------------------- Data setup --------------------------------

SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

# Load files
event_data = np.load(DIR+'preprocessed_event_data.npy', allow_pickle=True)
sample_weight = np.load(DIR+'preprocessed_sample_weights.npy', allow_pickle=True)
weight_nominal = np.load(DIR+'weight_nominal.npy', allow_pickle=True)
xs_weight = np.load(DIR+'xs_weight.npy', allow_pickle=True)
encoding_dict = pickle.load(open(DIR+'encoding_dict.pickle', 'rb'))
event_labels = pd.read_hdf(DIR+'preprocessed_event_labels.hdf')
event_labels = event_labels.values

test_fraction = 0.2
data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
    train_test_split(event_data, event_labels, 
                     sample_weight, test_size=test_fraction)

# Take a sample of the data to speed up training
sample_num = -1
data_train = data_train[:sample_num]
data_test = data_test[:sample_num]
labels_train = labels_train[:sample_num]
labels_test = labels_test[:sample_num]
sw_train = sw_train[:sample_num]
sw_test = sw_test[:sample_num]

# ------------------------------ Model training -------------------------------

INPUT_SHAPE = event_data.shape[1]
model = sequential_models.base2(64, 8, input_shape=INPUT_SHAPE, learning_rate=0.001)

print("Fitting sequential model on event training data...")
START = time.time()
history = model.fit(data_train, labels_train, batch_size = 128,
                    validation_data=(data_test, labels_test), 
                    sample_weight=sw_train, epochs=16, verbose=2)
print(f"    Elapsed training time: {time.time()-START:0.2f}s")

test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=2)
print(f"    Test accuracy: {test_acc:0.5f}")

# --------------------------------- Plotting ----------------------------------

# Plot training history
fig1 = plotlib.training_history_plot(history, 'Event neural network model accuracy')


# Get model predictions and convert predictions into binary values
labels_pred = model.predict(data_test)
cutoff_threshold = 0.5
labels_pred_binary = np.where(labels_pred > cutoff_threshold, 1, 0)

# Make confsuion matrix
cm = confusion_matrix(labels_test, labels_pred_binary)
class_names = ['ttH (signal)', 'tt¯ (background)']
title = f'Confusion matrix for discriminator threshold of {cutoff_threshold}'
fig2 = plotlib.confusion_matrix(cm, class_names, title)


# Plot ROC curve
title_roc = 'ROC curve for event data model'
fig3 = plotlib.plot_roc(labels_pred, labels_test, title_roc)


# Plot distribution of discriminator values
labels_pred_signal = labels_pred[np.array(labels_test, dtype=bool)]
labels_pred_background = labels_pred[np.invert(np.array(labels_test, dtype=bool))]
title = "Distribution of discriminator values for the event-nn"
fig4 = plotlib.plot_discriminator_vals(labels_pred_signal, labels_pred_background, title)

# Get model predictions
labels_pred = model.predict(event_data)
xs_weight = xs_weight.reshape((-1, 1))

def calc_significance(num_thresholds, dataset):
    bin_centres_sig = []
    bin_vals_sig = []
    bin_centres_back = []
    bin_vals_back = []
    
    z_vals = []
    z_vals2 = []
    thresholds = np.linspace(0, 1, num_thresholds)
    
    for i in range(len(thresholds)-1):
        
        df_selection = dataset[dataset['labels_pred'].between(thresholds[i], 1)]

        df_sig = df_selection[df_selection['event_labels']==1]
        df_back = df_selection[df_selection['event_labels']==0]
        sum_xs_weight_sig = df_sig['xs_weight'].sum()
        sum_xs_weight_back = df_back['xs_weight'].sum()
        
        if sum_xs_weight_sig==0 or sum_xs_weight_back==0:
            continue
        
        bin_centres_sig.append(thresholds[i])
        bin_vals_sig.append(sum_xs_weight_sig)
        bin_centres_back.append(thresholds[i])
        bin_vals_back.append(sum_xs_weight_back)
    
        s = sum_xs_weight_sig
        b = sum_xs_weight_back
        
        z = sqrt(2*((s+b)*np.log(1+(s/b))-s)) # Calculate significance 
        z_vals.append(z)
        z_vals2.append(s/(sqrt(b)))
    return bin_centres_sig, np.asarray(z_vals), np.asarray(z_vals2)


# Make dataset
dataset = pd.DataFrame(data=labels_pred, columns=['labels_pred'])
dataset['xs_weight'] = xs_weight*140000
dataset['event_labels'] = event_labels

bin_centres_sig, z_vals, z_vals2 = calc_significance(200, dataset)
print(f'Max significance at discriminator value of {bin_centres_sig[z_vals.argmax()]:0.3f}')


fig = plt.figure(figsize=(6, 4), dpi=200)
plt.title("Significance plot for event FFN binary classifier, using xs_weight")
plt.xlabel("Discrimintor threshold value")
plt.xlim(-0.1, 1)
plt.ylim(0, 8)
plt.ylabel("ZA")
plt.plot(bin_centres_sig, z_vals, '-', label='z_A')
plt.plot(bin_centres_sig, z_vals2, '-', label=' S/sqrt(B)')
plt.legend(loc='upper left')
