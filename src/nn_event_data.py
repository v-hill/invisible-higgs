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


# Get model predictions
labels_pred = model.predict(data_test)

# Convert predictions into binary values
cutoff_threshold = 0.5
labels_pred_binary = np.where(labels_pred > cutoff_threshold, 1, 0)

# Make confsuion matrix
cm = confusion_matrix(labels_test, labels_pred_binary)
class_names = ['signal', 'background']
title = f'Confusion matrix for threshold of {cutoff_threshold}'

# Plot confusion matrix
fig2 = plotlib.confusion_matrix(cm, class_names, title)


# Plot ROC curve
title_roc = 'ROC curve for event data model'
fig3 = plotlib.plot_roc(labels_pred, labels_test, title_roc)


# Plot distribution of discriminator values
labels_pred_signal = labels_pred[np.array(labels_test, dtype=bool)]
labels_pred_background = labels_pred[np.invert(np.array(labels_test, dtype=bool))]
title = "Distribution of discriminator values for the event-nn"
fig4 = plotlib.plot_discriminator_vals(labels_pred_signal, labels_pred_background, title)
