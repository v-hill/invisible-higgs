"""
This file contains the training script for a multilabel feedforward neural
network
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

# ---------------------------- Variable definitions --------------------------

# Possibel dataset_type
#['multi_classifier', 'multisignal_classifier']

dataset_type = 'multisignal_classifier'

if dataset_type == 'multi_classifier':    
    SAVE_FOLDER = 'data_multi_classifier'
else:
    SAVE_FOLDER = 'data_multisignal_classifier'

DIR = SAVE_FOLDER + '\\'

# -------------------------------- Data load-----------------------------------

#Load files
event_data = np.load(DIR+'preprocessed_event_data.npy', allow_pickle=True)
sample_weight = np.load(DIR+'preprocessed_sample_weights.npy', allow_pickle=True)
encoding_dict = pickle.load(open(DIR+'encoding_dict.pickle', 'rb'))
event_labels = pd.read_hdf(DIR+'preprocessed_event_labels.hdf')
event_labels = event_labels.values
n_classes = event_labels.shape[1]

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

model = sequential_models.multi_class_base(64, 8, input_shape=11, 
                                           output_shape = n_classes)

print("Fitting sequential model on event training data...")
START = time.time()
history = model.fit(data_train, labels_train, batch_size = 64,
                    validation_data=(data_test, labels_test), 
                    sample_weight=sw_train, epochs=32, verbose=2)
print(f"    Elapsed training time: {time.time()-START:0.2f}s")

test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=2)
print(f"    Test accuracy: {test_acc:0.5f}")

# --------------------------------- Plotting ----------------------------------

# Plot training history
fig1 = plotlib.training_history_plot(history, 'Event neural network model accuracy', dpi=200)

# Get model predictions
labels_pred = model.predict(data_test)

# Plot ROC curves
title = 'ROC curve for multi label classification event data'
class_labels = list(encoding_dict.keys())

if dataset_type == 'multi_classifier':    
    fig2 = plotlib.plot_multi_class_roc(labels_pred, labels_test, title, class_labels)
else:
    fig2 = plotlib.plot_multi_signal_roc(labels_pred, labels_test, title, class_labels)

# Transform data into binary
labels_pred = np.argmax(labels_pred, axis=1)
labels_test = np.argmax(labels_test, axis=1)

# Create confusion matrix
cm =  confusion_matrix(labels_test, labels_pred)
class_names = list(encoding_dict.keys())
title = 'Confusion matrix'

# Plot confusion matrix
fig3 = plotlib.confusion_matrix(cm, class_names, title, dpi=200)


























