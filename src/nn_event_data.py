"""
This file contains the run code for a simple feedforward neural network to 
classify different event types.
"""

# Code from other files in the repo
import models.sequential_models as sequential_models
import utilities.plotlib as plotlib

# Python libraries
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data_to_collect = ['ttH125_part1-1', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

# -------------------------------- Data setup --------------------------------

event_data = np.load('preprocessed_event_data.npy', allow_pickle=True)
event_labels = np.load('preprocessed_event_labels.npy', allow_pickle=True)
sample_weight = np.load('preprocessed_sample_weights.npy', allow_pickle=True)

test_fraction = 0.2
data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
    train_test_split(event_data, event_labels, 
                     sample_weight, test_size=test_fraction)

# ------------------------------ Model training -------------------------------

model = sequential_models.base2(42, 4)

print("Fitting sequential model on event training data...")
START = time.time()
history = model.fit(data_train, labels_train, 
                    validation_data=(data_test, labels_test), 
                    sample_weight=sw_train, epochs=16, verbose=2)
print(f"    Elapsed training time: {time.time()-START:0.2f}s")

test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=2)
print(f"    Test accuracy: {test_acc:0.5f}")

# # --------------------------------- Plotting ----------------------------------

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
title = 'Confusion matrix'

# Plot confusion matrix
fig2 = plotlib.confusion_matrix(cm, class_names, title)

# Plot ROC curve
title_roc = 'ROC curve for event data model'
fig = plotlib.plot_roc(labels_pred, labels_test, title_roc)
