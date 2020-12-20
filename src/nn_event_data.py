# -*- coding: utf-8 -*-
"""
This file contains the run code for a simple feedforward neural network to 
classify different event types.
"""

# Code from other files in the repo
import models.sequential_models as sequential_models
from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker

# Python libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

ROOT = "C:\\{Directory containing data}\\ml_postproc\\"
data_to_collect = ['ttH125_part1-1', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

# -------------------------------- Data setup --------------------------------

# Load in data
loader = DataLoader(ROOT)
loader.find_files()
loader.collect_data(data_to_collect)
data = DataProcessing(loader)

cols_to_ignore = ['entry', 'weight_nominal', 'hashed_filename']
cols_events = data.get_event_columns(cols_to_ignore)
# cols_jets = data.get_jet_columns()

data.set_nan_to_zero('DiJet_mass')
# data.remove_nan('DiJet_mass')

signal_list = ['ttH125']
data.label_signal_noise(signal_list)
#event_labels = LabelMaker.onehot_encoding(data.return_dataset_labels())
event_labels = LabelMaker.label_encoding(data.return_dataset_labels())
data.set_dataset_labels(event_labels)

# class_weight = WeightMaker.event_class_weights(data)
sample_weight = WeightMaker.weight_nominal_sample_weights(data)

# Select only the filtered columns from the data
data.filter_data(cols_events)

cols_to_log = ['HT', 'MHT_pt', 'MetNoLep_pt']
data.nat_log_columns(cols_to_log)

min_max_scale_range = (0, 1)
data.normalise_columns(min_max_scale_range)

test_fraction = 0.2
data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
    train_test_split(data.data, event_labels, sample_weight, test_size=test_fraction)

# ------------------------------ Model training -------------------------------

model = sequential_models.base(42, 4)

print("Fit sequential model on training data...")
START = time.time()
history = model.fit(data_train, labels_train, validation_data=(data_test, labels_test), sample_weight=sw_train, epochs=16, verbose=2)
print(f"    Elapsed training time: {time.time()-START:0.2f}s")

test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=2)
print(f"    Test accuracy: {test_acc:0.5f}")

# --------------------------------- Plotting ----------------------------------

# Plot training history
fig1 = plotlib.training_history_plot(history, 'Event neural network model accuracy')

# Make confsuion matrix
labels_pred = model.predict(data_test)
labels_pred = np.argmax(labels_pred, axis=1)
cm = confusion_matrix(labels_test, labels_pred)
class_names = ['signal', 'background']
title = 'Confusion matrix'

# Plot confusion matrix
fig2 = plotlib.confusion_matrix(cm, class_names, title)
