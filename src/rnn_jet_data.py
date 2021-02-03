"""
This file contains the code for preparing the jet data for an RNN neural network.
"""

# Code from other files in the repo
import models.recurrent_models as recurrent_models
from utilities.data_preprocessing import make_ragged_tensor
import utilities.plotlib as plotlib

# Python libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ROOT = "C:\\{Directory containing data}\\ml_postproc\\"
data_to_collect = ['ttH125_part1-1',
                   'ttH125_part1-2', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

# -------------------------------- Data setup --------------------------------

# Load in data
df_jet_data = pd.read_hdf('preprocessed_jet_data.hdf')
event_labels = np.load('preprocessed_event_labels.npy', allow_pickle=True)
sample_weight = np.load('preprocessed_sample_weights.npy', allow_pickle=True)

# TODO: Normalise the jet data on a per variable basis

df = df_jet_data

test_fraction = 0.2
data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
    train_test_split(df_jet_data, event_labels, 
                     sample_weight, test_size=test_fraction)


data_train_rt = rt_jet_data = make_ragged_tensor(data_train)
data_test_rt = rt_jet_data = make_ragged_tensor(data_test)
print(f"Shape: {data_train_rt.shape}")
print(f"Number of partitioned dimensions: {data_train_rt.ragged_rank}")
print(f"Flat values shape: {data_train_rt.flat_values.shape}")

# # ------------------------------ Model training -------------------------------

model = recurrent_models.base()

print("Fitting RNN model on jet training data...")
START = time.time()
history = model.fit(data_train_rt, labels_train, 
                    validation_data=(data_test_rt, labels_test), 
                    sample_weight=sw_train, epochs=16, verbose=2)
print(f"    Elapsed training time: {time.time()-START:0.2f}s")

test_loss, test_acc = model.evaluate(data_test_rt, labels_test, verbose=2)
print(f"    Test accuracy: {test_acc:0.5f}")

# --------------------------------- Plotting ----------------------------------

# Plot training history
fig1 = plotlib.training_history_plot(history, 'Jet RNN model accuracy')

# Make confsuion matrix
labels_pred = model.predict(data_test_rt)
labels_pred = np.argmax(labels_pred, axis=1)
cm = confusion_matrix(labels_test, labels_pred)
class_names = ['signal', 'background']
title = 'Confusion matrix'

# Plot confusion matrix
fig2 = plotlib.confusion_matrix(cm, class_names, title)
