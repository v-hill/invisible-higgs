"""
This file contains the training on the RNN model on the jet data.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import models.recurrent_models as recurrent_models
from utilities.data_preprocessing import make_ragged_tensor
import utilities.plotlib as plotlib

# Python libraries
import pickle
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -------------------------------- Data setup --------------------------------

SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

# Load files
df_jet_data = pd.read_hdf(DIR+'preprocessed_jet_data.hdf')
sample_weight = np.load(DIR+'preprocessed_sample_weights.npy', allow_pickle=True)
encoding_dict = pickle.load(open(DIR+'encoding_dict.pickle', 'rb'))
event_labels = pd.read_hdf(DIR+'preprocessed_event_labels.hdf')
event_labels = event_labels.values

test_fraction = 0.2
data_train, data_test, labels_train, labels_test_rnn, sw_train, sw_test  = \
    train_test_split(df_jet_data, event_labels, 
                      sample_weight, test_size=test_fraction)

# Take a sample of the data to speed up training
sample_num = 10000
data_train = data_train[:sample_num]
data_test = data_test[:sample_num]
labels_train = labels_train[:sample_num]
labels_test_rnn = labels_test_rnn[:sample_num]
sw_train = sw_train[:sample_num]
sw_test = sw_test[:sample_num]

data_train_rt = make_ragged_tensor(data_train)
data_test_rt = make_ragged_tensor(data_test)
print(f"Shape: {data_train_rt.shape}")
print(f"Number of partitioned dimensions: {data_train_rt.ragged_rank}")
print(f"Flat values shape: {data_train_rt.flat_values.shape}")

# ------------------------------ Model training -------------------------------

model = recurrent_models.base()

print("Fitting RNN model on jet training data...")
START = time.time()
history = model.fit(data_train_rt, labels_train, batch_size = 64,
                    validation_data=(data_test_rt, labels_test_rnn), 
                    sample_weight=sw_train, epochs=32, verbose=2)
print(f"    Elapsed training time: {time.time()-START:0.2f}s")

test_loss, test_acc = model.evaluate(data_test_rt, labels_test_rnn, verbose=2)
print(f"    Test accuracy: {test_acc:0.5f}")

# --------------------------------- Plotting ----------------------------------

# Plot training history
fig1 = plotlib.training_history_plot(history, 'Jet RNN model accuracy')


# Get model predictions
labels_pred = model.predict(data_test_rt)

# Convert predictions into binary values
cutoff_threshold = 0.5 
labels_pred_binary = np.where(labels_pred > cutoff_threshold, 1, 0)

# Make confsuion matrix
cm = confusion_matrix(labels_test_rnn, labels_pred_binary)
class_names = ['signal', 'background']
title = 'Confusion matrix'

# Plot confusion matrix
fig2 = plotlib.confusion_matrix(cm, class_names, title)


# Plot ROC curve
title_roc = 'ROC curve for jet data RNN'
fig3 = plotlib.plot_roc(labels_pred, labels_test_rnn, title_roc)


# Plot distribution of discriminator values
bins = np.linspace(0, 1, 50)
fig = plt.figure(figsize=(6, 4), dpi=200)

plt.title("Distribution of discriminator values for the RNN")
plt.xlabel("Label prediction")
plt.ylabel("Density")

labels_pred_signal = labels_pred[np.array(labels_test_rnn, dtype=bool)]
labels_pred_background = labels_pred[np.invert(np.array(labels_test_rnn, dtype=bool))]

# plt.hist(labels_pred, bins, alpha=0.5, label='all events')
plt.hist(labels_pred_signal, bins, alpha=0.5, label='signal', color='brown')
plt.hist(labels_pred_background, bins, alpha=0.5, label='background', color='teal')

plt.legend(loc='upper right')
plt.show()
