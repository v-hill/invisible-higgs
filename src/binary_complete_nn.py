"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# ---------------------------------- Imports ----------------------------------

# Tell tensorflow to ignore GPU device and run on CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Code from other files in the repo
from utilities.data_preprocessing import make_ragged_tensor
import models.combined_models as combined_models
import utilities.plotlib as plotlib

# Python libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#------------------------------- Load event data ------------------------------

SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

# Load files
event_data = np.load(DIR+'preprocessed_event_data.npy', allow_pickle=True)
sample_weight = np.load(DIR+'preprocessed_sample_weights.npy', allow_pickle=True)
encoding_dict = pickle.load(open(DIR+'encoding_dict.pickle', 'rb'))
event_labels = pd.read_hdf(DIR+'preprocessed_event_labels.hdf')
event_labels = event_labels.values

# Generate a fixed random state
random_state = np.random.randint(50)
test_fraction = 0.2

data_train1, data_test1, labels_train, labels_test, sw_train, sw_test  = \
            train_test_split(event_data,
                             event_labels,
                             sample_weight,
                             test_size=test_fraction,
                             random_state=random_state)
            
# Take a sample of the data to speed up training
sample_num = 10000
data_train1 = data_train1[:sample_num]
data_test1 = data_test1[:sample_num]

#-------------------------------- Load jet data -------------------------------

# Load files
df_jet_data = pd.read_hdf(DIR+'preprocessed_jet_data.hdf')

data_train2, data_test2, labels_train, labels_test, sw_train, sw_test  = \
            train_test_split(df_jet_data,
                             event_labels,
                             sample_weight,
                             test_size=test_fraction,
                             random_state=random_state)

# Take a sample of the data to speed up training
data_train2 = data_train2[:sample_num]
data_test2 = data_test2[:sample_num]
labels_train = labels_train[:sample_num]
labels_test = labels_test[:sample_num]
sw_train = sw_train[:sample_num]
sw_test = sw_test[:sample_num]

jet_data_train_rt = make_ragged_tensor(data_train2)
jet_data_test_rt = make_ragged_tensor(data_test2)

# ------------------------------ Model training -------------------------------

model = combined_models.base1(input_shape1=11, 
                              input_shape2=[None, 6])

history = model.fit(x=[data_train1, jet_data_train_rt], y=labels_train, 
                    validation_data=([data_test1, jet_data_test_rt], labels_test), 
                    sample_weight=sw_train, 
                    epochs=16, 
                    verbose=2)

test_loss, test_acc = model.evaluate([data_test1, jet_data_test_rt], labels_test, verbose=2)
print(f"    Test accuracy: {test_acc:0.5f}")

# --------------------------------- Plotting ----------------------------------

# Plot training history
fig1 = plotlib.training_history_plot(history, 'Event neural network model accuracy')


# Get model predictions
labels_pred = model.predict([data_test1, jet_data_test_rt])

# Convert predictions into binary values
cutoff_threshold = 0.5 
labels_pred_binary = np.where(labels_pred > cutoff_threshold, 1, 0)

# Make confsuion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred_binary)
class_names = ['signal', 'background']
title = 'Confusion matrix'

# Plot confusion matrix
fig2 = plotlib.confusion_matrix(cm, class_names, title)


# Plot ROC curve
title_roc = 'ROC curve for event data model'
fig = plotlib.plot_roc(labels_pred, labels_test, title_roc)


# Plot distribution of discriminator values
bins = np.linspace(0, 1, 50)
fig = plt.figure(figsize=(6, 4), dpi=200)

plt.title("Distribution of discriminator values for the RNN")
plt.xlabel("Label prediction")
plt.ylabel("Density")

labels_pred_signal = labels_pred[np.array(labels_test, dtype=bool)]
labels_pred_background = labels_pred[np.invert(np.array(labels_test, dtype=bool))]

# plt.hist(labels_pred, bins, alpha=0.5, label='all events')
plt.hist(labels_pred_signal, bins, alpha=0.5, label='signal', color='brown')
plt.hist(labels_pred_background, bins, alpha=0.5, label='background', color='teal')

plt.legend(loc='upper right')
plt.show()
