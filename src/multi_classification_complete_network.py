"""
This file contains the training script for a multilabel neural network. The 
model is a concatination of a feed forward neural network and a reccurent network.
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

#------------------------------- Load event data ------------------------------

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
sample_num = 50000
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

model = combined_models.multi_label_base(input_shape1=11, 
                              input_shape2=[None, 6],
                              output_shape=4)

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

# Plot ROC curves
title = 'ROC curve for multi label classification complete network'
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
fig3 = plotlib.confusion_matrix(cm, class_names, title)



