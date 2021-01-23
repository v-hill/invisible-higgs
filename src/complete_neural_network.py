"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# Code from other files in the repo
from utilities.data_loader import  DataLoader
import models.sequential_models as sequential_models
import models.recurrent_models as recurrent_models
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker
from utilities.data_preprocessing import make_ragged_tensor
# from utilities.data_preprocessing import split_data

# Python libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

#--------------------------------Load data------------------------------------

ROOT = "C:\\Users\\user\\Documents\\Fifth Year\\ml_postproc"
data_to_collect = ['ttH125_part1-1', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

# Load in data
loader = DataLoader(ROOT)
loader.find_files()
loader.collect_data(data_to_collect)
data = DataProcessing(loader)

cols_to_ignore = ['entry', 'weight_nominal', 'hashed_filename']
cols_events = data.get_event_columns(cols_to_ignore)

cols_to_ignore = ['cleanJetMask']
cols_jets = data.get_jet_columns(cols_to_ignore)

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
data.filter_data(cols_events + cols_jets)

cols_to_log = ['HT', 'MHT_pt', 'MetNoLep_pt']
data.nat_log_columns(cols_to_log)

min_max_scale_range = (0, 1)

#When using the argument col_events the data.data object remains a dataframe
data.normalise_columns(min_max_scale_range, cols_events) 

test_fraction = 0.2
data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
    train_test_split(data.data, event_labels, sample_weight, test_size=test_fraction)

#The data now needs to be split. The sequential model will be trained on the 
#event data and the recurent model will be trained on the jet data once it has
#been transformed to a ragged tensor

event_data_train_df = data_train[cols_events]
event_data_test_df = data_test[cols_events]

#Jet data needs to be turned into a raged tensor
jet_data_train_df = data_train[cols_jets] 
jet_data_test_df = data_test[cols_jets]

jet_data_train_rt = make_ragged_tensor(jet_data_train_df)
jet_data_test_rt = make_ragged_tensor(jet_data_test_df)

# TODO: Explore using parallel progrmaing to train each model

#--------------------------------Train models----------------------------------

#Sequential model

model1 = sequential_models.base(42, 4)

print("Fit sequential model on training data...")
START = time.time()
history = model1.fit(event_data_train_df, labels_train, validation_data=(event_data_test_df, labels_test), sample_weight=sw_train, epochs=16, verbose=2)
print(f"    Elapsed training time: {time.time()-START:0.2f}s")

test_loss_seq_nn, test_acc_seq_nn = model1.evaluate(event_data_test_df, labels_test, verbose=2)
print(f"    Test accuracy: {test_acc_seq_nn:0.5f}")

# Long short term memory model

model2 = recurrent_models.base()
history = model2.fit(jet_data_train_rt, 
                    labels_train, 
                    sample_weight=sw_train, 
                    epochs=4, 
                    verbose=2)

models = [model1 + model2]

test_loss_rnn, test_acc_rnn = model1.evaluate(jet_data_test_rt, labels_test, verbose=2)

#--------------------------------Combine models--------------------------------

def ensemble(models,test_data):
    pass










