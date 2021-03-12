"""
This script is used to process our data and generate pre-processed data files,
which can be fed directly in the neural network models.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker
from utilities.data_preprocessing import normalise_jet_columns

# Python libraries
import os
import copy
import numpy as np
import pickle

# ---------------------------- Variable definitions ---------------------------

ROOT = "C:\\users\\user\\Documents\\Fifth Year\\ml_postproc\\"
SAVE_FOLDER = 'data_multi_classifier'

data_to_collect = ['ttH125', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

binary_classifier = False
set_diJet_mass_nan_to_zero = True

# -------------------------------- Load in data -------------------------------

loader = DataLoader(ROOT)
loader.find_files()
loader.collect_data(data_to_collect)
data = DataProcessing(loader)

# ---------------------------- Create save folder -----------------------------

if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# --------------------------- Remove unwanted data ----------------------------

cols_to_ignore1 = ['entry', 'weight_nominal', 'hashed_filename', 'BiasedDPhi']
cols_to_ignore2 = ['cleanJetMask']

cols_events = data.get_event_columns(cols_to_ignore1)
cols_jets = data.get_jet_columns(cols_to_ignore2)

# Clean DiJet_mass values
if set_diJet_mass_nan_to_zero:
    data.set_nan_to_zero('DiJet_mass')
else:
    data.remove_nan('DiJet_mass')

# Removes all events with less than two jets
data.data = data.data[data.data.ncleanedJet > 1]

# ------------------------------ Label_encoding -------------------------------

if binary_classifier:
    signal_list = ['ttH125']
    data.label_signal_noise(signal_list)
    event_labels, encoding_dict = LabelMaker.label_encoding(data.return_dataset_labels())
else:
    event_labels, encoding_dict = LabelMaker.onehot_encoding(data.return_dataset_labels())
    
# class_weight = WeightMaker.event_class_weights(data)
sample_weight = WeightMaker.weight_nominal_sample_weights(data)

# Make seperate copies of the dataset
event_data = copy.deepcopy(data)
jet_data = copy.deepcopy(data)

# Select only the event columns from the data
event_data.filter_data(cols_events)

# Select only the jet columns from the data
jet_data.filter_data(cols_jets)

cols_to_log = ['HT', 'MHT_pt', 'MetNoLep_pt']
event_data.nat_log_columns(cols_to_log)

min_max_scale_range = (0, 1)
event_data.normalise_columns(min_max_scale_range)

df_jet_data = normalise_jet_columns(jet_data.data)
df_event_data = event_data.data

# -------------------------------- Data saving --------------------------------

DIR = SAVE_FOLDER + '\\'

pickle.dump(encoding_dict, open(DIR+'encoding_dict.pickle', 'wb' ))

np.save(DIR+'preprocessed_event_data', event_data.data)
np.save(DIR+'preprocessed_sample_weights', sample_weight)

event_labels.to_hdf(DIR+'preprocessed_event_labels.hdf', key='df', mode='w')
df_jet_data.to_hdf(DIR+'preprocessed_jet_data.hdf', key='dfj', mode='w')
