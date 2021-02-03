"""
This script is used to process our data and generate pre-processed data files,
which can be fed directly in the neural network models.
"""

# Code from other files in the repo
from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker

# Python libraries
import copy
import numpy as np

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

cols_to_ignore1 = ['entry', 'weight_nominal', 'hashed_filename']
cols_to_ignore2 = ['cleanJetMask']

cols_events = data.get_event_columns(cols_to_ignore1)
cols_jets = data.get_jet_columns(cols_to_ignore2)

data.set_nan_to_zero('DiJet_mass')
# data.remove_nan('DiJet_mass')

signal_list = ['ttH125']
data.label_signal_noise(signal_list)
#event_labels = LabelMaker.onehot_encoding(data.return_dataset_labels())
event_labels = LabelMaker.label_encoding(data.return_dataset_labels())
data.set_dataset_labels(event_labels)

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

df_jet_data = jet_data.data
df_event_data = event_data.data

# This step should be in the run script
# rt_jet_data = make_ragged_tensor(df_jet_data)
    
# -------------------------------- Data saving --------------------------------

np.save('preprocessed_event_data', event_data)
np.save('preprocessed_event_labels', event_labels)
np.save('preprocessed_sample_weights', sample_weight)

df_jet_data.to_hdf('preprocessed_jet_data.hdf', key='dfj', mode='w')

