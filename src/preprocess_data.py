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
    
