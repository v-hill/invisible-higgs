"""
This file contains the code for preparing the jet data for an RNN neural network.
"""

# Code from other files in the repo
from utilities.data_loader import  DataLoader
import models.recurrent_models as recurrent_models
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker
from utilities.data_preprocessing import make_ragged_tensor

# Python libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

ROOT = "C:\\{Directory containing data}\\ml_postproc\\"
data_to_collect = ['ttH125_part1-1',
                   'ttH125_part1-2', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

# -------------------------------- Data setup --------------------------------

# Load in data
loader = DataLoader(ROOT)
loader.find_files()
loader.collect_data(data_to_collect)
data = DataProcessing(loader)

cols_jets = data.get_jet_columns()

signal_list = ['ttH125']
data.label_signal_noise(signal_list)
event_labels = LabelMaker.label_encoding(data.return_dataset_labels())
data.set_dataset_labels(event_labels)

# class_weight = WeightMaker.event_class_weights(data)
sample_weight = WeightMaker.weight_nominal_sample_weights(data)

# Select only the filtered columns from the data
data.filter_data(cols_jets)

# TODO: Normalise the jet data on a per variable basis

df = data.data
rt = make_ragged_tensor(df)

print(f"Shape: {rt.shape}")
print(f"Number of partitioned dimensions: {rt.ragged_rank}")
print(f"Flat values shape: {rt.flat_values.shape}")

# ------------------------------ Model training -------------------------------

model = recurrent_models.base()
history = model.fit(rt, 
                    event_labels, 
                    sample_weight=sample_weight, 
                    epochs=4, 
                    verbose=2)
