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
df_jet_data = pd.read_hdf('preprocessed_jet_data.hdf')

# TODO: Normalise the jet data on a per variable basis

df = data.data
rt = rt_jet_data = make_ragged_tensor(df_jet_data)

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
