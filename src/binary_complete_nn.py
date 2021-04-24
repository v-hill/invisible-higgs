"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import binary_classifier as bcn
from binary_classifier import CombinedNN
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResultsMulti

# ------------------------------------ Main -----------------------------------

DIR = 'data_binary_classifier\\'
args_model = {'model_type' : 'binary_classifier',
              'model_architecture' : 'CombinedNN',
              'batch_size' : 64,
              'epochs' : 8,
              'model' : 'base1'}

num_runs = 1
dataset_sample = 0.25

all_results = ModelResultsMulti()
neural_net = CombinedNN(args_model)  
neural_net.load_data(DIR)
neural_net.load_event_data(DIR)
neural_net.load_jet_data(DIR)

for i in range(num_runs):
    model_result = bcn.run(i, neural_net, args_model, dataset_sample)
    all_results.add_result(model_result, args_model)

df_all_results = all_results.return_results()
