
# ---------------------------------- Imports ----------------------------------

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
 
# Code from other files in the repo
from utilities.data_analysis import ModelResults, ModelResultsMulti
from binary_classifier import EventNN

# Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------ Main -----------------------------------

SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

args_model = {'model_type' : 'binary_classifier',
              'model_architecture' : 'FNN',
              'layer_1_neurons' : 64,
              'layer_2_neurons' : 8,
              'learning_rate' : 0.001,
              'batch_size' : 64,
              'epochs' : 8,
              'model' : 'base2'}

num_runs = 10
dataset_sample = 0.5

test_param = 'layer_1_neurons'
param_values = np.linspace(4, 16, num=7, dtype=np.int)

all_results = ModelResultsMulti()
event_nn = EventNN(args_model)


#%%

event_nn.load_data(DIR)

#%%

event_nn.load_event_data(DIR)

for parameter in param_values:
    print(f'{test_param} = {parameter}')
    args_model[test_param] = parameter
    event_nn.args_model = args_model

    for i in range(num_runs):
        model_result = ModelResults(i)
        model_result.start_timer()
    
        event_nn.shuffle_data()
        event_nn.reduce_dataset(dataset_sample)
        event_nn.train_test_split(test_size=0.2)
        event_nn.create_model(args_model['model'])
        history = event_nn.train_model(verbose_level=0)
        
        # Calculate results
        model_result.training_history(history)
        model_result.confusion_matrix(event_nn, cutoff_threshold=0.5)
        model_result.roc_curve(event_nn)
        model_result.stop_timer(verbose=True)
        all_results.add_result(model_result, {test_param : parameter})

df_all_results = all_results.return_results()
# all_results.save('binary_event_nn_tuning.pkl')

# ------------------------------- Results plots -------------------------------

all_data = []
for parameter in param_values:
    test = df_all_results[df_all_results[test_param]==parameter]
    all_data.append(test['accuracy_test'].values)
    

fig, ax = plt.subplots(figsize=(6,4), dpi=200)
bplot = ax.boxplot(all_data, labels=param_values, showfliers=False, showmeans=True)

ax.yaxis.grid(True)
ax.set_xlabel('Number of neurons')
ax.set_ylabel('Model accuracy')

