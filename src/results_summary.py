# -*- coding: utf-8 -*-
"""
This file is going to summarise the results from the project
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utilities.plotlib as pltlib 

# -------------------------------- load data ---------------------------------

df_nn = pd.read_pickle('results.pkl')
df_nn.history_training_data = df_nn.history_training_data.apply(lambda x:np.array(x))
df_nn.history_test_data = df_nn.history_test_data.apply(lambda x:np.array(x))

df_rnn = pd.read_pickle('results_rnn.pkl')
df_rnn.history_training_data = df_rnn.history_training_data.apply(lambda x:np.array(x))
df_rnn.history_test_data = df_rnn.history_test_data.apply(lambda x:np.array(x))

df_cnn = pd.read_pickle('results_rnn.pkl')
df_cnn.history_training_data = df_cnn.history_training_data.apply(lambda x:np.array(x))
df_cnn.history_test_data = df_cnn.history_test_data.apply(lambda x:np.array(x))

# ---------------------------- plot training curve ---------------------------

params_history = {'title' : ('Model accuracy of feedforward neural network '
                              'trained on event data'),
                'x_axis' : 'Epoch number',
                'y_axis' : 'Accuracy',
                'legend' : ['training data', 'test data'],
                'figsize' : (6, 4),
                'dpi' : 200,
                'colors' : ['#662E9B', '#F86624'],
                'full_y' : False}

history_training_data_mean_nn = np.mean(df_nn.history_training_data, axis=0)
history_test_data_mean_nn = np.mean(df_nn.history_test_data, axis=0)

history_training_data_std_nn = np.std(df_nn.history_training_data.to_numpy(), axis=0)
history_test_data_std_nn = np.std(df_nn.history_test_data.to_numpy(), axis=0)

history_error_bars_nn = [history_training_data_std_nn, history_test_data_std_nn]

pltlib.training_history_plot(history_training_data_mean_nn,
                             history_test_data_mean_nn,
                             params_history,
                             history_error_bars_nn)

params_history = {'title' : ('Model accuracy of recurrent neural network '
                             'trained on jet data'),
                'x_axis' : 'Epoch number',
                'y_axis' : 'Accuracy',
                'legend' : ['training data', 'test data'],
                'figsize' : (6, 4),
                'dpi' : 200,
                'colors' : ['#662E9B', '#F86624'],
                'full_y' : False}

history_training_data_mean_rnn = np.mean(df_rnn.history_training_data, axis=0)
history_test_data_mean_rnn = np.mean(df_rnn.history_test_data, axis=0)

history_training_data_std_rnn = np.std(df_rnn.history_training_data.to_numpy(), axis=0)
history_test_data_std_rnn = np.std(df_rnn.history_test_data.to_numpy(), axis=0)

history_error_bars_rnn = [history_training_data_std_rnn, history_test_data_std_rnn]

pltlib.training_history_plot(history_training_data_mean_rnn,
                             history_test_data_mean_rnn,
                             params_history,
                             history_error_bars_rnn)










