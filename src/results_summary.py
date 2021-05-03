# -*- coding: utf-8 -*-
"""
This file is going to summarise the results from the project
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utilities.plotlib as pltlib 

# -------------------------------- load data ---------------------------------

df_nn = pd.read_pickle('results\\results_FFN.pkl')
df_nn.history_training_data = df_nn.history_training_data.apply(lambda x:np.array(x))
df_nn.history_test_data = df_nn.history_test_data.apply(lambda x:np.array(x))

df_rnn = pd.read_pickle('results\\results_RNN.pkl')
df_rnn.history_training_data = df_rnn.history_training_data.apply(lambda x:np.array(x))
df_rnn.history_test_data = df_rnn.history_test_data.apply(lambda x:np.array(x))

df_cnn = pd.read_pickle('results\\results_combined.pkl')
df_cnn.history_training_data = df_cnn.history_training_data.apply(lambda x:np.array(x))
df_cnn.history_test_data = df_cnn.history_test_data.apply(lambda x:np.array(x))

combined_results = [df_nn, df_rnn, df_cnn]

# ---------------------------- plot training curve ---------------------------

# params_history = {'title' : ('Model accuracy of feedforward neural network '
#                               'trained on event data'),
#                 'x_axis' : 'Epoch number',
#                 'y_axis' : 'Accuracy',
#                 'legend' : ['training data', 'test data'],
#                 'figsize' : (6, 4),
#                 'dpi' : 200,
#                 'colors' : ['#662E9B', '#F86624'],
#                 'full_y' : False}

# history_training_data_mean_nn = np.mean(df_nn.history_training_data, axis=0)
# history_test_data_mean_nn = np.mean(df_nn.history_test_data, axis=0)

# history_training_data_std_nn = np.std(df_nn.history_training_data.to_numpy(), axis=0)
# history_test_data_std_nn = np.std(df_nn.history_test_data.to_numpy(), axis=0)

# history_error_bars_nn = [history_training_data_std_nn, history_test_data_std_nn]

# fig1 = pltlib.training_history_plot(history_training_data_mean_nn,
#                               history_test_data_mean_nn,
#                               params_history,
#                               history_error_bars_nn)

# params_history = {'title' : ('Model accuracy of recurrent neural network '
#                               'trained on jet data'),
#                 'x_axis' : 'Epoch number',
#                 'y_axis' : 'Accuracy',
#                 'legend' : ['training data', 'test data'],
#                 'figsize' : (6, 4),
#                 'dpi' : 200,
#                 'colors' : ['#662E9B', '#F86624'],
#                 'full_y' : False}

# history_training_data_mean_rnn = np.mean(df_rnn.history_training_data, axis=0)
# history_test_data_mean_rnn = np.mean(df_rnn.history_test_data, axis=0)

# history_training_data_std_rnn = np.std(df_rnn.history_training_data.to_numpy(), axis=0)
# history_test_data_std_rnn = np.std(df_rnn.history_test_data.to_numpy(), axis=0)

# history_error_bars_rnn = [history_training_data_std_rnn, history_test_data_std_rnn]

# fig2 = pltlib.training_history_plot(history_training_data_mean_rnn,
#                               history_test_data_mean_rnn,
#                               params_history,
#                               history_error_bars_rnn)

# params_history = {'title' : ('Model accuracy of combined neural network '
#                               'trained on jet and event data'),
#                 'x_axis' : 'Epoch number',
#                 'y_axis' : 'Accuracy',
#                 'legend' : ['training data', 'test data'],
#                 'figsize' : (6, 4),
#                 'dpi' : 200,
#                 'colors' : ['#662E9B', '#F86624'],
#                 'full_y' : False}

# history_training_data_mean_cnn = np.mean(df_cnn.history_training_data, axis=0)
# history_test_data_mean_cnn = np.mean(df_cnn.history_test_data, axis=0)

# history_training_data_std_cnn = np.std(df_cnn.history_training_data.to_numpy(), axis=0)
# history_test_data_std_cnn = np.std(df_cnn.history_test_data.to_numpy(), axis=0)

# history_error_bars_cnn = [history_training_data_std_cnn, history_test_data_std_cnn]

# fig3 = pltlib.training_history_plot(history_training_data_mean_cnn,
#                               history_test_data_mean_cnn,
#                               params_history,
#                               history_error_bars_cnn)

# ----------------------------- Comparison plots ------------------------------

accuracy_results = [df.accuracy_test.to_list() for df in combined_results]
roc_auc_results = [df.roc_auc.to_list() for df in combined_results]
labels = ['FFN', 'RNN', 'FFN + RNN']

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fig4,ax4 = plt.subplots(figsize=(4,4), dpi=700)
ax4.yaxis.grid(True)
boxplot1 = ax4.boxplot(accuracy_results, patch_artist=True, showfliers=False)
ax4.set_xticklabels(labels)
ax4.set_xlabel('Network architecture')
ax4.set_ylabel('Accuracy')

fig5,ax5 = plt.subplots(figsize=(4,4), dpi=300)
ax5.yaxis.grid(True)
boxplot2 = ax5.boxplot(roc_auc_results, patch_artist=True, showfliers=False)
ax5.set_xticklabels(labels)
ax5.set_xlabel('Network architecture')
ax5.set_ylabel('roc auc')

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen']
for bplot in (boxplot1, boxplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
fig4.savefig('accuracy_boxplot.pdf')

# ---------------------------- Average ROC curve matrix -----------------------

# def mean_roc_curve(df,length=1000):
#     fpr_vals = []
#     tpr_vals = []
    
#     for i in range(df.shape[0]):
#         row = df.iloc[i]
#         l = row.roc_fpr_vals.shape[0]
#         indices = sorted(np.random.choice(l, length, replace=False))
#         fpr_vals.append(row.roc_fpr_vals[indices])
#         tpr_vals.append(row.roc_tpr_vals[indices])
        
#     fpr_vals = np.array(fpr_vals)
#     tpr_vals = np.array(tpr_vals)

#     mean_fpr_vals = np.mean(fpr_vals, axis=0)
#     mean_tpr_vals = np.mean(tpr_vals, axis=0)
    
#     return mean_fpr_vals, mean_tpr_vals
        
# mean_fpr_nn, mean_tpr_nn = mean_roc_curve(df_nn)
# mean_fpr_rnn, mean_tpr_rnn = mean_roc_curve(df_rnn)
# mean_fpr_cnn, mean_tpr_cnn = mean_roc_curve(df_cnn)

# mean_auc_nn = df_nn.roc_auc.mean()
# std_auc_nn = df_nn.roc_auc.std()

# mean_auc_rnn = df_rnn.roc_auc.mean()
# std_auc_rnn = df_rnn.roc_auc.std()

# mean_auc_cnn = df_cnn.roc_auc.mean()
# std_auc_cnn = df_cnn.roc_auc.std()

# label_nn = 'FFN auc = {:.3f} \u00B1 {:.3f}'.format(mean_auc_nn,std_auc_nn)
# label_rnn = 'RNN auc = {:.3f} \u00B1 {:.3f}'.format(mean_auc_rnn,std_auc_rnn)
# label_cnn = 'FFN + RNN auc = {:.3f} \u00B1 {:.3f}'.format(mean_auc_nn,std_auc_nn)

# fig6, ax6 = plt.subplots(figsize=(4,4), dpi=300)
# ax6.plot(mean_fpr_nn, mean_tpr_nn, label = label_nn)
# ax6.plot(mean_fpr_rnn, mean_tpr_rnn, label = label_rnn)
# ax6.plot(mean_fpr_cnn, mean_tpr_cnn, '--', label = label_cnn)

# ax6.plot([0, 1], [0, 1], 'k--')
    
# ax6.set_xlim([0.0, 1.005])
# ax6.set_ylim([0.0, 1.005])
# ax6.set_xlabel('False Positive Rate')
# ax6.set_ylabel('True Positive Rate')

    








