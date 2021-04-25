"""
This file contains the training on the RNN model on the jet data.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import binary_classifier as bcn
from binary_classifier import JetRNN
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResultsMulti

# ------------------------------------ Main -----------------------------------

DIR = 'data_binary_classifier\\'
args_model = {'model_type' : 'binary_classifier',
              'model_architecture' : 'JetRNN',
              'layer_1_neurons' : 64,
              'layer_2_neurons' : 8,
              'output_shape' : 1,
              'learning_rate' : 0.001,
              'batch_size' : 64,
              'epochs' : 8,
              'model' : 'base'}

num_runs = 10
dataset_sample = 0.25

all_results = ModelResultsMulti()
jet_rnn = JetRNN(args_model)
jet_rnn.load_data(DIR)
jet_rnn.load_jet_data(DIR)

for i in range(num_runs):
    model_result = bcn.run(i, jet_rnn, args_model, dataset_sample)
    all_results.add_result(model_result, args_model)

df_all_results = all_results.return_results()
all_results.save('binary_jet_rnn.pkl')

# -------------------------- Results plots parameters -------------------------

params_history = {'title' : ('Model accuracy of recurrent neural network '
                             'trained on jet data'),
                'x_axis' : 'Epoch number',
                'y_axis' : 'Accuracy',
                'legend' : ['training data', 'test data'],
                'figsize' : (6, 4),
                'dpi' : 200,
                'colors' : ['#662E9B', '#F86624'],
                'full_y' : False}

params_cm = {'title' : ('Confusion matrix of recurrent neural network '
                              'trained on jet data'),
              'x_axis' : 'Predicted label',
              'y_axis' : 'True label',
              'class_names' : ['ttH (signal)', 'tt¯ (background)'],
              'figsize' : (6, 4),
              'dpi' : 200,
              'colourbar' : False}

params_roc = {'title' : ('ROC curve for the recurrent neural network '
                              'trained on jet data'),
              'x_axis' : 'False Positive Rate',
              'y_axis' : 'True Positive Rate',
              'figsize' : (6, 4),
              'dpi' : 200}

# --------------------------- Averaged results plots --------------------------

# Plot average training history
data_mean1, data_std1 = all_results.average_training_history('history_training_data')
data_mean2, data_std2 = all_results.average_training_history('history_test_data')
fig = plotlib.training_history_plot(data_mean1, data_mean2, params_history, 
                                    error_bars=[data_std1, data_std2])
print(f'average training accuracy: {data_mean1[-1]:0.4f} \u00B1 {data_std1[-1]:0.4f}')
print(f'average test accuracy:     {data_mean2[-1]:0.4f} \u00B1 {data_std2[-1]:0.4f}')

# Plot average confusion matrix
data_mean1, data_std1 = all_results.average_confusion_matrix()
fig = plotlib.confusion_matrix(data_mean1, params_cm)

# Plot average ROC curve
fig = plotlib.plot_roc(all_results.average_roc_curve(), params_roc)
