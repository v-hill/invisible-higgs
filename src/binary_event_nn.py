"""
This file contains the run code for a simple feedforward neural network to 
classify different event types.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import binary_classifier as bcn
from binary_classifier import EventNN
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResultsMulti

# ------------------------------------ Main -----------------------------------

if __name__ == "__main__":
    DIR = 'data_binary_classifier\\'
    args_model = {'model_type' : 'binary_classifier',
                  'model_architecture' : 'EventNN',
                  'layer_1_neurons' : 64,
                  'layer_2_neurons' : 8,
                  'learning_rate' : 0.001,
                  'batch_size' : 64,
                  'epochs' : 8,
                  'model' : 'base2'}

    num_runs = 5
    dataset_sample = 0.25
    
    all_results = ModelResultsMulti()
    event_nn = EventNN(args_model)
    event_nn.load_data(DIR)
    event_nn.load_event_data(DIR)
    
    for i in range(num_runs):
        model_result = bcn.run(i, event_nn, args_model, dataset_sample)
        all_results.add_result(model_result, args_model)
    
    df_all_results = all_results.return_results()
    all_results.save('binary_event_nn.pkl')

# -------------------------- Results plots parameters -------------------------

params_history = {'title' : ('Model accuracy of feedforward neural network '
                              'trained on event data'),
                  'x_axis' : 'Epoch number',
                  'y_axis' : 'Accuracy',
                  'legend' : ['training data', 'test data'],
                  'figsize' : (6, 4),
                  'dpi' : 200,
                  'colors' : ['#662E9B', '#F86624'],
                  'full_y' : False}

params_cm = {'title' : ('Confusion matrix of feedforward neural network '
                              'trained on event data'),
             'x_axis' : 'Predicted label',
             'y_axis' : 'True label',
             'class_names' : ['ttH (signal)', 'tt¯ (background)'],
             'figsize' : (6, 4),
             'dpi' : 200,
             'colourbar' : False}

params_roc = {'title' : ('ROC curve for the feedforward neural network '
                              'trained on event data'),
              'x_axis' : 'False Positive Rate',
              'y_axis' : 'True Positive Rate',
              'figsize' : (6, 4),
              'dpi' : 200}

params_discrim = {'title' : ('Distribution of discriminator values for the '
                              'feedforward neural network trained on event data'),
                  'x_axis' : 'Label prediction',
                  'y_axis' : 'Number of events',
                  'num_bins' : 50,
                  'figsize' : (6, 4),
                  'dpi' : 200,
                  'colors' : ['brown', 'teal']}

params_signif = {'title' : ('Significance plot for the feedforward neural '
                              'network trained on event data'),
                  'x_axis' : 'Discrimintor threshold value',
                  'y_axis' : 's/sqrt(b)',
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

# ---------------------------- Plots of best result ---------------------------

# Get the index of the row with the best accuracy on the test dataset
idx_best = df_all_results['accuracy_training'].argmax()
df_model_result = df_all_results.iloc[idx_best]

# Plot best training history
fig = plotlib.training_history_plot(df_model_result['history_training_data'], 
                                    df_model_result['history_test_data'], 
                                    params_history)

# Plot best confusion matrix
fig = plotlib.confusion_matrix(df_model_result, params_cm)

# Plot best ROC curve
fig = plotlib.plot_roc(df_model_result, params_roc)

# Plot distribution of discriminator values
fig = plotlib.plot_discriminator_vals(*event_nn.discriminator_values(), params_discrim)

# Plot the signifiance plot
sig_x, sig_y = model_result.calc_significance(event_nn.significance_dataset(), num_thresholds=200)
fig = plotlib.significance(sig_x, sig_y, params_signif)
