"""
This file contains the run code for a simple feedforward neural network to 
classify different event types.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
from binary_classifier import BinaryClassifier
import models.sequential_models as sequential_models
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResults, ModelResultsMulti

# Python libraries
import pickle
import numpy as np
import pandas as pd
import time

class EventNN(BinaryClassifier):
    """
    This class provides a wrapper for the data loading, model creating, model
    training and analysis of sequential keras models. 
    """
    def __init__(self, args_model):
        """
        Initialse the model by specifying the location of the input data as 
        well as the paramters to use in training the model.

        Parameters
        ----------
        args_model : dict
            Dictionary of model arguments.
        """
        super().__init__(args_model)
        
    def create_model(self, model_name):
        """
        Define which model from the models.sequential_models module to use and 
        compile it using the model arguments stored in self.args_model.

        Parameters
        ----------
        model_name : str
            Name of function in models.sequential_models to use to create the
            model.
        """
        if model_name=='base2':
            self.model = sequential_models.base2(self.args_model['layer_1_neurons'], 
                                    self.args_model['layer_2_neurons'], 
                                    input_shape=self.args_model['layer_input_shape'],
                                    learning_rate=self.args_model['learning_rate'])

    def train_model(self, verbose_level):
        """
        Trains the model for a fixed number of epochs.

        Parameters
        ----------
        verbose_level : int
            0, 1, or 2. Verbosity mode.

        Returns
        -------
        history : tensorflow.python.keras.callbacks.History
            It's History.history attribute is a record of training loss values 
            and metrics values at successive epochs, as well as validation 
            loss values and validation metrics values.
        """
        data_train = self.df_event_data.iloc[:self.test_train_split]
        data_test = self.df_event_data.iloc[self.test_train_split:]
        labels_train = self.df_labels['label_encoding'].iloc[:self.test_train_split]
        labels_test = self.df_labels['label_encoding'].iloc[self.test_train_split:]
        sample_weight = self.df_weights['sample_weight'].iloc[:self.test_train_split]
        
        history = self.model.fit(data_train, labels_train,
                                 batch_size = args_model['batch_size'],
                                 validation_data=(data_test, labels_test),
                                 sample_weight=sample_weight,
                                 epochs=args_model['epochs'],
                                 verbose=verbose_level)
        return history

    def predict_test_data(self):
        """
        Return predictions of trainined model for the test dataset.

        Returns
        -------
        labels_pred : numpy.ndarray
            Predicted labels for the test dataset.
        """
        data_test = self.df_event_data.iloc[self.test_train_split:]
        labels_pred = self.model.predict(data_test)
        return labels_pred
    
    def discriminator_values(self):
        labels_test = self.labels_test()
        labels_pred = self.predict_test_data()
        labels_pred_signal = labels_pred[np.array(labels_test, dtype=bool)]
        labels_pred_background = labels_pred[np.invert(np.array(labels_test, dtype=bool))]
        return labels_pred_signal, labels_pred_background

    def significance_dataset(self):
        # Make dataset
        labels_pred = self.predict_test_data()
        labels_test = self.df_labels['label_encoding'].iloc[self.test_train_split:]
        xs_weight = self.df_weights['xs_weight'].iloc[self.test_train_split:]
        
        dataset = pd.DataFrame(data=labels_pred, columns=['labels_pred'], index=labels_test.index)
        dataset['xs_weight'] = xs_weight*140000
        dataset['event_labels'] = labels_test
        return dataset
  
# ------------------------------------ Main -----------------------------------

SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

args_model = {'model_type' : 'binary_classifier',
              'model_architecture' : 'FNN',
              'layer_1_neurons' : 64,
              'layer_2_neurons' : 8,
              'learning_rate' : 0.0004,
              'batch_size' : 64,
              'epochs' : 8,
              'model' : 'base2'}

num_runs = 2

model_results_multi = ModelResultsMulti()
event_nn = EventNN(args_model)
event_nn.load_data(DIR)
event_nn.load_event_data(DIR)

for i in range(num_runs):
    START = time.time()
    event_nn.shuffle_data()
    event_nn.train_test_split(test_size=0.2)
    event_nn.create_model(args_model['model'])
    history = event_nn.train_model(verbose_level=0)
    
    # Calculate results
    model_results = ModelResults(i)
    model_results.training_history(history)
    model_results.confusion_matrix(event_nn, cutoff_threshold=0.5)
    model_results.roc_curve(event_nn)
    model_results_multi.add_result(model_results)
    print(f"    Run {i} time: {time.time()-START:0.2f}s")
    
df_results = model_results_multi.return_results()

# ------------------------ Miscellaneous results plots ------------------------

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

# Plot average training history
data_mean1, data_std1 = model_results_multi.average_training_history('history_training_data')
data_mean2, data_std2 = model_results_multi.average_training_history('history_test_data')
fig = plotlib.training_history_plot(data_mean1, data_mean2, params_history, error_bars=[data_std1, data_std2])
print(f'average training accuracy: {data_mean1[-1]:0.4f} \u00B1 {data_std1[-1]:0.4f}')
print(f'average test accuracy:     {data_mean2[-1]:0.4f} \u00B1 {data_std2[-1]:0.4f}')

# Get the index of the row with the best accuracy on the test dataset
idx_best = df_results['accuracy_training'].argmax()
df_model_results = df_results.iloc[idx_best]

# Plot best training history
fig = plotlib.training_history_plot(df_model_results['history_training_data'], 
                                      df_model_results['history_test_data'], 
                                      params_history)

# Plot best confusion matrix
fig = plotlib.confusion_matrix(df_model_results, params_cm)

# Plot best ROC curve
fig = plotlib.plot_roc(df_model_results, params_roc)

# Plot distribution of discriminator values
fig = plotlib.plot_discriminator_vals(*event_nn.discriminator_values(), params_discrim)

# Plot the signifiance plot
sig_x, sig_y = model_results.calc_significance(event_nn.significance_dataset(), num_thresholds=200)
fig = plotlib.significance(sig_x, sig_y, params_signif)
