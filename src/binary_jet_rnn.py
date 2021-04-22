"""
This file contains the training on the RNN model on the jet data.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
from binary_classifier import BinaryClassifier
import models.recurrent_models as recurrent_models
from utilities.data_preprocessing import make_ragged_tensor
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResults, ModelResultsMulti

# Python libraries
import tensorflow as tf
import time

# ----------------------------- Class definitions -----------------------------

class JetRNN(BinaryClassifier):
    """
    This class provides a wrapper for the data loading, model creating, model
    training and analysis of recurrent neural network keras models. 
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
        self.model = None
        
    def make_ragged_tensor(self, verbose=False):
        """
        Transforms a pandas dataframe into a tf.RaggedTensor object.

        Parameters
        ----------
        verbose : bool, optional
            Prints execution time if True. The default is False.
        """
        START = time.time()
        self.jet_data_rt = make_ragged_tensor(self.df_jet_data.iloc[:self.dataset_end])
        if verbose:
            print(f"    Elapsed ragged tensor creation time: {time.time()-START:0.3f}s")
            print(f'Shape: {self.jet_data_rt.shape}')
            print(f'Number of partitioned dimensions: {self.jet_data_rt.ragged_rank}')
            print(f'Flat values shape: {self.jet_data_rt.flat_values.shape}')
            

    def create_model(self, model_name):
        """
        Define which model from the models.sequential_models module to use.

        Parameters
        ----------
        model_name : str
            Name of function in models.recurrent_models to use to create the
            model.
        """
        tf.keras.backend.clear_session()
        del self.model
        if model_name == 'base':
            self.model = recurrent_models.base(self.args_model['layer_1_neurons'], 
                                    self.args_model['layer_2_neurons'], 
                                    input_shape=self.args_model['jet_layer_input_shape'])

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
        data_train = self.jet_data_rt[:self.test_train_split]
        data_test = self.jet_data_rt[self.test_train_split:self.dataset_end]
        labels_train = self.df_labels['label_encoding'].iloc[:self.test_train_split].values
        labels_test = self.df_labels['label_encoding'].iloc[self.test_train_split:self.dataset_end].values
        sample_weight = self.df_weights['sample_weight'].iloc[:self.test_train_split].values
        
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
        data_test = self.jet_data_rt[self.test_train_split:self.dataset_end]
        labels_pred = self.model.predict(data_test)
        return labels_pred
    
# ------------------------------------ Main -----------------------------------

SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

args_model = {'model_type' : 'binary_classifier',
              'model_architecture' : 'RNN',
              'layer_1_neurons' : 16,
              'layer_2_neurons' : 8,
              'batch_size' : 64,
              'epochs' : 8,
              'model' : 'base'}

num_runs = 5
dataset_sample = 0.25

model_results_multi = ModelResultsMulti()
jet_rnn = JetRNN(args_model)
jet_rnn.load_data(DIR)
jet_rnn.load_jet_data(DIR)

for i in range(num_runs):
    model_results = ModelResults(i)
    model_results.start_timer()
    
    jet_rnn.shuffle_data()
    jet_rnn.reduce_dataset(dataset_sample)
    jet_rnn.make_ragged_tensor()
    jet_rnn.train_test_split(test_size=0.2)
    jet_rnn.create_model(args_model['model'])
    history = jet_rnn.train_model(verbose_level=0)
    
    # Calculate results
    model_results.training_history(history)
    model_results.confusion_matrix(jet_rnn, cutoff_threshold=0.5)
    model_results.roc_curve(jet_rnn)
    model_results.stop_timer(verbose=True)
    model_results_multi.add_result(model_results)
    
df_results = model_results_multi.return_results()

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
data_mean1, data_std1 = model_results_multi.average_training_history('history_training_data')
data_mean2, data_std2 = model_results_multi.average_training_history('history_test_data')
fig = plotlib.training_history_plot(data_mean1, data_mean2, params_history, 
                                    error_bars=[data_std1, data_std2])
print(f'average training accuracy: {data_mean1[-1]:0.4f} \u00B1 {data_std1[-1]:0.4f}')
print(f'average test accuracy:     {data_mean2[-1]:0.4f} \u00B1 {data_std2[-1]:0.4f}')

# Plot average confusion matrix
data_mean1, data_std1 = model_results_multi.average_confusion_matrix()
fig = plotlib.confusion_matrix(data_mean1, params_cm)

# Plot average ROC curve
fig = plotlib.plot_roc(model_results_multi.average_roc_curve(), params_roc)
