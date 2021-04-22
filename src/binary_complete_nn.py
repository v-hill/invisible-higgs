"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# ---------------------------------- Imports ----------------------------------

# Tell tensorflow to ignore GPU device and run on CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Code from other files in the repo
from utilities.data_preprocessing import make_ragged_tensor
import models.combined_models as combined_models
import utilities.plotlib as plotlib
from binary_classifier import BinaryClassifier
from utilities.data_analysis import ModelResults, ModelResultsMulti

# Python libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# ----------------------------- Class definitions -----------------------------

class CombinedNeuralNetwork(BinaryClassifier):
    """
    This class provides a wrapper for the data loading, model creating, model
    training and analysis of the complete neural network keras models. 
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
        if model_name == 'base1':
            self.model = combined_models.base1(input_shape1=self.args_model['event_layer_input_shape'],
                                               input_shape2=self.args_model['jet_layer_input_shape'])
            
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
        data_train = self.df_event_data.iloc[:self.test_train_split].values
        data_test = self.df_event_data.iloc[self.test_train_split:self.dataset_end].values
        
        data_train_rt = self.jet_data_rt[:self.test_train_split]
        data_test_rt = self.jet_data_rt[self.test_train_split:self.dataset_end]
        
        labels_train = self.df_labels['label_encoding'].iloc[:self.test_train_split].values
        labels_test = self.df_labels['label_encoding'].iloc[self.test_train_split:self.dataset_end].values
        sample_weight = self.df_weights['sample_weight'].iloc[:self.test_train_split].values
        
        history = self.model.fit(x=[data_train,data_train_rt], y=labels_train,
                                 batch_size=self.args_model['batch_size'],
                                 validation_data=([data_test,data_test_rt], labels_test),
                                 sample_weight=sample_weight,
                                 epochs=self.args_model['epochs'],
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
        data_test = self.df_event_data.iloc[self.test_train_split:self.dataset_end].values
        data_test_rt = self.jet_data_rt[self.test_train_split:self.dataset_end]
        labels_pred = self.model.predict([data_test,data_test_rt])
        return labels_pred
    
SAVE_FOLDER = 'data_binary_classifier'
DIR = SAVE_FOLDER + '\\'

args_model = {'model_type' : 'binary_classifier',
              'model_architecture' : 'FCN',
              'batch_size' : 64,
              'epochs' : 8,
              'model' : 'base1'}

num_runs = 1
dataset_sample = 0.05
model_results_multi = ModelResultsMulti()
FCN = CombinedNeuralNetwork(args_model)  
FCN.load_data(DIR)
FCN.load_event_data(DIR)
FCN.load_jet_data(DIR)

for i in range(num_runs):
     model_results = ModelResults(i)
     model_results.start_timer()
     
     FCN.shuffle_data()
     FCN.reduce_dataset(dataset_sample)
     FCN.make_ragged_tensor()
     FCN.train_test_split(test_size=0.2)
     FCN.create_model(args_model['model'])
     history = FCN.train_model(verbose_level=0)
     
     # Calculate results
     model_results.training_history(history)
     model_results.confusion_matrix(FCN, cutoff_threshold=0.5)
     model_results.roc_curve(FCN)
     model_results.stop_timer(verbose=True)
     model_results_multi.add_result(model_results)
    
df_results = model_results_multi.return_results()


