"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
from binary_classifier import BinaryClassifier
import models.sequential_models as sequential_models
import models.recurrent_models as recurrent_models
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResults, ModelResultsMulti

# Python libraries
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time

# ----------------------------- Class definitions -----------------------------

class CompleteNN(BinaryClassifier):
    """
    This class provides a wrapper for the data loading, model creating, model
    training and analysis of combined FFN and RNN keras models. 
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
        
    def create_model(self, model_name):
        pass
    def train_model(self, verbose_level):
        pass
