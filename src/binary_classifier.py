"""
Binary classifier base functions common to the eventNN, JetRNN and complete
neural network.
"""

# ---------------------------------- Imports ----------------------------------

# Python libraries
import pickle
import numpy as np
import pandas as pd
import time

# ----------------------------- Class definitions -----------------------------

class BinaryClassifier():
    """
    This class stores the base functions common to all binary classifier neural
    network models.
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
        self.args_model = args_model

    def load_data(self, data_dir, verbose=True):
        START = time.time()
        self.encoding_dict = pickle.load(open(data_dir+'encoding_dict.pkl', 'rb'))
        self.args = pickle.load(open(data_dir+'data_preprocessing_arguments.pkl', 'rb'))
        self.df_labels = pd.read_pickle(data_dir+'df_labels.pkl')
        self.df_weights = pd.read_pickle(data_dir+'df_weights.pkl')
        if verbose:
            print(f"    Elapsed data loading time: {time.time()-START:0.3f}s")
            
    def load_event_data(self, data_dir, verbose=True):
        START = time.time()
        self.df_event_data = pd.read_pickle(data_dir+'preprocessed_event_data.pkl')
        self.args_model['layer_input_shape'] = self.df_event_data.shape[1]
        if verbose:
            print(f"    Elapsed event data loading time: {time.time()-START:0.3f}s")
            
    def load_jet_data(self, data_dir, verbose=True):
        START = time.time()
        self.df_jet_data = pd.read_pickle(data_dir+'preprocessed_jet_data.pkl')
        num_jet_cols = self.df_jet_data.shape[1]
        self.args_model['layer_input_shape'] = [None, num_jet_cols]
        if verbose:
            print(f"    Elapsed jet data loading time: {time.time()-START:0.3f}s")
            
    def shuffle_data(self):
        """
        Function to randomly shuffle the dataset.
        """
        idx = np.random.permutation(self.df_labels.index)
        self.df_labels = self.df_labels.reindex(idx)
        self.df_weights = self.df_weights.reindex(idx)
        try:
            self.df_event_data = self.df_event_data.reindex(idx)
        except:
            pass
        try:
            self.df_jet_data = self.df_jet_data.reindex(idx)
        except:
            pass

    def train_test_split(self, test_size):
        """
        Define where to split the data to create seperate training and test
        datasets.

        Parameters
        ----------
        test_size : float
            Size of the test datast as a fraction of the whole dataset (0 to 1).
        """
        self.args_model['test_train_fraction'] = test_size
        training_size = 1-test_size
        test_train_split = int(len(self.df_labels)*training_size)
        self.test_train_split = test_train_split
        
