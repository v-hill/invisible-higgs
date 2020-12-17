# -*- coding: utf-8 -*-
"""
This file contains Classes for preparing the data for input into the neural 
networks.
"""

# Python libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing

class DataProcessing():
    def __init__(self, data):
        self.data_list = data.data
        self.data = None
        self.all_columns = []
        
        self.merge_data_list()
        
    def merge_data_list(self):
        self.data = pd.concat(self.data_list, axis=0, ignore_index=True)
        self.all_columns = list(self.data.columns.values)
        
    #--------------------------------------------------------------------------
    
    def label_signal_noise(self, signal_list):
        """
        This function converts the dataset labels to one of two numeric values
        representing either signal or noise (0 or 1). 

        Parameters
        ----------
        signal_list : list
            List of df['dataset'] values to consider signal.
            e.g. ['ttH125']
        """
        label_dict = {}
        signal_label = 0
        noise_label = 1
        
        for df in self.data_list:
            dataset = df.iloc[0]['dataset']
            if dataset in signal_list:
                label_dict[dataset] = signal_label
            else:
                label_dict[dataset] = noise_label
        labels = self.data['dataset']
        labels = labels.replace(label_dict)
        self.data['dataset'] = labels
        
    def return_dataset_labels(self):
        labels = self.data['dataset'].copy(deep=False)
        return labels.values.astype(str)
        
    #--------------------------------------------------------------------------
    
    def get_event_columns(self, columns_to_ignore, verbose=True):
        """
        This function generates a list of the columns to be used in the 
        training data for the event level variables. 
    
        Parameters
        ----------
        columns_to_ignore : list
            List of columns to explicitly exclude
    
        Returns
        -------
        columns_filtered : list
            list of columns to use for training
        """
        columns_filtered = []
        
        for idx, col in enumerate(self.all_columns):
            if isinstance(self.data[col].iloc[idx], (np.float32, np.float64, np.int64, np.uint32)):
                if col not in columns_to_ignore:
                    columns_filtered.append(col)
                    if verbose:
                        print(f"{col:<32}: {self.data[col].dtypes}")
        return columns_filtered

    def get_jet_columns(self, verbose=True):
        """
        This function generates a list of the columns to be used in the 
        training data for the jet variables.

        Returns
        -------
        columns_filtered : list
            list of columns to use for training
        """
        columns_filtered = []
        
        for col in self.all_columns:
            if isinstance(self.data[col].iloc[0], np.ndarray):
                columns_filtered.append(col)
                if verbose:
                    print(f"{col:<32}: {self.data[col].dtypes}")
        return columns_filtered


class LabelMaker():
    """
    This Class contains functions for creating numeric labels for the training
    categories. Encodes categorical values to intergers.
    """
    
    def label_encoding(label_data, verbose=True):
        encoder = preprocessing.LabelEncoder()
        labels = encoder.fit_transform(label_data)
        if verbose:
            keys = encoder.classes_
            values = encoder.transform(keys)
            print(dict(zip(keys, values)))
        return labels
    
    def onehot_encoding(data_list, verbose=True):
        data_list = data_list.reshape(-1, 1)
        
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        labels = onehot_encoder.fit_transform(data_list)
        return labels
    

class WeightMaker():
    def event_class_weights(data):
        weight_nominal_vals = data.data['weight_nominal']
        print(f"len: {len(weight_nominal_vals)}")
        return
        
