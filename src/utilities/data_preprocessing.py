"""
This file contains Classes for preparing the data for input into the neural 
networks.
"""

# Python libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import time

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
        signal_label = 'signal'
        noise_label = 'background'
        
        for df in self.data_list:
            dataset = df.iloc[0]['dataset']
            if dataset in signal_list:
                label_dict[dataset] = signal_label
            else:
                label_dict[dataset] = noise_label
        labels = self.data['dataset']
        labels = labels.replace(label_dict)
        self.data['dataset'] = labels
        
    def label_signal_noise_multi(self, data_dict):
        labels = self.data['dataset']
        labels = labels.replace(data_dict)
        self.data['dataset'] = labels
        
    def return_dataset_labels(self):
        labels = self.data['dataset'].copy(deep=False)
        return labels.values.astype(str)
    
    def set_dataset_labels(self, event_labels, onehot):
        if onehot:
            for i in reversed(range(event_labels.shape[1])):
                values = event_labels.iloc[:,i].values
                col_name = event_labels.columns[i]
                self.data.insert(loc=1, column=col_name, value=values)
        else:
            values = event_labels.iloc[:,0].values
            col_name = event_labels.columns[0]
            self.data.insert(loc=1, column=col_name, value=values)
            
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
        print("columns to be used for event level neural network:")
        columns_filtered = []
        
        for idx, col in enumerate(self.all_columns):
            if isinstance(self.data[col].iloc[idx], (np.float32, np.float64, np.int64, np.uint32)):
                if col not in columns_to_ignore:
                    columns_filtered.append(col)
                    if verbose:
                        print(f"    {col:<32}: {self.data[col].dtypes}")
        return columns_filtered

    def get_jet_columns(self, columns_to_ignore, verbose=True):
        """
        This function generates a list of the columns to be used in the 
        training data for the jet variables.
        
        Parameters
        ----------
        columns_to_ignore : list
            List of columns to explicitly exclude

        Returns
        -------
        columns_filtered : list
            list of columns to use for training
        """
        print("columns to be used for jet data neural network:")
        columns_filtered = []
        
        for col in self.all_columns:
            if isinstance(self.data[col].iloc[0], np.ndarray):
                if col not in columns_to_ignore:
                    columns_filtered.append(col)
                    if verbose:
                        print(f"    {col:<32}: {self.data[col].dtypes}")
        return columns_filtered
    
    def filter_data(self, column_filter, ignore=[]):
        """
        Remove all columns from the self.data dataframe except those in the
        list column_filter and the optional ignore list.

        Parameters
        ----------
        column_filter : list
            list of columns in the new self.data

        ignore : list, optional
            List of optional extra columns to keep. The default is [].

        Returns
        -------
        None.
        """
        self.data = self.data[column_filter]
        
    def set_nan_to_zero(self, column):
        self.data = self.data.fillna({column: 0})
    
    def remove_nan(self, column):
        start_len = len(self.data)
        self.data = self.data.dropna(subset=[column])
        nans_removed = start_len-len(self.data)
        print(f"{nans_removed} nan values removed")
        print(f"    {nans_removed/start_len:.3%} of events removed")
    
    def nat_log_columns(self, columns):
        """
        This function takes the natural log of the values for each column 
        listed in the 'columns' input variable. This is used to reduce the 
        skewness in highly skewed data.

        Parameters
        ----------
        columns : list
            list of column names
        """
        for col in columns:
            if col not in self.data.columns:
                print(f"{col} column not present in dataset")
                continue
                
            try:
                self.data[col] = np.log(self.data[col])
            except:
                raise Exception(f"{col} column cannot be logged")
            
    def normalise_columns(self, span=(0, 1), columns=None, return_df = False):
        """
        Use the sklearn MinMaxScaler to scale each columns values to a given 
        range. By deafult all columns are scaled, but a list of specific 
        columns can optionally be passed in.
        
        Parameters
        ----------
        span : tuple, optional
            A tuple where the first entry is the min and the second
            value is the max. The default is (0, 1).
        columns : list, optional
            List of columns to scale. The default is None, meaning all columns.
        return_df : bool, optional
            If False the self.data will be returned as a ndarray if True 
            self.data will be returned as a pandas dataframe
        """
        mm_scaler = preprocessing.MinMaxScaler(feature_range=span)
        if columns==None:
            columns = self.data.columns
            self.data = mm_scaler.fit_transform(self.data)
        else:
            self.data[columns] = mm_scaler.fit_transform(self.data[columns])
        
        if return_df == True:
            self.data = pd.DataFrame(self.data)
            self.data.columns = columns
        else:
            pass
        
class LabelMaker():
    """
    This Class contains functions for creating numeric labels for the training
    categories. Encodes categorical values to intergers.
    """
    def label_encoding(label_data, verbose=True):
        """
        Encodes the dataset labels into binary values 0 for signal and 1 for 
        background. A column of the same length as the label_data is returned 
        with the encoded labels returned.

        Parameters
        ----------
        label_data : Pandas.series
            Column from dataframe containing labal names, normally this will be
            the 'dataset' column.

        Returns
        -------
        df_labels : pandas.DataFrame
            Array with encoed labels corrosponding to event type (signal/noise)
        """
        encoder = preprocessing.LabelEncoder()
        labels = encoder.fit_transform(label_data)
        
        if verbose:
            keys = encoder.classes_
            values = encoder.transform(keys)
            encoding_dict = dict(zip(keys.tolist(), values.tolist()))
            print(f"label encoding: {encoding_dict}")
            
        df_labels = pd.DataFrame(data=labels, columns=['label encoding'])
        return df_labels, encoding_dict
    
    def onehot_encoding(data_list, verbose=True):
        """
        Produce a onehot encoding of the event labels.

        Parameters
        ----------
        data_list : numpy.ndarray
            data.return_dataset_labels() 
        verbose : Bool, optional
            Prints the encoding to console. The default is True.

        Returns
        -------
        df_labels : dict
            Dictionary of encoding.
        encoding_dict : pandas.DataFrame
            Dataframe of onehot encoded labels.
        """
        data_list = data_list.reshape(-1, 1)
        
        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        labels = onehot_encoder.fit_transform(data_list)
        
        keys = onehot_encoder.categories_[0]
        values = onehot_encoder.fit_transform(keys.reshape(-1, 1))
        encoding_dict = dict(zip(keys.tolist(), values.tolist()))
        
        # Make column names
        col_names = ['']*np.shape(labels)[1]
        for i, key in enumerate(keys):
            index = np.argwhere(values[i]==1)[0][0]
            col_name = 'onehot_' + str(key)
            col_names[index] = col_name
        
        if verbose:
            print('onehot encoding:')
            for item in encoding_dict.items():
                print(f'    label: {item[0]:20} encoding: {item[1]}')
                
        df_labels = pd.DataFrame(data=labels, columns=col_names)
        
        return df_labels, encoding_dict

class WeightMaker():
    """
    This Class contains functions for creating the class weights and sample
    weights to be added to the neural network.
    """
    def event_class_weights(data):
        """
        This function creates a dictionary of class weights. The weights are
        calculated from the number of events of each classification label.

        Parameters
        ----------
        data : DataProcessing

        Returns
        -------
        class_weight_dict : dict
        """
        weight_nominal_vals = data.data['weight_nominal']
        total_events = len(weight_nominal_vals)
        print(f"total number of events: {total_events}")
        
        unique_labels = data.data['dataset'].unique()
        class_weights = []
        
        for label in unique_labels:
            weight_selection = data.data.loc[data.data['dataset'] == label]['weight_nominal']
            weight = total_events/len(weight_selection)
            print(f"    {total_events}/{len(weight_selection)} = {weight:0.3f}")
            class_weights.append(weight)
            
        class_weight_dict = dict(zip(unique_labels, class_weights))
        print(f"class weights: {class_weight_dict}")
        return class_weight_dict
        
    def weight_nominal_sample_weights(data, weight_col='weight_nominal'):
        """
        This function uses the 'weight_nominal' or 'xs_weight' parameter to 
        create an event by event sample_weight. The sample weights are 
        normalised based on the sum of the 'weight_nominal' for each 
        classification label.

        Parameters
        ----------
        data : DataProcessing
        weight_col : str, optional
            Weight to make sample weights from. Either 'weight_nominal' or 
            'xs_weight'. The default is 'weight_nominal'.

        Returns
        -------
        new_weights : np.ndarray
            Array containing the sample weights to be fed into the model.fit()
        """
        weight_vals = data.data[weight_col]
        total_weight_val = weight_vals.sum()
        print(f"total {weight_col}: {total_weight_val:0.5f}")
        
        unique_labels = data.data['dataset'].unique()
        weight_list = []
        
        for label in unique_labels:
            weight_selection = data.data.loc[data.data['dataset'] == label][weight_col]
            # normalisation = total_weight_val/weight_selection.sum()
            normalisation = 1/weight_selection.sum()
            print(f"    {label} total {weight_col}: {weight_selection.sum():0.5f}")
            weight_selection *= normalisation
            weight_list.append(weight_selection)
            
        new_weights = pd.concat(weight_list, axis=0, ignore_index=True)
        return new_weights.values

# ---------------------------- Making RaggedTensor ----------------------------

# See: 'Uniform inner dimensions' in tensorflow documentation:
#       https://www.tensorflow.org/guide/ragged_tensor

def make_ragged_tensor(input_data):
    """
    Turns a pandas dataframe of the jet data into a tensorflow ragged tensor.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Dataframe containing the jet data.

    Returns
    -------
    rt : tensorflow.RaggedTensor
        Ragged tensor with 3 dimensions:
        TensorShape([{Number of events}, {Number of jets in the event}, 6])
        Each jet is specified by 6 values, with a variable number of jets per
        event. This second dimension denoting the number of jets is a ragged
        dimension.
    """
    data_list = input_data.values.tolist()
    
    row_current = 0
    row_splits = [0]
    rt_list_new = []
    for idx, event in enumerate(data_list):
        rt = np.stack(event, axis=0).T.tolist()
        row_current += len(rt)
        rt_list_new += rt
        row_splits.append(row_current)
    
    rt = tf.RaggedTensor.from_row_splits(
        values=rt_list_new,
        row_splits=row_splits)
    return rt

def split_data(event_data, labels, weights, test_size, shuffle=True):
    """
    This function splits a numpy array containg event data into test 
    and train sets matched with the right labels.
    
    Parameters
    ----------
    event_data : np.darray
        Event data in a (m,n) array where m is the different simulated event 
        data and n is the features from an event.
    labels : np.darray
        The labels that correspond to the different events
    weights : np.darray
        The weights for each event to control for cross section of different 
        events
    test_size : float
        The fraction of the data that is test data
    shuffle : bool, optional
        True if you want the data shuffled
        
    Results
    -------
    training_data : list 
        Data in a [(m,n),l,w] list where l is the label for an event and w is 
        weight. If shuffle is true then the events have been shuffled. This
        data is to be trained on.
    test_data : list 
        Data in a [(m,n),l,w] this data is for a network to be tested on.
    """
    if shuffle is True:    
        #Shuffle the training data by the same amount
        rng_state = np.random.get_state()
        
        np.random.shuffle(event_data)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        np.random.set_state(rng_state)
        np.random.shuffle(weights)
        
    else:
        #test data does not need to be shuffled
        pass
    
    train_length = int(len(event_data)*(1-test_size))
    
    training_data = event_data[:train_length,]
    training_label = labels[:train_length]
    training_weight = weights[:train_length]

    test_data = event_data[train_length:,]
    test_label = labels[train_length:]
    test_weight = weights[train_length:]
    
    training_data = [training_data,
                     training_label,
                     training_weight]
    
    test_data = [test_data,
                 test_label,
                 test_weight]
    
    return training_data, test_data

def normalise_jet_columns(data_train, span=(0,1), columns=None):
    '''
    This function normalises the data in the jet columns

    Parameters
    ----------
    data_train : pandas.DataFrame
        DataFrame containg the jet data with each element being a numpy array.
    span : tuple, optional
        A tuple where the first entry is the min and the second
        value is the max. The default is (0, 1).
    columns : list, optional
        List of columns which contain the jet data. The default is None.

    Returns
    -------
    data_train : pandas.DataFrame
        Dataframe containg the normalised jet data.

    '''
    print('normalising the jet data')
    start = time.time()
    df = data_train.copy(deep=True)
    if columns == None:
        columns = data_train.columns.tolist()
    else:
        pass
    
    for i, col in enumerate(columns):
        mm_scaler = preprocessing.MinMaxScaler(feature_range=span)
        data_to_fit = np.concatenate(df[col].values).reshape(-1,1)
        mm_scaler.fit(data_to_fit)
        df[col] = df[col].apply(lambda x:x.reshape(-1,1))
        df[col] = df[col].apply(mm_scaler.transform)
        df[col] = df[col].apply(lambda x:x.reshape(-1))
        
        print(f"    col {i}/6:    {time.time()-start:0.2f}")
    print(f"    Elapsed time: {time.time()-start}")
    return df

#------------------------ Test and build new functions-------------------------
'''Anything written in here will not be run when a module is called'''

if __name__ == "__main__":
    pass
