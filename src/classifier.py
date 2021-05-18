"""
Classifier base Class, eventNN, JetRNN and combined neural network.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import models.sequential_models as sequential
import models.recurrent_models as recurrent
import models.combined_models as combined
from utilities.data_preprocessing import make_ragged_tensor
from utilities.data_analysis import ModelResults

# Python libraries
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import time

# ----------------------------- Class definitions -----------------------------

class Classifier():
    """
    This class stores the base functions common to all classifier neural 
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
        self.dataset_start = 0
        self.dataset_end = -1

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
        self.args_model['event_layer_input_shape'] = self.df_event_data.shape[1]
        if verbose:
            print(f"    Elapsed event data loading time: {time.time()-START:0.3f}s")
            
    def remove_variable(self, index):
        print(f'column removed: {self.df_event_data.columns[index]}')
        self.df_event_data = self.df_event_data.drop(self.df_event_data.columns[index], axis=1)
        self.args_model['event_layer_input_shape'] = self.df_event_data.shape[1]
        
            
    def load_jet_data(self, data_dir, verbose=True):
        START = time.time()
        self.df_jet_data = pd.read_pickle(data_dir+'preprocessed_jet_data.pkl')
        num_jet_cols = self.df_jet_data.shape[1]
        self.args_model['jet_layer_input_shape'] = [None, num_jet_cols]
        if verbose:
            print(f"    Elapsed jet data loading time: {time.time()-START:0.3f}s")
            
    def load_full_dataset(self, data_dir, verbose=True):
        START = time.time()
        self.df_all_data = pd.read_pickle(data_dir+'df_all_data.pkl')
        if verbose:
            print(f"    Elapsed all data loading time: {time.time()-START:0.3f}s")
            
    def shuffle_data(self):
        """
        Function to randomly shuffle the dataset.
        """
        idx = np.random.permutation(self.df_labels.index)
        self.df_labels.index = idx
        self.df_labels = self.df_labels.sort_index()
        
        self.df_weights.index = idx
        self.df_weights = self.df_weights.sort_index()
        try:
            self.df_event_data.index = idx
            self.df_event_data = self.df_event_data.sort_index()
        except:
            pass
        try:
            self.df_jet_data.index = idx
            self.df_jet_data = self.df_jet_data.sort_index()
        except:
            pass
        try:
            self.df_all_data.index = idx
            self.df_all_data = self.df_all_data.sort_index()
        except:
            pass

    def reduce_dataset(self, dataset_fraction):
        """
        Reduce the size of the dataset. Samples dataset values up to the       
        'dataset_fraction' specified. 

        Parameters
        ----------
        dataset_fraction : float
            New dataset size as a fraction of the original. Values 0 to 1.
        """
        self.dataset_end = int(len(self.df_labels)*dataset_fraction)

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
        test_train_split = int(len(self.df_labels[:self.dataset_end])*training_size)
        self.tt_split = test_train_split
        
        self.train_idx = list(range(0, test_train_split))
        self.test_idx = list(range(test_train_split, self.dataset_end))
        
    def set_test_train_indices(self, idx_list, fold_num):
        """
        Manually set the indices for the testing and training data.

        Parameters
        ----------
        idx_list : list
            List of list of new indices.
        fold_num : int
            fold num
        """
        self.test_idx = idx_list[fold_num]
        idx_list_new = [item for i, item in enumerate(idx_list) if i!=fold_num]
        idx = [item for sublist in idx_list_new for item in sublist]
        self.train_idx = idx
        
    def set_label_encoding(self, signal):
        """
        Update label encoding

        Parameters
        ----------
        signal : str
            Multisignal label to use as binary label.
        """
        self.df_labels['label_encoding'] = np.asarray(self.df_labels[signal]).astype('int32')
        
    def labels_test(self):
        """
        Returns event labels of the test dataset.
        -------
        TYPE
            Event labels for the test dataset.
        """
        try:
            labels = self.df_labels['label_encoding'].iloc[self.test_idx].values
        except:
            cols = [col for col in self.df_labels.columns if col not in ['raw_dataset', 'dataset']]
            labels = self.df_labels[cols].iloc[self.test_idx].values
        return labels
    
    def multi_labels_test(self):
        """
        Returns event labels of the test dataset.
        -------
        TYPE
            One hot encoded event labels for the test dataset.
        """
        cols = [col for col in self.df_labels.columns if col not in ['raw_dataset', 'dataset']]
        return self.df_labels[cols].iloc[self.tt_split:self.dataset_end].values
    
    def discriminator_values(self):
        labels_test = self.labels_test()
        labels_pred = self.predict_test_data()
        labels_pred_signal = labels_pred[np.array(labels_test, dtype=bool)]
        labels_pred_background = labels_pred[np.invert(np.array(labels_test, dtype=bool))]
        return labels_pred_signal, labels_pred_background

# -----------------------------------------------------------------------------

class EventNN(Classifier):
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
        self.model = None
        
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
        tf.keras.backend.clear_session()
        del self.model
        if model_name == 'base':
            self.model = sequential.base(self.args_model)

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
        # test = self.df_event_data.iloc[self.train_idx].iloc[:100]
        # test2 = self.df_all_data.iloc[self.train_idx].iloc[:100]
        # test3 = self.train_idx[:100]
        
        data_train = self.df_event_data.iloc[self.train_idx]
        data_test = self.df_event_data.iloc[self.test_idx]
        labels_train = self.df_labels['label_encoding'].iloc[self.train_idx].astype('int32')
        labels_test = self.df_labels['label_encoding'].iloc[self.test_idx].astype('int32')
        
        print(f'training, signal: {labels_train.sum():7}, background: {(len(labels_train)-labels_train.sum()):7}, '
              f'ratio: {(labels_train.sum())/len(labels_train):0.4f}')
        print(f'testing,  signal: {labels_test.sum():7}, background: {len(labels_test)-labels_test.sum():7}, ' 
               f'ratio: {(labels_test.sum())/len(labels_test):0.4f}')
        
        sample_weight = self.df_weights['sample_weight'].iloc[self.train_idx]
        
        history = self.model.fit(data_train, labels_train,
                                 batch_size=self.args_model['batch_size'],
                                 validation_data=(data_test, labels_test),
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
        data_test = self.df_event_data.iloc[self.test_idx]
        labels_pred = self.model.predict(data_test)
        return labels_pred
    
    def predict_all_data(self):
        """
        Return predictions of trainined model for the whole dataset.

        Returns
        -------
        labels_pred : numpy.ndarray
            Predicted labels for the test dataset.
        """
        labels_pred = self.model.predict(self.df_event_data)
        return labels_pred
    
    def significance_dataset(self):
        # Make dataset
        labels_pred = self.predict_test_data()
        labels_test = self.df_labels['label_encoding'].iloc[self.test_idx]
        xs_weight = self.df_weights['xs_weight'].iloc[self.test_idx]
        
        dataset = pd.DataFrame(data=labels_pred, columns=['labels_pred'], index=labels_test.index)
        dataset['xs_weight'] = xs_weight*140000
        dataset['event_labels'] = labels_test
        return dataset

# -----------------------------------------------------------------------------

class JetRNN(Classifier):
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
        
    def reverse_jet_order(self):
        """
        The jet data is ordered from highest to lowest transverse momentum pt,
        this function reverses the order of the jets to see if it has any effect
        on the training.

        """
        for col in self.df_jet_data.columns:
            self.df_jet_data[col] = self.df_jet_data[col].apply(lambda x:x[::-1])
            
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
        Define which model from the models.recurrent_models module to use.

        Parameters
        ----------
        model_name : str
            Name of function in models.recurrent_models to use to create the
            model.
        """
        tf.keras.backend.clear_session()
        del self.model
        if model_name == 'base':
            self.model = recurrent.base(self.args_model)

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

        data_train = make_ragged_tensor(self.df_jet_data.iloc[self.train_idx])
        data_test = make_ragged_tensor(self.df_jet_data.iloc[self.test_idx])
        self.data_test = data_test
        
        labels_train = self.df_labels['label_encoding'].iloc[self.train_idx].values
        labels_test = self.df_labels['label_encoding'].iloc[self.test_idx].values
        sample_weight = self.df_weights['sample_weight'].iloc[self.train_idx].values
        
        history = self.model.fit(data_train, labels_train,
                                 batch_size=self.args_model['batch_size'],
                                 validation_data=(data_test, labels_test),
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
        labels_pred = self.model.predict(self.data_test)
        return labels_pred

# -----------------------------------------------------------------------------

class CombinedNN(Classifier):
    """
    This class provides a wrapper for the data loading, model creating, model
    training and analysis of combined event feedforward network and the 
    recurrent neural network. 
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
        
    make_ragged_tensor = JetRNN.make_ragged_tensor

    def create_model(self, model_name):
        """
        Creat and compile the combined FFN + RNN network.

        Parameters
        ----------
        model_name : str
            Name of function in models.combined_models to use to create the
            model.
        """
        tf.keras.backend.clear_session()
        del self.model
        if model_name == 'base':
            self.model = combined.base(self.args_model)
            
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
        cols = [col for col in self.df_labels.columns if col not in ['raw_dataset', 'dataset']]
        # cols = 'label_encoding'
        data_train = self.df_event_data.iloc[self.train_idx].values
        data_test = self.df_event_data.iloc[self.test_idx].values
        
        data_train_rt = make_ragged_tensor(self.df_jet_data.iloc[self.train_idx])
        data_test_rt = make_ragged_tensor(self.df_jet_data.iloc[self.test_idx])
        self.data_test_rt = data_test_rt
        
        labels_train = self.df_labels[cols].iloc[self.train_idx].values
        labels_test = self.df_labels[cols].iloc[self.test_idx].values
        
        sample_weight = self.df_weights['sample_weight'].iloc[self.train_idx].values
        
        history = self.model.fit(x=[data_train, data_train_rt], 
                                 y=labels_train,
                                 batch_size=self.args_model['batch_size'],
                                 validation_data=([data_test, data_test_rt], labels_test),
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
        data_test = self.df_event_data.iloc[self.test_idx].values

        labels_pred = self.model.predict([data_test, self.data_test_rt])
        return labels_pred
    
    
# -------------------------------- Run functions ------------------------------

def run(index, neural_net, args_model, dataset_sample, test_size=0.2):
    """
    Train and evaluate a neural network once.

    Parameters
    ----------
    index : int
        Run index.
    neural_net : EventNN, JetRNN, CombinedNN
        Neural network to be trained.
    args_model : dict
        Dictionary of arguments hich define the model.
    dataset_sample : float
        Fraction of dataset to use in training. Value 0 to 1.
    test_size : float, optional
        Fraction of dataset to use as the test set. Value 0 to 1. 
        The default is 0.2.

    Returns
    -------
    model_result : ModelResults Class
        Results of trained model
    """
    # Make results object
    model_result = ModelResults(index)
    model_result.start_timer()
    
    # Create and train model
    neural_net.shuffle_data()
    neural_net.reduce_dataset(dataset_sample)
    neural_net.train_test_split(test_size)
    neural_net.create_model(args_model['model'])
    history = neural_net.train_model(verbose_level=0)
    success = model_result.verify_training(neural_net)
    
    # Calculate results
    if success:
        model_result.training_history(history)
        model_result.discriminator_hist(neural_net, 50)
        model_result.confusion_matrix(neural_net, cutoff_threshold=0.5)
        model_result.roc_curve(neural_net)
    model_result.stop_timer(verbose=True)
    return model_result

def run_multi(index, neural_net, args_model, dataset_sample, test_size=0.2):
    """
    Same as run() function but for the multisignal/multiclassifier networks.
    
    Returns
    -------
    model_result : ModelResults Class
        Results of trained model

    """
    # Make results object
    model_result = ModelResults(index)
    model_result.start_timer()
    
    # Create and train model
    neural_net.shuffle_data()
    neural_net.reduce_dataset(dataset_sample)
    neural_net.train_test_split(test_size)
    neural_net.create_model(args_model['model'])
    history = neural_net.train_model(verbose_level=0)
    success = model_result.verify_training(neural_net)
    
    # Calculate results
    if success:
        model_result.training_history(history)
        model_result.multi_class_roc_curve(neural_net)
        model_result.multi_class_confusion_matrix(neural_net)
    model_result.stop_timer(verbose=True)
    return model_result
