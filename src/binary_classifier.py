"""
Binary classifier base functions common to the eventNN, JetRNN and complete
neural network.
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
            
    def load_jet_data(self, data_dir, verbose=True):
        START = time.time()
        self.df_jet_data = pd.read_pickle(data_dir+'preprocessed_jet_data.pkl')
        num_jet_cols = self.df_jet_data.shape[1]
        self.args_model['jet_layer_input_shape'] = [None, num_jet_cols]
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
        
    def labels_test(self):
        """
        Returns event labels of the test dataset.
        -------
        TYPE
            Event labels for the test dataset.
        """
        return self.df_labels['label_encoding'].iloc[self.tt_split:self.dataset_end].values

# -----------------------------------------------------------------------------

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
        if model_name == 'base2':
            self.model = sequential.base2(self.args_model['layer_1_neurons'], 
                                          self.args_model['layer_2_neurons'], 
                                          input_shape=self.args_model['event_layer_input_shape'],
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
        data_train = self.df_event_data.iloc[:self.tt_split]
        data_test = self.df_event_data.iloc[self.tt_split:self.dataset_end]
        labels_train = self.df_labels['label_encoding'].iloc[:self.tt_split]
        labels_test = self.df_labels['label_encoding'].iloc[self.tt_split:self.dataset_end]
        sample_weight = self.df_weights['sample_weight'].iloc[:self.tt_split]
        
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
        data_test = self.df_event_data.iloc[self.tt_split:self.dataset_end]
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
        labels_test = self.df_labels['label_encoding'].iloc[self.tt_split:self.dataset_end]
        xs_weight = self.df_weights['xs_weight'].iloc[self.tt_split:self.dataset_end]
        
        dataset = pd.DataFrame(data=labels_pred, columns=['labels_pred'], index=labels_test.index)
        dataset['xs_weight'] = xs_weight*140000
        dataset['event_labels'] = labels_test
        return dataset

# -----------------------------------------------------------------------------

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
            self.model = recurrent.base(self.args_model['layer_1_neurons'], 
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
        data_train = self.jet_data_rt[:self.tt_split]
        data_test = self.jet_data_rt[self.tt_split:self.dataset_end]
        labels_train = self.df_labels['label_encoding'].iloc[:self.tt_split].values
        labels_test = self.df_labels['label_encoding'].iloc[self.tt_split:self.dataset_end].values
        sample_weight = self.df_weights['sample_weight'].iloc[:self.tt_split].values
        
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
        data_test = self.jet_data_rt[self.tt_split:self.dataset_end]
        labels_pred = self.model.predict(data_test)
        return labels_pred

# -----------------------------------------------------------------------------

class CombinedNN(BinaryClassifier):
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
        if model_name == 'base1':
            self.model = combined.base1(input_shape1=self.args_model['event_layer_input_shape'],
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
        data_train = self.df_event_data.iloc[:self.tt_split].values
        data_test = self.df_event_data.iloc[self.tt_split:self.dataset_end].values
        data_train_rt = self.jet_data_rt[:self.tt_split]
        data_test_rt = self.jet_data_rt[self.tt_split:self.dataset_end]
        
        labels_train = self.df_labels['label_encoding'].iloc[:self.tt_split].values
        labels_test = self.df_labels['label_encoding'].iloc[self.tt_split:self.dataset_end].values
        sample_weight = self.df_weights['sample_weight'].iloc[:self.tt_split].values
        
        history = self.model.fit(x=[data_train,data_train_rt], 
                                 y=labels_train,
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
        data_test = self.df_event_data.iloc[self.tt_split:self.dataset_end].values
        data_test_rt = self.jet_data_rt[self.tt_split:self.dataset_end]
        
        labels_pred = self.model.predict([data_test,data_test_rt])
        return labels_pred
    
# -------------------------------- Run function -------------------------------

def run(index, neural_net, args_model, dataset_sample, test_size=0.2):
    # Make results object
    model_result = ModelResults(index)
    model_result.start_timer()
    
    # Create and train model
    neural_net.shuffle_data()
    neural_net.reduce_dataset(dataset_sample)
    if isinstance(neural_net, (JetRNN, CombinedNN)):
        neural_net.make_ragged_tensor()
    neural_net.train_test_split(test_size)
    neural_net.create_model(args_model['model'])
    history = neural_net.train_model(verbose_level=0)
    
    # Calculate results
    model_result.training_history(history)
    model_result.confusion_matrix(neural_net, cutoff_threshold=0.5)
    model_result.roc_curve(neural_net)
    model_result.stop_timer(verbose=True)
    return model_result
