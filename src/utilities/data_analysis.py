"""
Functions for calculating results from trained neural networks.
"""

# ---------------------------------- Imports ----------------------------------

# Python libraries
from datetime import datetime
import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# ----------------------------- Class definitions -----------------------------

class ModelResults():
    """
    This class calculates results from the trained EventNN model.
    """
    def __init__(self, index):
        self.run_index = index
        self.time_start = 0
        self.time_elapsed = 0
        
    def start_timer(self):
        """
        Begin timing code execution.
        """
        self.time_start = time.time()
        
    def stop_timer(self, verbose=True):
        """
        Stop programme timer and optionally print the run time.

        Parameters
        ----------
        verbose : TYPE, bool
            Print run time. The default is True.
        """
        self.time_elapsed = time.time() - self.time_start
        if verbose:
            print(f'    Run {self.run_index} time: {self.time_elapsed:0.2f}s')
            if not self.test_passed:
                print('        Training failed, model rejected')
        
    def verify_training(self, neural_net, uniques_limit=250):
        """
        Check that the model correctly trained.
        1) look at the number of unique predictions made by the model on the 
           test dataset.

        Parameters
        ----------
        neural_net : EventNN, JetRNN, CombinedNN
            Keras model class.
        uniques_limit : int, optional
            Limit on the number of unique model predictions in order to say 
            that the model successfully trained. The default is 250.

        Returns
        -------
        bool
            Returns True is model successfully trained, else False.
        """
        self.test_passed = True
        
        test_predictions = neural_net.predict_test_data()
        num_unique = len(np.unique(test_predictions, axis=0))
        max_pred = test_predictions.max()
        min_pred = test_predictions.min()
        print(f'    len unique predictions: {num_unique}')
        print(f'    min/max prediction: {min_pred:0.4f}, {max_pred:0.4f}')
        
        if num_unique <= uniques_limit:
            self.test_passed = False
            
        if min_pred >= 0.15 or max_pred <= 0.85:
            self.test_passed = False
            
        return self.test_passed
        
    def training_history(self, history):
        """
        Add training histroy to ModelResults object.

        Parameters
        ----------
        history : tensorflow.python.keras.callbacks.History
            Model training history.
        """
        self.history_training_data = history.history['accuracy']
        self.history_test_data = history.history['val_accuracy']
        self.accuracy_training = history.history['accuracy'][-1]
        self.accuracy_test = history.history['val_accuracy'][-1]
        self.accuracy_training_start = history.history['accuracy'][0]
        self.accuracy_test_start = history.history['val_accuracy'][0]
        
    def confusion_matrix(self, neural_net, cutoff_threshold):
        """
        Calculate confusion matrix and confusion matrix derived results.
        """
        # Get model predictions and convert predictions into binary values
        test_predictions = neural_net.predict_test_data()
        labels_pred_binary = np.where(test_predictions > cutoff_threshold, 1, 0)
        
        # Make confsuion matrix
        self.confusion_matrix = confusion_matrix(neural_net.labels_test(), 
                                                 labels_pred_binary)

        TP = self.confusion_matrix[0, 0]
        TN = self.confusion_matrix[1, 1]
        FP = self.confusion_matrix[1, 0]
        FN = self.confusion_matrix[0, 1]
        
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f_score = 2*(recall * precision) / (recall + precision)
    
        self.cm_accuracy = accuracy
        self.cm_recall = recall
        self.cm_precision = precision
        self.cm_f_score = f_score
    
    def roc_curve(self, neural_net, sample_vals=250):
        """
        Produce the receiver operating characteristic curve (ROC curve) values.

        Parameters
        ----------
        neural_net : EventNN, JetRNN, CombinedNN
            Keras model class.
        sample_vals : int, optional
            The number of ROC curve values to keep. Saves a random sample
            of the full set of values. The default is 250.
        """
        test_predictions = neural_net.predict_test_data()
        fpr, tpr, _ = roc_curve(neural_net.labels_test(), test_predictions)
        roc_auc = auc(fpr, tpr)

        indices = sorted(np.random.choice(len(fpr), sample_vals, replace=False))

        self.roc_fpr_vals = fpr[indices]
        self.roc_tpr_vals = tpr[indices]
        self.roc_auc = roc_auc
        
    def calc_significance(self, dataset, num_thresholds=200, ZA=True):
        """
        Calculate the significance plot.

        Parameters
        ----------
        dataset : pandas.core.frame.DataFrame
            Dataframe of values needed to create significance plot.
        num_thresholds : int, optional
            Number of discriminator threshold values to calculate the 
            significance for. Number of points generated to plot.
            The default is 200.
        ZA : bool, optional
            Whether to use the full formula for the significance, or simple
            use s/sqrt(b) approximation. The default is True.

        Returns
        -------
        bin_centres_sig : list
            List of threshold values. x-axis vals.
        sig_vals : numpy.ndarray
            Values for the significance. 
        """
        bin_centres_sig = []
        bin_vals_sig = []
        bin_centres_back = []
        bin_vals_back = []
        sig_vals = []
        thresholds = np.linspace(0, 1, num_thresholds)
        
        for i in range(len(thresholds)-1):
            df_selection = dataset[dataset['labels_pred'].between(thresholds[i], 1)]
            df_sig = df_selection[df_selection['event_labels'] == 1]
            df_back = df_selection[df_selection['event_labels'] == 0]
            sum_xs_weight_sig = df_sig['xs_weight'].sum()
            sum_xs_weight_back = df_back['xs_weight'].sum()
            
            if sum_xs_weight_sig <= 0 or sum_xs_weight_back <= 0:
                continue
            
            bin_centres_sig.append(thresholds[i])
            bin_vals_sig.append(sum_xs_weight_sig)
            bin_centres_back.append(thresholds[i])
            bin_vals_back.append(sum_xs_weight_back)
        
            s = sum_xs_weight_sig
            b = sum_xs_weight_back
            
            if ZA == True:
                z = sqrt(2*((s+b)*np.log(1+(s/b))-s)) # Calculate significance 
                sig_vals.append(z)
            else:
                sig_vals.append(s/(sqrt(b)))
                
        sig_vals = np.asarray(sig_vals)
        self.max_significance = bin_centres_sig[sig_vals.argmax()]
        return bin_centres_sig, sig_vals
        
    def to_dict(self, floats_only=False):
        """
        Return a dictionary of the class attributes.
        
        Parameters
        ----------
        floats_only : bool, optional
            If floats_only is True, then only the attributes which are single
            values will be returned. Else all attribute are returned.
            The default is False.

        Returns
        -------
        self_dict : dict
            Dictionary representation of class attributes.
        """
        self_dict = self.__dict__
        if floats_only:
            types = [float, int, np.float64, np.int32]
            self_dict = {k: v for k, v in self_dict.items() if type(v) in types}
        return self_dict
    
    def to_dataframe(self):
        """
        Return a pandas dataframe of the class attributes.

        Returns
        -------
        results_df : pandas.DataFrame
            Dataframe representation of class attributes.
        """
        output_dict = self.to_dict()
        results_df = pd.DataFrame([output_dict])
        results_df = results_df.reindex(sorted(results_df.columns), axis=1)
        
        index_col = results_df.pop('run_index')
        results_df.insert(0, 'run_index', index_col)
        return results_df

class ModelResultsMulti():
    """
    This class stores the results from multiple runs of training the EventNN 
    model.
    """
    def __init__(self):
        self.df_results = pd.DataFrame()
        
    def add_result(self, result, additional_results={}):
        """
        Add a training result to the main dataframe.

        Parameters
        ----------
        result : ModelResults
            ModelResults class object of training results.
         additional_results : dict
            Additional values to add to model_results dataframe in the form 
            of a dictionary. Used to add model arguments to results.
        """
        df = result.to_dataframe()
        for key, value in additional_results.items():
            try:
                df.insert(1, key, value)
            except:
                pass
            
        self.df_results = pd.concat([self.df_results, df], axis=0, ignore_index=True)

    def return_results(self):
        return self.df_results
    
    def save(self, filename, timestamp=True):
        if timestamp:
            now = datetime.today().strftime('%Y_%m_%d__%H_%M_%S__')
            filename = now + filename
        self.df_results.to_pickle(filename)
        
    
    def average_training_history(self, history_val):
        """
        Computes the mean and standard deviation for the training history curve.

        Parameters
        ----------
        history_val : str
            Either 'history_training_data' or 'history_test_data'

        Returns
        -------
        data_mean : numpy.ndarray
            Mean values for each epoch.
        data_std : numpy.ndarray
            Standard deviation values for each epoch.
        """

        all_data = self.df_results[history_val].dropna().values
        epochs = len(all_data[0])
        all_data = np.concatenate(all_data)
        all_data = all_data.reshape([-1, epochs])
        data_mean = all_data.mean(axis=0)
        data_std = all_data.std(axis=0)
        return data_mean, data_std
        
    def average_confusion_matrix(self):
        """
        Computes the mean and standard deviation for the confusion matrix.

        Returns
        -------
        data_mean : numpy.ndarray
            Mean values of the confusion matrix.
        data_std : numpy.ndarray
            Standard deviation values of the confusion matrix.
        """
        all_data = self.df_results['confusion_matrix'].dropna().values
        all_data = np.concatenate(all_data, axis=0)
        all_data = all_data.reshape([-1, 2, 2])
        data_mean = all_data.mean(axis=0)
        data_std = all_data.std(axis=0)
        return data_mean, data_std
        
    def average_roc_curve(self):
        """
        Computes the mean values for the ROC curve plot.

        Returns
        -------
        results_df : pandas.DataFrame
            Dataframe containing the ROC curve results to plot.
        """
        results = {}
        results['roc_auc'] = self.df_results['roc_auc'].mean()
        

        all_data_fpr = self.df_results['roc_fpr_vals'].dropna().values
        epochs = all_data_fpr.shape[0]
        all_data_fpr = np.concatenate(all_data_fpr, axis=0)
        all_data_fpr = all_data_fpr.reshape([epochs, -1])
        data_mean1 = all_data_fpr.mean(axis=0)
        results['roc_fpr_vals'] = data_mean1

        all_data_tpr = self.df_results['roc_tpr_vals'].dropna().values
        all_data_tpr = np.concatenate(all_data_tpr, axis=0)
        all_data_tpr = all_data_tpr.reshape([epochs, -1])
        data_mean2 = all_data_tpr.mean(axis=0)
        results['roc_tpr_vals'] = data_mean2
        
        results_df = pd.DataFrame([results])
        
        return results_df.iloc[0]
        
