"""
Functions for calculating results from trained neural networks.
"""

# ---------------------------------- Imports ----------------------------------

# Python libraries
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
        self.index = index
        
    def training_history(self, history):
        """
        Add training histroy to ModelResults object.

        Parameters
        ----------
        history : TYPE
            DESCRIPTION.
        """
        self.history_training_data = history.history['accuracy']
        self.history_test_data = history.history['val_accuracy']
        self.accuracy_training = history.history['accuracy'][-1]
        self.accuracy_test = history.history['val_accuracy'][-1]
        
    def confusion_matrix(self, neural_net, cutoff_threshold):
        """
        Calculate confusion matrix and confusion matrix derived results.
        """
        # Get model predictions and convert predictions into binary values
        labels_pred = neural_net.predict_test_data()
        labels_pred_binary = np.where(labels_pred > cutoff_threshold, 1, 0)
        
        # Make confsuion matrix
        self.confusion_matrix = confusion_matrix(neural_net.labels_test(), 
                                                 labels_pred_binary)

        TP = self.confusion_matrix[0,0]
        TN = self.confusion_matrix[1,1]
        FP = self.confusion_matrix[1,0]
        FN = self.confusion_matrix[0,1]
        
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        recall = TP/(TP+FN)
        precision = TP/(TP+FP)
        f_score = 2*(recall * precision) / (recall + precision)
    
        self.cm_accuracy = accuracy
        self.cm_recall = recall
        self.cm_precision = precision
        self.cm_f_score = f_score
    
    def roc_curve(self, neural_net):
        fpr, tpr, _ = roc_curve(neural_net.labels_test(), neural_net.predict_test_data())
        roc_auc = auc(fpr, tpr)
        step = int(len(fpr)/1000)   # Select only 1000 values to plot
        
        self.roc_fpr_vals = fpr[::step]
        self.roc_tpr_vals = tpr[::step]
        self.roc_auc = roc_auc
        
    def calc_significance(self, dataset, num_thresholds=200, ZA=True):
        bin_centres_sig = []
        bin_vals_sig = []
        bin_centres_back = []
        bin_vals_back = []
        sig_vals = []
        thresholds = np.linspace(0, 1, num_thresholds)
        
        for i in range(len(thresholds)-1):
            df_selection = dataset[dataset['labels_pred'].between(thresholds[i], 1)]
            df_sig = df_selection[df_selection['event_labels']==1]
            df_back = df_selection[df_selection['event_labels']==0]
            sum_xs_weight_sig = df_sig['xs_weight'].sum()
            sum_xs_weight_back = df_back['xs_weight'].sum()
            
            if sum_xs_weight_sig<=0 or sum_xs_weight_back<=0:
                continue
            
            bin_centres_sig.append(thresholds[i])
            bin_vals_sig.append(sum_xs_weight_sig)
            bin_centres_back.append(thresholds[i])
            bin_vals_back.append(sum_xs_weight_back)
        
            s = sum_xs_weight_sig
            b = sum_xs_weight_back
            
            if ZA==True:
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
            self_dict = {k: v for k, v in self_dict.items() 
                         if type(v) is float or type(v) is np.float64}
            return self_dict
        return self_dict
    
    def to_dataframe(self, floats_only=False):
        """
        Return a pandas dataframe of the class attributes.

        Parameters
        ----------
        floats_only : bool, optional
            If floats_only is True, then only the attributes which are single
            values will be returned. Else all attribute are returned.
            The default is False.

        Returns
        -------
        results_df : pandas.DataFrame
            Dataframe representation of class attributes.
        """
        output_dict = self.to_dict(floats_only)
        results_df = pd.DataFrame([output_dict])
        return results_df

class ModelResultsMulti():
    """
    This class stores the results from multiple runs of training the EventNN 
    model.
    """
    def __init__(self):
        self.df_results = pd.DataFrame()
        self.results_list = []
        
    def add_result(self, result):
        self.results_list.append(result)
        df = result.to_dataframe(floats_only=False)
        self.df_results = pd.concat([self.df_results, df], axis=0, ignore_index=True)
        
    def return_results(self):
        return self.df_results
    
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
        epochs = len(self.df_results[history_val].iloc[0])
        all_data = self.df_results[history_val].values
        all_data = np.concatenate(all_data)
        all_data = all_data.reshape([-1,epochs])
        data_mean = all_data.mean(axis=0)
        data_std = all_data.std(axis=0)
        return data_mean, data_std
        