"""
This file contains code for producing matplotlib figures for data 
visualisation.
"""

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
import itertools
from textwrap import wrap
from scipy import interp
from sklearn.metrics import roc_curve, auc

# ---------------------------- Plotting functions -----------------------------

def training_history_plot(history_training_data, history_test_data, params, error_bars=[]):
    """
    Plots the training history of the neural network.

    Parameters
    ----------
    model_results : pandas.DataFrame
        Dataframe containing the training history data to plot.
    params : dict
        Plot parameters.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=params['figsize'], dpi=params['dpi'])
    epochs = np.arange(len(history_training_data))+1
    
    if len(error_bars)==0:
        ax.plot(epochs, history_training_data, '-x', 
                lw=1, color=params['colors'][0])
        ax.plot(epochs, history_test_data, '-x', 
                lw=1, color=params['colors'][1])
        ax.annotate(f'max:\n{history_training_data[-1]:0.3f}', 
                    (epochs[-1], history_training_data[-1]), 
                    xytext=(0, -15), textcoords='offset points',
                    ha='center', va='center', 
                    fontsize=8, color=params['colors'][0])
    else:
        ax.errorbar(epochs, history_training_data, 
                     yerr=error_bars[0], 
                     capsize=3, lw=1, ls='--' , 
                     color=params['colors'][0])
        ax.errorbar(epochs, history_test_data, 
                     yerr=error_bars[1], 
                     capsize=3, lw=1, ls='--' ,
                     color=params['colors'][1])
        ax.annotate(f'max accuracy:\n{history_training_data[-1]:0.4f} \u00B1 {error_bars[0][-1]:0.4f}', 
                    (epochs[-1], history_training_data[-1]), 
                    xytext=(10, -35), textcoords='offset points',
                    ha='right', va='center', 
                    fontsize=8, color=params['colors'][0])


    if params['full_y']:
        plt.ylim(0, 1)
    
    plt.title(params['title'])
    plt.xlabel(params['x_axis'])
    plt.ylabel(params['y_axis'])
    plt.legend(params['legend'], loc='lower right')
    return fig

def confusion_matrix(model_results, params):
    """
    Plots the confusion matrix of the neural network evaluated on the test 
    data.

    Parameters
    ----------
    model_results : pandas.DataFrame
        Dataframe containing the confusion matrix data to plot.
    params : dict
        Plot parameters.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Setup plot
    fig = plt.figure(figsize=params['figsize'], dpi=params['dpi'])
    
    plt.title("\n".join(wrap(params['title'], 36)))
    plt.xlabel(params['x_axis'])
    plt.ylabel(params['y_axis'])

    tick_marks = np.arange(len(params['class_names']))
    plt.xticks(tick_marks, params['class_names'], rotation=45)
    plt.yticks(tick_marks, params['class_names'])
    
    # Normalize the confusion matrix
    cm = model_results['confusion_matrix']
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="k")
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    if params['colourbar']:
        plt.colorbar()
    return fig

def mulit_confusion_matrix(cm_list, class_names, network_names, figsize=(4, 4), dpi=200, norm=True):
    """
    This function returns multiple confusion matrices ploted on the same grid
    to make comparison between networks easier.
    
    Parameters
    ----------
    cm_list : list or array
        Array containing the different confusion matrices.
    class_names : list or array
        Array containg the names of the different classes that are being 
        distinguished.
    network_names : list or array
        Array containg the names of the different networks that are being 
        compared.
    title : str
        title 
    figsize : TYPE, optional
        figsize. The default is (4, 4).
    dpi : TYPE, optional
        The default is 200.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(ncols=len(cm_list), sharex=True, sharey=True)
    axes[0].set_ylabel('True label')
    tickmarks = np.arange(len(class_names))
    plt.yticks(tickmarks,class_names)
    plt.xticks(tickmarks,class_names)
    
    for idx,ax in enumerate(axes):
        ax.set(adjustable='box', aspect='equal')
        ax.set_xlabel('Predicted label')
        ax.set_title(network_names[idx])
        
        if norm == True:
            cm_list[idx] = np.around(cm_list[idx].astype('float') / cm_list[idx].sum(axis=1)[:, np.newaxis], decimals=2)
        else:
            pass
        
        ax.imshow(cm_list[idx],interpolation='nearest', cmap=plt.cm.Reds)
        
        for i, j in itertools.product(range(cm_list[idx].shape[0]), range(cm_list[idx].shape[1])):
            ax.text(j, i, cm_list[idx][i, j], horizontalalignment="center", color="k")
            
    return fig

def plot_roc(model_results, params):
    """
    Plots the ROC curve of a trained keras model.

    Parameters
    ----------
    model_results : pandas.DataFrame
        Dataframe containing the ROC curve results to plot.
    params : dict
        Plot parameters.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=params['figsize'], dpi=params['dpi'])
    
    plt.plot(model_results['roc_fpr_vals'],
             model_results['roc_tpr_vals'], 
             label=f'ROC curve (area = {model_results["roc_auc"]:0.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlim([0.0, 1.005])
    plt.ylim([0.0, 1.005])
    plt.title(params['title'])
    plt.xlabel(params['x_axis'])
    plt.ylabel(params['y_axis'])
    plt.legend(loc='lower right')
    return fig

def plot_multi_class_roc(pred, y, title, class_labels, figsize=(6, 4), dpi=200):
    """
    Plots roc curves for multi label classification by transforming each label
    into a binary classifier problem.
    Parameters
    ----------
    pred : numpy.ndarray
        Array of predicted values.
    y : numpy.ndarray
        Array of target values.
    title : str
        ROC curve title.
    class_label : list
        List containg class label names as strings.
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Creates storage locations
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(class_labels)
    
    # Calculate binary ROC curves for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    # Average mean_tpr
    mean_tpr /= n_classes
    
    #Compute AUC    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot([0, 1], [0, 1], 'k--')
    
    for i in range(n_classes):
        label = '{0} | auc = {1:.2f}'.format(class_labels[i],roc_auc[i])
        ax.plot(fpr[i], tpr[i], label = label)
        
    # Plot micro/macro-average ROC        
    ax.plot(fpr["micro"], tpr["micro"],
         label='micro-average | area = {0:0.2f}'
               ''.format(roc_auc["micro"]),
               linestyle=':',linewidth=2)
    
    ax.plot(fpr["macro"], tpr["macro"],
         label='macro-average | area = {0:0.2f}'
               ''.format(roc_auc["macro"]),
               linestyle=':',linewidth=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    return fig

def plot_multi_signal_roc(pred, y, title, class_labels, figsize=(6, 4), dpi=200):
    """
    Plots roc curves for multi label classification by transforming each label
    into a binary classifier problem.
    Parameters
    ----------
    pred : numpy.ndarray
        Array of predicted values.
    y : numpy.ndarray
        Array of target values.
    title : str
        ROC curve title.
    class_label : list
        List containg class label names as strings.
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Creates storage locations
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    signal_idx = []
    signal_lab = []
    
    # Calculate binary ROC curves for each class
    for idx,lab in enumerate(class_labels):
        fpr[lab], tpr[lab], _ = roc_curve(y[:, idx], pred[:, idx])
        roc_auc[lab] = auc(fpr[lab], tpr[lab])
        
        if lab[:3] == 'sig':
            signal_idx += [idx]
            signal_lab += [lab]
            
    # Compute micro-average ROC curve and ROC area for signal classes
    fpr['micro'], tpr['micro'], _ = roc_curve(y[:,signal_idx].ravel(), pred[:,signal_idx].ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # Aggregate all false positive rates
    signal_fpr = np.unique(np.concatenate([fpr[lab] for lab in signal_lab]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(signal_fpr)
    for lab in signal_lab:
        mean_tpr += interp(signal_fpr, fpr[lab], tpr[lab])
        
    # Average mean_tpr
    n_signal_classes = len(signal_lab)
    mean_tpr /= n_signal_classes
    
    #Compute AUC    
    fpr["macro"] = signal_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot([0, 1], [0, 1], 'k--')
    
    for lab in class_labels:
        if lab in signal_lab:
            label = '{0} | auc = {1:.2f}'.format(lab,roc_auc[lab])
            ax.plot(fpr[lab], tpr[lab], label = label)
        else:
            label = '{0} | auc = {1:.2f}'.format(lab,roc_auc[lab])
            ax.plot(fpr[lab], tpr[lab], label = label, alpha=0.2)
        
    # Plot micro/macro-average ROC        
    ax.plot(fpr["micro"], tpr["micro"],
         label='micro-average | area = {0:0.2f}'
               ''.format(roc_auc["micro"]),
               linestyle=':',linewidth=2)
    
    ax.plot(fpr["macro"], tpr["macro"],
         label='macro-average | area = {0:0.2f}'
               ''.format(roc_auc["macro"]),
               linestyle=':',linewidth=2)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    return fig

def plot_discriminator_vals(pred_signal, pred_background, params):
    """
    Plots the distribution of discriminator values for a keras model.

    Parameters
    ----------
    pred_signal : numpy.ndarray
        Array of discriminator values for known signal data
    pred_background : numpy.ndarray
        Array of discriminator values for known background data
    params : dict
        Plot parameters.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """

    fig = plt.figure(figsize=params['figsize'], dpi=params['dpi'])
    
    bins = np.linspace(0, 1, params['num_bins'])
    
    plt.hist(pred_signal, bins, alpha=0.5, 
             label='ttH (signal)', color=params['colors'][0])
    plt.hist(pred_background, bins, alpha=0.5, 
             label='tt¯ (background)', color=params['colors'][1])
    
    plt.title("\n".join(wrap(params['title'], 50)))
    plt.xlabel(params['x_axis'])
    plt.ylabel(params['y_axis'])
    plt.legend(loc='upper right')
    return fig

def significance(bin_centres_sig, sig_vals, params):
    """
    Significance plot.

    Parameters
    ----------
    bin_centres_sig : list
        List of discriminator values.
    sig_vals : numpy.ndarray
        Significance of each discriminator threshold value.
    params : dict
        Plot parameters.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Setup plot
    fig = plt.figure(figsize=params['figsize'], dpi=params['dpi'])
    
    plt.plot(bin_centres_sig, sig_vals)
    plt.xlim(-0.1, 1)
    plt.title("\n".join(wrap(params['title'], 60)))
    plt.xlabel(params['x_axis'])
    plt.ylabel(params['y_axis'])
    return fig
