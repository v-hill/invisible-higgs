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

def training_history_plot(history, title, figsize=(6, 4), dpi=200, full_y=False):
    """
    Plots the training history of the neural network.

    Parameters
    ----------
    history : tensorflow.python.keras.callbacks.History
        Model training history
    title : str
        Plot title
    figsize : tuple, optional
        Size of the matplotlib figure. The default is (6, 4).
    dpi : int, optional
        The default is 200.
    full_y : bool, optional
        Set to true in order to set the y-axis limits to (0, 1). 
        The default is False, giving automatic scaling.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if full_y:
        plt.ylim(0, 1)
    
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch number')
    plt.legend(['train', 'test'], loc='upper left')
    return fig

def confusion_matrix(cm, class_names, title, figsize=(4, 4), dpi=200, colourbar=False):
    """
    Plots the confusion matrix.

    Parameters
    ----------
    cm : numpy.ndarray
        The input confusion matrix
    class_names : list
        List of class names 
    title : str
        Plot title
    figsize : tuple, optional
        Size of the matplotlib figure. The default is (6, 4).
    dpi : int, optional
        The default is 200.
    colourbar : bool, optional
        Option to add colourbar to plot. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # Setup plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    plt.title("\n".join(wrap(title, 30)))
    if colourbar:
        plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="k")
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def plot_roc(pred, y, title, figsize=(6, 4), dpi=200):
    """
    Plots the ROC curve of a keras model.
    
    Parameters
    ----------
    pred : numpy.ndarray
        Predicted labels
    y : numpy.ndarray
        True labels
    figsize : tuple, optional
        DESCRIPTION. The default is (6, 4).
    dpi : int, optional
        DESCRIPTION. The default is 200.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.title(title)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
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

def plot_discriminator_vals(pred_signal, pred_background, title, figsize=(6, 4), dpi=200):
    """
    Plots the distribution of discriminator values for a keras model.

    Parameters
    ----------
    pred_signal : numpy.ndarray
        Array of discriminator values for known signal data
    pred_background : numpy.ndarray
        Array of discriminator values for known background data
    title : str
        Plot title
    figsize : tuple, optional
        DESCRIPTION. The default is (6, 4).
    dpi : int, optional
        DESCRIPTION. The default is 200.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    bins = np.linspace(0, 1, 50)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title)
    plt.xlabel("Label prediction")
    plt.ylabel("Number of events")
    # plt.xlim(0, 10)
    
    plt.hist(pred_signal, bins, alpha=0.5, label='ttH (signal)', color='brown')
    plt.hist(pred_background, bins, alpha=0.5, label='tt¯ (background)', color='teal')
    
    plt.legend(loc='upper right')
    return fig
    
