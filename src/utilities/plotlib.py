"""
This file contains code for producing matplotlib figures for data 
visualisation.
"""

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
import itertools
from textwrap import wrap

# ---------------------------- Plotting functions -----------------------------

def training_history_plot(history, title, figsize=(6, 4), dpi=300, full_y=False):
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
        The default is 300.
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

def confusion_matrix(cm, class_names, title, figsize=(4, 4), dpi=300, colourbar=False):
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
        Size of the matplotlib figure. The default is (4, 4).
    dpi : int, optional
        The default is 300.
    colourbar : bool, optional
        Option to add colourbar to plot. The default is False.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    
    # Setup plot
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
    
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
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def plot_roc(pred, y, figsize=(4, 4), dpi=300):
    """
    Plots the ROC curve of a keras model.
    
    Parameters
    ----------
    pred : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    figsize : TYPE, optional
        DESCRIPTION. The default is (4, 4).
    dpi : TYPE, optional
        DESCRIPTION. The default is 300.

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
    
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    return fig
