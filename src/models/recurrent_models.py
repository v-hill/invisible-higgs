"""
This file contains functions for generating recurrent neural network models.
"""

# ---------------------------------- Imports ----------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

USE_GPU = False
if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Running tensorflow on GPU")
else:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("Running tensorflow on CPU")

# -----------------------------------------------------------------------------  

def base(layer1, layer2, input_shape=[None, 8]):
    """
    This function creates a neural network capable of taking as input the 
    variable length jet data in the form of a ragged tensor. This consists
    of a single hidden LSTM layer.

    Parameters
    ----------
    layer1 : int
        Number of neurons in layer 1, the LSTM layer.
    layer2 : int
        Number of neurons in layer 2.
    input_shape : int, optional
        The shape of jet event. The first entry 'None' specifies an unknown
        number of jets. The second entry (with default of 6), the number of 
        variables which characterise a single jet. 

    Returns
    -------
    model : keras.Sequential
    """
    # Define an RNN with a single LSTM layer
    model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape, ragged=True),
        layers.LSTM(layer1, kernel_initializer='random_normal'),
        layers.Dense(layer2, activation='relu', kernel_initializer='random_normal'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def base_custom_learn(input_shape=[None, 6], learning_rate=0.00003):
    # Define an RNN with a single LSTM layer
    model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape, ragged=True),
        layers.LSTM(64),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def multi_labels_base(layer1, layer2, input_shape=[None, 6], output_shape=4):
    """
    This function creates a neural network for multilabel classification.
    
    Parameters
    ----------
    layer1 : int
        Number of neurons in layer 1, the LSTM layer.
    layer2 : int
        Number of neurons in layer 2.
    input_shape : int, optional
        The input shape of the data. The default is [None, 6].
        
    Returns
    -------
    model : keras.Sequential
    """
    # Define an RNN with a single LSTM layer
    model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape, ragged=True),
        layers.LSTM(layer1),
        layers.Dense(layer2, activation='relu'),
        layers.Dense(output_shape, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])    
    return model
