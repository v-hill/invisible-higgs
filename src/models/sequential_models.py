"""
This file contains functions for generating sequential neural network 
models.
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

def base(args):
    """
    This function creates a dense/fully connected neuarl network with 2 hidden
    layers. Uses custom learning rate for the Adam optimiser. Uses 
    'binary_crossentropy' loss function.

    Parameters
    ----------
    args : dict
        Dictionary of values to use in creating the model.
        Contains the following values as an example:
             {'layer_1_neurons' : 64,
              'layer_2_neurons' : 8,
              'output_shape' : 1,
              'learning_rate' : 0.001}

    Returns
    -------
    model : keras.Sequential
    """
    # Define Sequential model with 2 hidden layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(args['event_layer_input_shape'],)))
    model.add(layers.Dense(args['layer_1_neurons'], 
                           activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(args['layer_2_neurons'], 
                           activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(args['output_shape'], 
                           activation='sigmoid', 
                           kernel_initializer='random_normal'))
    # Compile model
    if args['learning_rate'] == 0:
        opt = keras.optimizers.Adam()
    else:
        opt = keras.optimizers.Adam(learning_rate=args['learning_rate'])
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def base_with_dropout(args, dropout=0.2):
    """
    Functionally identical to base model, but with the addition of two 
    dropout layers after each hidden layer.
    
    Returns
    -------
    model : keras.Sequential
    """
    # Define Sequential model with 2 hidden layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(args['event_layer_input_shape'],)))
    model.add(layers.Dense(args['layer_1_neurons'], 
                           activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(args['layer_2_neurons'], 
                           activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(args['output_shape'], 
                           activation='sigmoid', 
                           kernel_initializer='random_normal'))
    # Compile model
    if args['learning_rate'] == 0:
        opt = keras.optimizers.Adam()
    else:
        opt = keras.optimizers.Adam(learning_rate=args['learning_rate'])
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def multi_class_base(args):
    """
    This function creates a dense/fully connected neural network with 2 
    hidden layers for multi label classification.
    Parameters
    ----------
    layer1 : int
        Number of neurons in layer 1.
    layer2 : int
        Number of neurons in layer 2.
    input_shape : int, optional
        The input shape of the data. The default is 11.
    Returns
    -------
    model : keras.Sequential
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(args['event_layer_input_shape'],)))
    model.add(layers.Dense(args['layer_1_neurons'], activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(args['layer_2_neurons'], activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(args['output_shape'], activation='softmax', 
                           kernel_initializer='random_normal'))
    
    # Compile model
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
