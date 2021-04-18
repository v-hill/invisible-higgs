"""
This file contains functions for generating sequential neural network 
models.
"""

USE_GPU = False

import tensorflow as tf
if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("Running tensorflow on GPU")
else:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("Running tensorflow on CPU")

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# -----------------------------------------------------------------------------  

def base(layer1, layer2, input_shape=11):
    """
    This function creates a dense/fully connected neuarl network with 2 hidden
    layers.
    Parameters
    ----------
    layer1 : int
        Number of neurons in the first layer
    layer2 : int
        Number of neurons in the second layer
    input_shape : int, optional
        The default is 12.
    Returns
    -------
    model : keras.Sequential
    """
    # Define Sequential model with 2 hidden layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(layer1, activation="relu"))
    model.add(layers.Dense(layer2, activation="relu"))
    model.add(layers.Dense(2))
    
    # Compile model
    model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

def base2(layer1, layer2, input_shape=11, learning_rate=0.01):
    """
    base + custom learning rate for the Adam optimiser
         + 'binary_crossentropy' loss function

    Parameters
    ----------
    layer1 : int
        Number of neurons in the first layer
    layer2 : int
        Number of neurons in the second layer
    input_shape : int, optional
        The default is 12.

    Returns
    -------
    model : keras.Sequential
    """
    # Define Sequential model with 2 hidden layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(layer1, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(layer2, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(1, activation='sigmoid', 
                           kernel_initializer='random_normal'))
    
    # Compile model
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def base3(layer1, layer2, input_shape=11):
    """
    base2 with standard learning rate

    Parameters
    ----------
    layer1 : int
        Number of neurons in the first layer
    layer2 : int
        Number of neurons in the second layer
    input_shape : int, optional
        The default is 12.

    Returns
    -------
    model : keras.Sequential
    """
    # Define Sequential model with 2 hidden layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(layer1, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(layer2, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(1, activation='sigmoid', 
                           kernel_initializer='random_normal'))
    
    # Compile model
    model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def base_with_dropout(layer1, layer2, input_shape=11, dropout=0.2):
    """
    Functionally identical to base3 model, but with the addition of two 
    dropout layers after each hidden layer.
    Returns
    -------
    model : keras.Sequential
    """
    # Define Sequential model with 2 hidden layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(layer1, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(layer2, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid', 
                           kernel_initializer='random_normal'))
    
    # Compile model
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def multi_class_base(layer1, layer2, input_shape=11, output_shape=4):
    """
    This function creates a dense/fully connected neural network with 2 
    hidden layers for multi label classification.
    Parameters
    ----------
    layer1 : int
        Number of neurons in layer 1 .
    layer2 : int
        Number of neurons in layer 2 .
    input_shape : int, optional
        The input shape of the data. The default is 11.
    Returns
    -------
    model : keras.Sequential
    """
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(layer1, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(layer2, activation='relu', 
                           kernel_initializer='random_normal'))
    model.add(layers.Dense(output_shape, activation='softmax', 
                           kernel_initializer='random_normal'))
    
    # Compile model
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

models_list = [base, base2, base_with_dropout]
