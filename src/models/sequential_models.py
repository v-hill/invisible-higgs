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

# -----------------------------------------------------------------------------  

def base(layer1, layer2, input_shape=12):
    """
    This functoin creates a dense/fully connected neuarl network with 2 hidden
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
    model.add(layers.Dense(1))
    
    # Compile model
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

def base_with_dropout(layer1, layer2, input_shape=12):
    """
    Functionally identical to base model, but with the addition of two 
    dropout layers after each hidden layer.
    Returns
    -------
    model : keras.Sequential
    """
    # Define Sequential model with 2 hidden layers
    model = keras.Sequential()
    model.add(keras.Input(shape=(input_shape,)))
    model.add(layers.Dense(layer1, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(layer2, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))
    
    # Compile model
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
