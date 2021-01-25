"""
This file contains functions for generating recurrent neural network models.
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

# -----------------------------------------------------------------------------  

def base(input_shape=[None, 6]):
    """
    This function creates a neural network capable of taking as input the 
    variable length jet data in the form of a ragged tensor. This consists
    of a single hidden LSTM layer.

    Parameters
    ----------
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
        keras.layers.InputLayer(input_shape=input_shape, ragged=True),
        keras.layers.LSTM(64),  
        keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

model = base()
