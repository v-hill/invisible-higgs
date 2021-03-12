"""
This file contains functions for generating the multi-input neural network models.
"""

# ---------------------------------- Imports ----------------------------------

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

# ----------------------------------- Models ---------------------------------- 

def base1(input_shape1=11, input_shape2=[None, 6]):
    """
    Create a multi-input combined neural network with two branches.
    One branch uses an FFN for the event data, the other branch and RNN on the
    jet data.

    Parameters
    ----------
    input_shape1 : int, optional
        Shape of event level variable data. The default is 11.
    input_shape2 : list, optional
        Dimensions of ragged tensor jet data. The default is [None, 6].

    Returns
    -------
    model : tensorflow.python.keras.engine.functional.Functional
        The complete neural network
    """
    # define two seperate inputs
    inputA = keras.Input(input_shape1)
    inputB = keras.Input(input_shape2, ragged=True)
    
    # create the sequential event nn
    x = layers.Dense(42, activation="relu")(inputA)
    x = layers.Dense(4, activation="relu")(x)
    x = keras.Model(inputs=inputA, outputs=x)
    
    # the second branch opreates on the second input
    y = keras.layers.LSTM(64)(inputB)
    y = keras.layers.Dense(4, activation="relu")(y)
    y = keras.Model(inputs=inputB, outputs=y)
    
    # combine the output of the two branches
    combined = layers.Concatenate()([x.output, y.output])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = layers.Dense(2, activation="relu")(combined)
    z = layers.Dense(1, activation='sigmoid')(z)
    
    model = keras.Model(inputs=[x.input, y.input], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def multi_label_base(input_shape1=11, input_shape2=[None, 6], output_shape=4):
    """
    Create a multi-input combined neural network with two branches.
    One branch uses an FFN for the event data, the other branch and RNN on the
    jet data. This model is used for multilabel classification

    Parameters
    ----------
    input_shape1 : int, optional
        Shape of event level variable data. The default is 11.
    input_shape2 : list, optional
        Dimensions of ragged tensor jet data. The default is [None, 6].

    Returns
    -------
    model : tensorflow.python.keras.engine.functional.Functional
        The complete neural network
    """
    # define two seperate inputs
    inputA = keras.Input(input_shape1)
    inputB = keras.Input(input_shape2, ragged=True)
    
    # create the sequential event nn
    x = layers.Dense(64, activation="relu")(inputA)
    x = layers.Dense(8, activation="relu")(x)
    x = keras.Model(inputs=inputA, outputs=x)
    
    # the second branch opreates on the second input
    y = keras.layers.LSTM(64)(inputB)
    y = keras.layers.Dense(8, activation="relu")(y)
    y = keras.Model(inputs=inputB, outputs=y)
    
    # combine the output of the two branches
    combined = layers.Concatenate()([x.output, y.output])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = layers.Dense(8, activation="relu")(combined)
    z = layers.Dense(output_shape, activation='softmax')(z)
    
    model = keras.Model(inputs=[x.input, y.input], outputs=z)
    model.compile(optimizer='adam', loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model


