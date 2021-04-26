"""
This file contains functions for generating the combined feedforward network
and recurrent neural network architecture.
"""

# ---------------------------------- Imports ----------------------------------

USE_GPU = False

import tensorflow as tf
if USE_GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------------- Models ---------------------------------- 

def base(args):
    """
    Create a multi-input combined neural network with two branches.
    One branch uses an FFN for the event data, the other branch and RNN on the
    jet data.

    Parameters
    ----------
    args : dict
        Dictionary of values to use in creating the model. e.g.
        {'ffn_layer_1_neurons' : 16,
        'ffn_layer_2_neurons' : 8,
        'rnn_layer_1_neurons' : 64,
        'rnn_layer_2_neurons' : 8,
        'final_layer_neurons' : 8,
        'output_shape' : 1,
        'loss_function' : 'binary_crossentropy',
        'learning_rate' : 0.001}
        
    Returns
    -------
    model : tensorflow.python.keras.engine.functional.Functional
        The complete neural network
    """
    # define two seperate inputs
    inputA = keras.Input(args['event_layer_input_shape'])
    inputB = keras.Input(args['jet_layer_input_shape'], ragged=True)
    
    # create the sequential event nn
    x = layers.Dense(args['ffn_layer_1_neurons'], 
                     activation="relu", 
                     kernel_initializer='random_normal')(inputA)
    x = layers.Dense(args['ffn_layer_2_neurons'], 
                     activation="relu", 
                     kernel_initializer='random_normal')(x)
    x = keras.Model(inputs=inputA, outputs=x)
    
    # the second branch opreates on the second input
    y = keras.layers.LSTM(args['rnn_layer_1_neurons'])(inputB)
    y = keras.layers.Dense(args['rnn_layer_2_neurons'], 
                           activation="relu", 
                           kernel_initializer='random_normal')(y)
    y = keras.Model(inputs=inputB, outputs=y)
    
    # combine the output of the two branches
    combined = layers.Concatenate()([x.output, y.output])
    
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = layers.Dense(args['final_layer_neurons'], 
                     activation="relu", 
                     kernel_initializer='random_normal')(combined)
    z = layers.Dense(args['output_shape'], activation='sigmoid')(z)
    
    model = keras.Model(inputs=[x.input, y.input], outputs=z)
    
    # Compile model
    if args['learning_rate'] == 0:
        opt = keras.optimizers.Adam()
    else:
        opt = keras.optimizers.Adam(learning_rate=args['learning_rate'])
    model.compile(optimizer=opt,
                  loss=args['loss_function'],
                  metrics=['accuracy'])
    return model
