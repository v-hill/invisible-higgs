"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# Code from other files in the repo
from utilities.data_preprocessing import make_ragged_tensor
from utilities.data_preprocessing import normalise_jet_columns

# Python libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#------------------------------- Load event data ------------------------------

event_data = np.load('preprocessed_event_data.npy', allow_pickle=True)
event_labels = np.load('preprocessed_event_labels.npy', allow_pickle=True)
sample_weight = np.load('preprocessed_sample_weights.npy', allow_pickle=True)

# Generate a fixed random state
random_state = np.random.randint(50)
test_fraction = 0.2

data_train1, data_test1, labels_train, labels_test, sw_train, sw_test  = \
            train_test_split(event_data,
                             event_labels,
                             sample_weight,
                             test_size=test_fraction,
                             random_state=random_state)

#-------------------------------- Load jet data -------------------------------

# Load in data
df_jet_data = pd.read_hdf('preprocessed_jet_data.hdf')

data_train2, data_test2, labels_train, labels_test, sw_train, sw_test  = \
            train_test_split(df_jet_data,
                             event_labels,
                             sample_weight,
                             test_size=test_fraction,
                             random_state=random_state)

#normalise the training set
jet_data_train_df = normalise_jet_columns(data_train2)
jet_data_test_df = normalise_jet_columns(data_test2)

jet_data_train_rt = make_ragged_tensor(jet_data_train_df)
jet_data_test_rt = make_ragged_tensor(jet_data_test_df)

# # TODO: Explore using parallel progrmaing to train each model

#------------------------- Create combined neural net -------------------------

from tensorflow import keras
from tensorflow.keras import layers

# define two seperate inputs
inputA = keras.Input(shape=12)
inputB = keras.Input(shape=[None, 6], ragged=True)

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
z = layers.Dense(1, activation='softmax')(z)

model = keras.Model(inputs=[x.input, y.input], outputs=z)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=[data_train1, jet_data_train_rt], y=labels_train, 
          validation_data=([data_test1, jet_data_test_rt], labels_test), 
          sample_weight=sw_train, 
          epochs=16, 
          verbose=2)

