# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 18:41:41 2020

@author: user
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random
import missingno as msno
from sklearn.decomposition import PCA

def list_files(ROOT, filename_hd5="df_ml_inputs.hd5"):
    """
    Function to list all files containing data

    Parameters
    ----------
    ROOT : str
        The root directory
    filename_hd5 : str, optional
        The default is "df_ml_inputs.hd5".

    Returns
    -------
    filepaths : list
        List of files called 'df_ml_inputs.hd5'

    """
    filepaths = []
    for path, subdirs, files in os.walk(ROOT):
        for name in files:
            if filename_hd5 in name:
                filepaths.append(os.path.join(path, name))
    return filepaths

def collect_data(all_files, input_files):
    """
    Funcion to create a pandas dataframe with all the data specificed by the 
    input_files list.

    Parameters
    ----------
    all_files : list
        list of all files
    input_files : list
        files to be containing in the dataframe

    Returns
    -------
    li : list of pandas dataframes
    """
    li = []
    for filename in all_files:
        for t in input_files:
            if t in filename:
                df = pd.read_hdf(filename)
                print(f"{filename.split('ml_postproc')[-1]:42}: {df.shape}")
                li.append(df)
    return li

def get_columns_list(data, columns_to_ignore):
    """
    This function generates a list of the columns to be used in the training 
    data.

    Parameters
    ----------
    data : pd.DataFrame
        All collected data
    columns_to_ignore : list
        List of columns to explicitly exclude

    Returns
    -------
    columns_filtered : list
        list of columns to use for training
    """
    columns = list(data.columns.values)
    columns_filtered = []
    for idx, col in enumerate(columns):
        if isinstance(data[col].iloc[idx ], (np.float32, np.float64, np.int64, np.uint32)):
            if col not in columns_to_ignore:
                columns_filtered.append(col)
                print(f"{col:<32}: {data[col].dtypes}")
    return columns_filtered

def ExtractListElement(List,index=1):
    """Trys to see if possible to extract a element for a given index
    
    Parameters
    ----------
    List : list
        List for element extraction
    index : int
        The index of the element to extract 
        
    Returns
    -------
    Value : float,np.nan 
        The value of the second element of jet data if it exists or a np.nan 
        if that data dosn't exist. 
    """
    Value = 0
    try:
        Value = List[index]
    except:
        Value = np.nan
        
    return Value

def UnpackLeadingJet_data(data,number_of_jets=2):
    """Creates a DataFrame of the object data from the first and
    second leading jet
    
    Parameters
    ----------
    data : pandas dataframe
        The dataframe containing the jet data 
    number_of_jets : int,optional
        The number of jets you want data from
    """
    df=pd.DataFrame()
    columns=list(data.columns)
    for idx,col in enumerate(columns):
        if isinstance(data[col].iloc[0],np.ndarray):
            if col != 'cleanJetMask':
                df[col + '_1'] = data[col].apply(lambda x: x[0])
                df[col + '_2'] = data[col].apply(lambda x: ExtractListElement(x,number_of_jets-1))
            
    return df

def Normalise(feature_data):
    """This function normalises data from a filtered dataframe, dataframe only
    contains features that are gonna be passed into a neural net.
    
    Parameters
    ----------
    feature_data : pd.DataFrame
        Data containing feature infomation
    scale : function, opt
        The function to scale the data
    
    Returns
    -------
    event_data : np.darray
        Array of shape (m,n) where m indexes the different simulated event 
        data and n indexes the features from an event.
    
    """
    event_data = []
    for col in feature_data.columns:
        feature = feature_data[col].values.reshape(-1,1)
        if feature_data[col].skew() >= 2.0:
            feature = np.log(feature)
        else:
            pass
        scale=preprocessing.MinMaxScaler()
        feature = scale.fit_transform(feature)
        event_data.append(feature.reshape(-1,))
    
    event_data = np.array(event_data).T
    
    return event_data
        
def SplitData(event_data,labels,weights,test_size,shuffle=True):
    """This function splits a numpy array containg event data into test 
    and train sets matched with the right labels.
    
    Parameters
    ----------
    event_data : np.darray
        Event data in a (m,n) array where m is the different simulated event 
        data and n is the features from an event.
    labels : np.darray
        The labels that correspond to the different events
    weights : np.darray
        The weights for each event to control for cross section of different 
        events
    test_size : float
        The fraction of the data that is test data
    shuffle : bool, optional
        True if you want the data shuffled
        
    Results
    -------
    training_data : list 
        Data in a [(m,n),l,w] list where l is the label for an event and w is 
        weight. If shuffle is true then the events have been shuffled. This
        data is to be trained on.
    test_data : list 
        Data in a [(m,n),l,w] this data is for a network to be tested on.

    """
    if shuffle is True:    
        #Shuffle the training data by the same amount
        rng_state = np.random.get_state()
        
        np.random.shuffle(event_data)
        np.random.set_state(rng_state)
        
        np.random.shuffle(labels)
        np.random.set_state(rng_state)
        
        np.random.shuffle(weights)
        #test data does not need to be shuffled
    
    else:
        pass
    
    train_length = int(round(len(event_data)*(1-test_size),0))
    
    training_data = event_data[:train_length,]
    training_label = labels[:train_length]
    training_weight = weights[:train_length]

    test_data = event_data[train_length:,]
    test_label = labels[train_length:]
    test_weight = weights[train_length:]
    
    
    training_data = [training_data,
                     training_label,
                     training_weight]
    
    test_data = [test_data,
                 test_label,
                 test_weight]
    
    return training_data,test_data


ROOT = "\\Users\\user\\Documents\\Fifth Year\\Resarch-Project\\Data"
input_files = ['ttH125_part1-1', 'TTTo2L2Nu', 'TTToHadronic', 'TTToSemiLeptonic']

all_files = list_files(ROOT)
data_list = collect_data(all_files, input_files)

data = pd.concat(data_list, axis=0, ignore_index=True)

#Unpack the data from the first two jets.
jet_data = UnpackLeadingJet_data(data)

#Merge jet data and event data and drop any nan
data = data.join(jet_data)
data = data.dropna()


# List the columns to explicitly exclude
print("The following columns will be used to train the NN:")
columns_to_ignore = ['entry', 'weight_nominal', 'hashed_filename']
columns_filtered = get_columns_list(data, columns_to_ignore)

# Make a dictionary of 'dataset' : label
label_dict = {'ttH125' : 0, 'TTTo2L2Nu' : 1, 'TTToHadronic' : 1, 'TTToSemiLeptonic' : 1}


# Make labels 
labels = data['dataset']
labels = labels.replace(label_dict)
labels = labels.values

# Make sample weights
sample_weight = data['weight_nominal']
sample_weight = sample_weight.values


# Select only the filtered columns from the data
data = data[columns_filtered]

# event_data = Normalise(data)

# train,test = SplitData(event_data,labels,sample_weight,0.2)

# data_train = train[0]
# labels_train = train[1]
# weights_train = train[2]

# data_test = test[0]
# labels_test = test[1]
# weights_train = test[2]


# inputs = tf.keras.Input(shape=(data_train.shape[-1]))

# x = tf.keras.layers.Dense(64, activation="relu")(inputs)
# x = tf.keras.layers.Dense(16, activation="relu")(x)
# x = tf.keras.layers.Dense(2)(x)

# model_events = tf.keras.Model(inputs, x)

# model_events.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])#,
#               #loss_weights=sample_weight)

# # Fit and evalute model
# history = model_events.fit(data_train, labels_train, epochs=15)
# test_loss, test_acc = model_events.evaluate(data_test, labels_test, verbose=2)
# print('\nTest accuracy:', test_acc)

# # summarize history for accuracy

# fig = plt.figure(figsize=(8, 6), dpi=100)

# plt.plot(history.history['accuracy'])

# plt.title('Simple NN model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch number')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# #Get data sample
# data_sample=data.sample(frac=0.01)

# #Normalise the data
# scaler = preprocessing.StandardScaler()
# scaler.fit(data_sample)
# scaled_data = scaler.transform(data_sample)

# #Fit PCA to data
# pca = PCA(n_components=2)
# pca.fit(scaled_data)

# #Project data on new pca axis
# projected_data = pca.transform(scaled_data)

# axis1 = projected_data[:,0]
# axis2 = projected_data[:,1]

# fig,ax = plt.subplots()
# ax.scatter(axis1,axis2)









