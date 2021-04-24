# -*- coding: utf-8 -*-
"""
This file contains a script to see what happens to the accuracy of a sequential
neural network when two features are taken out
"""

# Code from other files in the repo
import models.sequential_models as sequential_models

#Stops numpy from trying to capture multiple cores on cluster nodes
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


#Python modules
from mpi4py import MPI
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

def pickle_in(path):
    pickle_in = open(path,'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()
    
    return data

def grid_element(taskid):
    column1 = int(taskid/11)
    column2 = int(taskid%11)
    
    # if column2 >= column1:
    #     column2 +=1
    
    return column1,column2

class generate_grid():
    pass
    

def generate_grid_coordinates(feature_length):
    """
    A list is created with each element corrosponding to a coordinate of a 
    feature_length x feature_length grid. The list is then broken up into roughly 
    equal chunks so each chunk can be sent to a different core.

    Parameters
    ----------
    feature_length : int
        The length of the feature space you want to explore.
    numtasks:
        The number of cores the grid is being divided between

    Returns
    -------
    missing_feature_index : list
        Coordinate list that is brooken into chunks.
    """
    missing_feature_index = []
    for i in range(feature_length):
        for j in range(feature_length):
            missing_feature_index += [[i,j]]
            
    return missing_feature_index

def chunk_data(data,numtasks):
    '''
    Divids a list into roughly equal chunks is useful for MPI scatter.

    Parameters
    ----------
    data : list
        The list you want to break into chunks.
    numtasks : int
        Number of tasks the data is going to be divided between.

    Returns
    -------
    chunked_data : list
        The data that has been chunked
    '''
    if numtasks == 1:
        chunked_data = [data]
    else:
        numworkers = numtasks - 1
        step_size = int(len(data)/(numworkers))
        remainder = len(data) % numworkers
        chunked_data = [data[i * step_size : (i+1) * step_size] for i in range(numworkers)]
        chunked_data.append(data[-remainder:])
    
    return chunked_data
    

MASTER = 0
comm = MPI.COMM_WORLD
taskid = comm.Get_rank()
numtasks = comm.Get_size()

#************************* master code *******************************/
if taskid == MASTER:
    missing_feature_index = generate_grid_coordinates(12)
    missing_feature_index = chunk_data(missing_feature_index,numtasks)
    print('Data that is being divided up\n {}'.format(missing_feature_index))
    
else:
    missing_feature_index = None

recived_missing_feature_index = comm.scatter(missing_feature_index, root=MASTER)

print('I am rank {0} and I have:\n {1}'.format(taskid,recived_missing_feature_index))
    
#     #-------------------------------- Data load ----------------------------
    
#     feature_data = pickle_in('feature_data.pickle')
#     event_labels = pickle_in('event_labels.pickle')
#     sample_weight = pickle_in('sample_weight.pickle')
#     columns = pickle_in('columns.pickle')
    
#     results = open('results.txt','w')
#     results.write('Missing column 1, Missing coulumn 2, Accuracy\n')
#     results.close()

# #************************* workers code **********************************/
# elif taskid != MASTER:
#     feature_data = None
#     columns = None
#     event_labels = None
#     sample_weight = None
    

# # ------------------------------ Task is divided up ---------------------------

# grid_element_id = grid_element(taskid)

# feature_data = comm.bcast(feature_data, root=0)

# feature_data = np.delete(feature_data,grid_element_id[0],1)
# feature_data = np.delete(feature_data,grid_element_id[1],1)

# event_labels = comm.bcast(event_labels,root=0)
# sample_weight = comm.bcast(sample_weight,root=0)
# columns = comm.bcast(columns,root=0)

# missing_columns = [columns[grid_element_id[0]],columns[grid_element_id[1]+1]]

# # ------------------------------ Model training -------------------------------


# test_fraction = 0.2
# data_train, data_test, labels_train, labels_test, sw_train, sw_test  = \
#     train_test_split(feature_data, event_labels, sample_weight, test_size=test_fraction)


# accuracy = []
# for i in range(0,5):
#     tf.keras.backend.clear_session()
#     model = sequential_models.base(42, 4, input_shape= 10)
#     history = model.fit(data_train, labels_train, validation_data=(data_test, labels_test), sample_weight=sw_train, epochs=16, verbose=2)
#     test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=2)
#     accuracy += [test_acc]

# accuracy = np.mean(accuracy)

# results = open('results.txt','a')
# results.write('{0}, {1}, {2}\n'.format(missing_columns[0],missing_columns[1],accuracy))
# results.close()








