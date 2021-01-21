"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# Code from other files in the repo
from utilities.data_loader import  DataLoader
import models.sequential_models as sequential_models
import models.recurrent_models as recurrent_models
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker
# from utilities.data_preprocessing import split_data

# Python libraries
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

ROOT = "C:\\{Directory containing data}\\ml_postproc\\"
data_to_collect = ['ttH125_part1-1', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']
