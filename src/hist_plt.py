# -*- coding: utf-8 -*-
"""
This aim of this script is to produce histograms of the different event 
variables.
"""

# Code from other files in the repo
from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker
from utilities.data_preprocessing import normalise_jet_columns

# Python libraries
import copy
import numpy as np
import matplotlib.pyplot as plt

ROOT = "C:\\Users\\user\\Documents\\Fifth Year\\ml_postproc"
data_to_collect = ['ttH125_part1-1', 
                   'TTTo2L2Nu', 
                   'TTToHadronic', 
                   'TTToSemiLeptonic']

# -------------------------------- Data setup --------------------------------

loader = DataLoader(ROOT)
loader.find_files()
loader.collect_data(data_to_collect)
data = DataProcessing(loader)

data.data.weight_nominal = WeightMaker.weight_nominal_sample_weights(data)

cols_to_ignore1 = ['entry', 'weight_nominal', 'hashed_filename']
cols_to_ignore2 = ['cleanJetMask']

cols_events = data.get_event_columns(cols_to_ignore1)
cols_jets = data.get_jet_columns(cols_to_ignore2)

signal_list = ['ttH125']
data.label_signal_noise(signal_list)

data.data.weight_nominal = WeightMaker.weight_nominal_sample_weights(data)

signal_df = data.data[data.data.dataset == 'inv_signal']
background_df = data.data[data.data.dataset != 'inv_signal']

sample_weight_signal = signal_df.weight_nominal.values
sample_weight_background = background_df.weight_nominal.values

# sample_weight_signal *= total_len / len(sample_weight_signal)
# sample_weight_background *= total_len / len(sample_weight_background)

'''This produces the weighted hist plots if you only care about the ratio of the
signal to background'''
# for col in cols_events:
    
#     fig,axes = plt.subplots(nrows=2, ncols=2)
#     ax1,ax2,ax3,ax4 = axes.flatten()
    
#     ax1.hist(signal_df[col], bins=100, alpha=0.5, label='Signal')
#     ax1.hist(background_df[col], bins=100, alpha=0.5, label='Background')
#     ax1.set_ylabel('Count')
#     ax1.set_xlabel(col)
#     ax1.legend()
    
#     ax2.hist(signal_df[col], bins=100, weights=sample_weight_signal, 
#               alpha=0.5, label='Signal')
#     ax2.hist(background_df[col], bins=100, weights=sample_weight_background, 
#               alpha=0.5, label='Background')
#     ax2.set_ylabel('Normalised Count')
#     ax2.set_xlabel(col)
#     ax2.legend()
    
#     ax3.hist([signal_df[col],background_df[col]], bins=100, stacked=True,
#               label=['Signal','Background'])
#     ax3.set_ylabel('Count')
#     ax3.set_xlabel(col)
#     ax3.legend()
    
#     ax4.hist([signal_df[col],background_df[col]], bins=100, 
#               weights=[sample_weight_signal,sample_weight_background],
#               stacked=True,label=['Signal','Background'])
#     ax4.set_ylabel('Normalised Count')
#     ax4.set_xlabel(col)
#     ax4.legend()
    
'''This produces the weighted hist plots if you want the different production
events distinguished'''
# signal_df = data.data[data.data.dataset == 'ttH125']
# background_df1 = data.data[data.data.dataset == 'TTTo2L2Nu']
# background_df2 = data.data[data.data.dataset == 'TTToHadronic']
# background_df3 = data.data[data.data.dataset == 'TTToSemiLeptonic']

# sample_weight0 = signal_df.sample_weight.values
# sample_weight1 = background_df1.sample_weight.values
# sample_weight2 = background_df2.sample_weight.values
# sample_weight3 = background_df3.sample_weight.values


# for col in cols_events:

#     fig1,axes = plt.subplots(ncols=2)
#     ax1,ax2 = axes.flatten()

#     ax1.hist(signal_df[col], bins=100, alpha=0.5, label='ttH125')
#     ax1.hist(background_df1[col], bins=100, alpha=0.5, label='TTTo2L2Nu')
#     ax1.hist(background_df2[col], bins=100, alpha=0.5, label='TTToHadronic')
#     ax1.hist(background_df3[col], bins=100, alpha=0.5, label='TTToSemiLeptonic')
#     ax1.set_xlabel(col)
#     ax1.legend()   

#     ax2.hist(signal_df[col], bins=100, weights=sample_weight0, alpha=0.5, label='ttH125')
#     ax2.hist(background_df1[col], bins=100, weights=sample_weight1, alpha=0.5, label='TTTo2L2Nu')
#     ax2.hist(background_df2[col], bins=100, weights=sample_weight2, alpha=0.5, label='TTToHadronic')
#     ax2.hist(background_df3[col], bins=100, weights=sample_weight3, alpha=0.5, label='TTToSemiLeptonic')
#     ax2.set_xlabel(col)
#     ax2.legend()


    












