"""
This file contains Classes for loading data into Python.
"""

# Python libraries
import os
import pandas as pd

class DataLoader():
    def __init__(self, root):
        self.root = root
        self.hd5_filename = "df_ml_inputs.hd5"
        self.filepaths = []     # list of str
        self.data = []          # list of pandas.DataFrame
        
    def find_files(self):
        """
        Function to populate the self.filepaths list with all files containing 
        data.
        """
        for path, subdirs, files in os.walk(self.root):
            for name in files:
                if self.hd5_filename in name:
                    self.filepaths.append(os.path.join(path, name))
        
        if len(self.filepaths)==0:
            raise Exception("No files found in specified root folder")
                    
    def collect_data(self, data_to_collect, verbose=True):
        """
        Funcion to create a a list of pandas dataframes. Each dataframe contains
        the data for one of the entires specified by the data_to_collect list.
        
        Parameters
        ----------
        data_to_collect : list
            list of folder names to collect data from
            e.g. ['ggF125', 'VBF125', 'ZH125']
        """
        for filename in self.filepaths:
            for t in data_to_collect:
                if t in filename:
                    df = pd.read_hdf(filename)
                    self.data.append(df)
                    if verbose:
                        print(f"{filename.split('ml_postproc')[-1]:42}: {df.shape}")
