"""
This script is used to process our data and generate pre-processed data files,
which can be fed directly in the neural network models.
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
from utilities.data_loader import  DataLoader
from utilities.data_preprocessing import DataProcessing
from utilities.data_preprocessing import LabelMaker
from utilities.data_preprocessing import WeightMaker

# Python libraries
import os
import copy
import pickle
from datetime import datetime

# ---------------------------- Function definitions ---------------------------

def write_args_to_txt(args):
    """
    Write the programme setup parameters to a log file.

    Parameters
    ----------
    args : dict
        Arguments used during the data preprocessing to generate the 
        preprocessed data.
    """
    output_file = open(args['dir_output']+'data_preprocessing_arguments.txt', 'w')
    output_file.write('The following paramters were used to generate this '
                      'dataset: \n')
    for k, v in args.items():
      # write line to output file
      output_file.write(f"   '{k}'\t{v}")
      output_file.write('\n')
    output_file.close()
    
# ------------------------------ Programme setup ------------------------------

# Use 'input_dataset'='old' for old dataset, 'input_dataset'='new' for new dataset
# For 'weight_col', choice of 'weight_nominal' or 'xs_weight'
args = {'dir_root' : 'C:\\Users\\user\\Documents\\Fifth Year\\ml_postproc\\',
        'input_dataset' : 'new',
        'output_datasets' : ['binary_classifier', 'multi_classifier', 'multisignal_classifier'],
        'chosen_output' : 'multisignal_classifier',
        'set_diJet_mass_nan_to_zero' : True,
        'weight_col' : 'xs_weight',
        'timestamp' : datetime.today().strftime('%Y-%m-%d %H:%M:%S')}

args['save_folder'] = 'data_' + args['chosen_output']
args['dir_output'] = args['save_folder'] + '\\'

if args['chosen_output']=='binary_classifier' or args['chosen_output']=='multi_classifier':
    args['data_to_collect'] = ['ttH125', 
                               'TTTo2L2Nu', 
                               'TTToHadronic', 
                               'TTToSemiLeptonic']
else:
    args['data_to_collect'] = ['ttH125',
                               'TTTo2L2Nu', 
                               'TTToHadronic', 
                               'TTToSemiLeptonic',
                               'WminusH125',
                               'WplusH125',
                               'WJetsToLNu',
                               'ZJetsToNuNu',
                               'ZH125',
                               'VBF125',
                               'ggF125']

# -------------------------------- Load in data -------------------------------

loader = DataLoader(args['dir_root'])
loader.find_files()
loader.collect_data(args['data_to_collect'])
data = DataProcessing(loader)

# ---------------------------- Create save folder -----------------------------

if not os.path.exists(args['save_folder']):
    os.makedirs(args['save_folder'])

# --------------------------- Remove unwanted data ----------------------------

args['cols_to_ignore_events'] = ['entry', 'weight_nominal', 
                                 'xs_weight', 'hashed_filename', 
                                 'BiasedDPhi', 'InputMet_InputJet_mindPhi', 
                                 'InputMet_phi', 'InputMet_pt', 'MHT_phi']
args['cols_to_ignore_jets'] = ['cleanJetMask']

cols_events = data.get_event_columns(args['cols_to_ignore_events'])
cols_jets = data.get_jet_columns(args['cols_to_ignore_jets'])

# Clean DiJet_mass values
if args['set_diJet_mass_nan_to_zero']:
    data.set_nan_to_zero('DiJet_mass')
else:
    data.remove_nan('DiJet_mass')

# Removes all events with less than two jets
data.data = data.data[data.data.ncleanedJet > 1]

# ------------------------------ Label_encoding -------------------------------

# Store the original dataset labels, before label encoding
data.data.insert(loc=0, column='raw_dataset', value=data.data['dataset'])

if args['chosen_output']=='binary_classifier':
    signal_list = ['ttH125']
    data.label_signal_noise(signal_list)
    event_labels, encoding_dict = LabelMaker.label_encoding(data.return_dataset_labels())
    data.set_dataset_labels(event_labels, onehot=False)
    df_labels = data.data[['raw_dataset', 'label_encoding', 'dataset']]
    
elif args['chosen_output']=='multi_classifier':
    event_labels, encoding_dict = LabelMaker.onehot_encoding(data.return_dataset_labels())
    data.set_dataset_labels(event_labels, onehot=True)
    df_labels = data.data[['raw_dataset', 'dataset'] + list(event_labels.columns)]
    
elif args['chosen_output']=='multisignal_classifier':
    data_dict = {'ttH125' : 'tth',
                 'TTTo2L2Nu' : 'ttbar',
                 'TTToHadronic' : 'ttbar',
                 'TTToSemiLeptonic' : 'ttbar',
                 'WminusH125' : 'VH',
                 'WplusH125' : 'VH',
                 'ZH125': 'VH',
                 'VBF125' : 'VBF',
                 'ggF125' : 'ggF'}
    
    all_datset_vals = data.data['dataset'].unique()
    for dataset in all_datset_vals:
        if 'WJetsToLNu' in dataset:
            data_dict[dataset] = 'wjets'
        elif 'ZJetsToNuNu' in dataset:
            data_dict[dataset] = 'zjets'
        elif 'QCD_HT' in dataset:
            data_dict[dataset] = 'QCD'
            
    data.label_signal_noise_multi(data_dict)
    event_labels, encoding_dict = LabelMaker.onehot_encoding(data.return_dataset_labels())
    data.set_dataset_labels(event_labels, onehot=True)
    df_labels = data.data[['raw_dataset', 'dataset'] + list(event_labels.columns)]
    
else:
    raise Exception('Invalid choice of chosen_output argument')

# -------------------------------- Data weights -------------------------------

sample_weight = WeightMaker.weight_nominal_sample_weights(data, weight_col=args['weight_col'])
data.data.insert(loc=len(data.data.columns), column='sample_weight', value=sample_weight)

if args['input_dataset'] == 'new':
    df_weights = data.data[['weight_nominal','xs_weight', 'sample_weight']]
elif args['input_dataset'] == 'old':
    df_weights = data.data[['weight_nominal', 'sample_weight']]

# ------------------------------ Normalise data -------------------------------

# Make seperate copies of the dataset
df_event_data = copy.deepcopy(data)
df_jet_data = copy.deepcopy(data)

# Normalise event data
cols_to_log = ['HT', 'MHT_pt', 'MetNoLep_pt']
df_event_data.nat_log_columns(cols_to_log)
df_event_data.normalise_columns(cols_events, span=(0, 1))

# Normalise jet data
df_jet_data = df_jet_data.normalise_jet_columns(cols_jets, span=(0, 1))

# Select only the event columns from the data
df_event_data.filter_data(cols_events)
df_event_data = df_event_data.data

# -------------------------------- Data saving --------------------------------

print('saving processed data')
pickle.dump(encoding_dict, open(args['dir_output']+'encoding_dict.pkl', 'wb' ))
pickle.dump(args, open(args['dir_output']+'data_preprocessing_arguments.pkl', 'wb' ))
write_args_to_txt(args)

# data.data.to_pickle(args['dir_output']+'df_all_data.pkl')
df_labels.to_pickle(args['dir_output']+'df_labels.pkl')
df_weights.to_pickle(args['dir_output']+'df_weights.pkl')
df_jet_data.to_pickle(args['dir_output']+'preprocessed_jet_data.pkl')
df_event_data.to_pickle(args['dir_output']+'preprocessed_event_data.pkl')
