"""
This file contains a neural network which combines the outputs of the 
feedforward neural network (for the events data) and the recurrent neural 
network (for the jet data).
"""

# ---------------------------------- Imports ----------------------------------

# Code from other files in the repo
import classifier
import utilities.plotlib as plotlib
from utilities.data_analysis import ModelResultsMulti

# ------------------------------------ Main -----------------------------------

if __name__ == "__main__":
    DIR = 'data_binary_classifier\\'
    
    args_model = {'model_type' : 'binary_classifier',
                  'model_architecture' : 'CombinedNN',
                  'ffn_layer_1_neurons' : 16,
                  'ffn_layer_2_neurons' : 8,
                  'rnn_layer_1_neurons' : 64,
                  'rnn_layer_2_neurons' : 8,
                  'final_layer_neurons' : 8,
                  'output_shape' : 1,
                  'loss_function' : 'binary_crossentropy',
                  'learning_rate' : 0,
                  'batch_size' : 64,
                  'epochs' : 16,
                  'model' : 'base'}
    
    num_runs = 5
    dataset_sample = 0.25
    
    all_results = ModelResultsMulti()
    neural_net = classifier.CombinedNN(args_model)  
    neural_net.load_data(DIR)
    neural_net.load_event_data(DIR)
    neural_net.load_jet_data(DIR)
    
    for i in range(num_runs):
        model_result = classifier.run(i, neural_net, args_model, dataset_sample)
        all_results.add_result(model_result, args_model)
    
    df_all_results = all_results.return_results()
    all_results.save('binary_combined_nn.pkl')
