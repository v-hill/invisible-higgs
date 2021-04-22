# invisible-higgs
Repository for 2020/2021 Physics MSci project "Searches for new physics at the LHC using machine learning" .

# Structure
This repository is currently structured as follows.

    ├── Images
        └── ....
    ├── dice_scripts
        └── ....
    ├── src            
        ├── models   
        │   ├── combined_models.py
        │   ├── recurrent_models.py
        │   └── sequential_models.py
        ├──notebooks
            ├── binary_classifier_results.ipynb
            ├── dataset_exploration.ipynb
            └── neural_network_results.ipynb
        ├── tuning 
        │   ├── nn_events_hyperparameter_tuning.ipynb
        │   └── nn_multifeature_tuning.py
        ├── utilities   
            ├── data_analysis.py
            ├── data_loader.py
            ├── data_preprocessing.py
            └── plotlib.py
        ├── binary_classifier.py
        ├── binary_complete_nn.py
        ├── binary_event_nn.py
        ├── binary_jet_rnn.py
        ├── hist_plt.py
        ├── multi_classification_complete_network.py
        ├── multi_classification_nn.py
        ├── multi_classification_rnn.py
        └── preprocess_data.py
    
<img src="./Images/code_structure_3.png" alt="drawing" width="600"/>
