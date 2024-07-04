"""
Top-level call to determine FIG for DOM dataset paper revision
"""

import FCNets.BitumenWorkflows as BWF
import torch.nn as nn

dom_file_directory = 'TestDOMCSV'
open_param_dict = {'num_layers': 3,
                   'strategy': 'exp',
                   'batch_norm': True,
                   'dropout': 0.0,
                   'softmax': False,
                   'activation': nn.ReLU()}

possible_formula_list = [(-1,-2,0,1,0)]
holdout_list = []
csv_output_name = 'Test_wCV_results.csv'
number_repeat = 1
val_split = 0.1
test_split = 0.1
training_epochs = 100
batch_size = 750
learning_rate = 0.001
lr_patience = 10
es_patience = 25
cv_splits = 5
log_int = True

BWF.measure_fig_levels_dom(dom_file_directory,
                           open_param_dict,
                           holdout_list,
                           possible_formula_list,
                           number_repeat,
                           csv_output_name,
                           val_split,
                           test_split,
                           training_epochs,
                           batch_size,
                           learning_rate,
                           lr_patience,
                           es_patience,
                           cv_splits,
                           log_int)