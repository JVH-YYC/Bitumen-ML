"""
Top-level call to determine FIG for DOM dataset paper revision
"""

import FCNets.BitumenWorkflows as BWF
import torch.nn as nn

dom_file_directory = 'DOM10'
open_param_dict = {'num_layers': 3,
                   'strategy': 'exp',
                   'batch_norm': True,
                   'dropout': 0.0,
                   'softmax': False,
                   'activation': nn.ReLU()}

possible_formula_list = [(50,50,50,50,50),
                         (6,10,0,5,0),
                         (6,8,0,4,0),
                         (1,2,0,1,0),
                         (8,11,1,2,0),
                         (1,2,0,0,0),
                         (2,2,0,1,0),
                         (2,4,0,1,0),
                         (3,6,0,1,0),
                         (8,8,0,0,0),
                         (10,8,0,4,0),
                         (5,8,0,0,0),
                         (1,0,0,2,0),
                         (0,2,0,1,0),
                         (0,3,1,0,0),
                         (0,2,0,0,1),
                         (0,2,0,0,0),
                         (0,-2,0,1,0),
                         (0,-2,0,2,0),
                         (0,0,0,1,0),
                         (0,1,1,0,0),
                         (0,0,0,0,1),
                         (9,8,0,2,0),
                         (6,4,0,1,0),
                         (9,10,0,3,0),
                         (7,2,0,3,0),
                         (6,4,0,3,0),
                         (7,6,0,2,0),
                         (8,8,0,3,0),
                         (0,0,0,3,1),
                         (0,-2,0,2,1)]
holdout_list = []
csv_output_name = 'DOM10_FIG_results.csv'
number_repeat = 1
val_split = 0.1
test_split = 0.1
training_epochs = 300
batch_size = 750
learning_rate = 0.001
lr_patience = 20
es_patience = 40
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