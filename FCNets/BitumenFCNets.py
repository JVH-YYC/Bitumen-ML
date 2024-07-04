#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:18:28 2021

@author: jvh

Scripts for creating/training/testing fully-connected neural networks
that act on dictionary sets of molecular formulae (expressed as tuples)
Scripts for calculating Kennard-Stone ranking, generating enumerated sets, etc.
will be present in a separate file

Narrow does not include -Cl, or -Na, which are not observed in CFGC data set

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import time
from collections import OrderedDict
from pathlib import Path
from torch.utils.data import Dataset
import DataTools.BitumenCreateUseDataset as CUD
import DataTools.BitumenCSVtoDict as CTD
import numpy as np
import os
import pickle
import random

def ceildiv(a, b):
    """
    A function that does the equivalent of ceiling integer division in the safest way

    Parameters
    ----------
    a : numerator
        Self-descriptive
    b : divisior
        Self-descriptive

    Returns
    -------
    An integer (ceiling division)

    """

    return -(-a // b)

class FCLayer(nn.Module):
    def __init__(self,
                 nodes_in,
                 nodes_out,
                 batch_norm,
                 softmax,
                 activation,
                 dropout):
        """
        A quick function for creating new layers of fully connected neural networks

        Parameters
        ----------
        nodes_in : integer
            number of fc nodes leading into layer
        nodes_out : integer
            number of fc nodes exiting layer
        batch_norm : Boolean
            If 'true', add batch normalization, if 'false', exclude batchnorm
        activation : nn.Relu(), nn.tanh(), etc. or None
        dropout : float
            A number between 0-1 indicating how much dropout to apply to all layers
        
        Returns
        -------
        A pytorch fc layer for assembling into a network

        """
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(nodes_in, nodes_out)
        self.batch_norm = batch_norm
        self.softmax = softmax
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        
        if batch_norm == True:
            self.bn = nn.BatchNorm1d(nodes_out)
        
        if softmax == True:
            self.sm = nn.Softmax(dim=0)
        
    def forward(self, in_tensor):
        """
        Linear layer forward pass

        Parameters
        ----------
        in_tensor : Pytorch tensor
            1-dimensional Pytorch tensor

        Returns
        -------
        Tensor processed according to FCLayer parameters

        """
        out_tensor = self.dropout(self.fc(in_tensor))
        if self.batch_norm == True:
            out_tensor = self.bn(out_tensor)
        if self.activation is not None:
            out_tensor = self.activation(out_tensor)
        if self.softmax == True:
            out_tensor = self.sm(out_tensor)
        return out_tensor

class FIGNet(nn.Module):
    def __init__(self,
                 num_layers,
                 width_start,
                 strategy,
                 batch_norm,
                 softmax,
                 activation,
                 dropout_amt):
        super(FIGNet, self).__init__()
        self.num_layers = num_layers
        self.width_start = width_start
        self.strategy = strategy
        self.batch_norm = batch_norm
        self.softmax = softmax
        self.activation = activation
        self.dropout_amt = dropout_amt
        
        self.fc_layers = self.create_fig_network()
    
    def forward(self,
                in_tensor):
        out_tensor = self.fc_layers(in_tensor)
        return out_tensor
    
    def create_fig_network(self):
        
        layers = []
        current_width = self.width_start
        
        for new_layer in range(self.num_layers - 1):    
            if self.strategy == 'str':
                layers.append(FCLayer(current_width,
                                      current_width,
                                      self.batch_norm,
                                      False,
                                      self.activation,
                                      self.dropout_amt))
            if self.strategy == 'exp':
                if new_layer == 0:
                    layers.append(FCLayer(current_width,
                                          (current_width * 2),
                                          self.batch_norm,
                                          False,
                                          self.activation,
                                          self.dropout_amt))
                    current_width = current_width * 2
                else:
                    layers.append(FCLayer(current_width,
                                          ceildiv(current_width, 2),
                                          self.batch_norm,
                                          False,
                                          self.activation,
                                          self.dropout_amt))
                    current_width = ceildiv(current_width, 2)
            if self.strategy == 'nar':
                layers.append(FCLayer(current_width,
                                      ceildiv(current_width, 2),
                                      self.batch_norm,
                                      False,
                                      self.activation,
                                      self.dropout_amt))
                current_width = ceildiv(current_width, 2)
                                
        #Append final layer
        layers.append(FCLayer(current_width,
                              1,
                              False,
                              self.softmax,
                              None,
                              self.dropout_amt))
        
        return nn.Sequential(*layers)

class FCExtNetTIS(nn.Module):
    def __init__(self,
                 layer_size_list,
                 starting_width,
                 batch_norm,
                 activation,
                 dropout_amt):
        super(FCExtNetTIS, self).__init__()
        self.layer_size_list = layer_size_list
        self.starting_width = starting_width
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout_amt = dropout_amt
        
        self.fc_layers = self.create_extract_net()
    
    def forward(self,
                ext_tensor):
        out_tensor = self.fc_layers(ext_tensor)
        
        return out_tensor
    
    def create_extract_net(self):
        
        layers = []
        current_width = self.starting_width
        
        for new_layer in self.layer_size_list:
            layers.append(FCLayer(current_width,
                          new_layer,
                          self.batch_norm,
                          None,
                          self.activation,
                          self.dropout_amt))
            current_width = new_layer
        
        #Append final layer
        layers.append(FCLayer(current_width,
                              1,
                              False,
                              None,
                              None,
                              0.0))
                
        return nn.Sequential(*layers)
    
class FCExtNet(nn.Module):
    def __init__(self,
                 num_layers,
                 width_start,
                 strategy,
                 batch_norm,
                 softmax,
                 activation,
                 dropout_amt):
        super(FCExtNet, self).__init__()
        self.num_layers = num_layers
        self.width_start = width_start
        self.strategy = strategy
        self.batch_norm = batch_norm
        self.softmax = softmax
        self.activation = activation
        self.dropout_amt = dropout_amt
        
        self.fc_layers = self.create_extract_net()
    
    def forward(self,
                sm_tensor,
                cond_tensor):
        combined_tens = torch.cat((sm_tensor, cond_tensor), dim=1)
        out_tensor = self.fc_layers(combined_tens)
        
        return out_tensor
    
    def create_extract_net(self):
        
        layers = []
        current_width = self.width_start
        
        for new_layer in range(self.num_layers - 1):    
            if self.strategy == 'str':
                layers.append(FCLayer(current_width,
                                      current_width,
                                      self.batch_norm,
                                      False,
                                      self.activation,
                                      self.dropout_amt))
            if self.strategy == 'exp':
                if new_layer == 0:
                    layers.append(FCLayer(current_width,
                                          (current_width * 2),
                                          self.batch_norm,
                                          False,
                                          self.activation,
                                          self.dropout_amt))
                    current_width = current_width * 2
                else:
                    layers.append(FCLayer(current_width,
                                          ceildiv(current_width, 2),
                                          self.batch_norm,
                                          False,
                                          self.activation,
                                          self.dropout_amt))
                    current_width = ceildiv(current_width, 2)
            if self.strategy == 'nar':
                layers.append(FCLayer(current_width,
                                      ceildiv(current_width, 2),
                                      self.batch_norm,
                                      False,
                                      self.activation,
                                      self.dropout_amt))
                current_width = ceildiv(current_width, 2)
                                
        #Append final layer
        layers.append(FCLayer(current_width,
                              1,
                              False,
                              self.softmax,
                              None,
                              self.dropout_amt))
        
        return nn.Sequential(*layers)

def load_fig_network(trained_net_directory,
                     trained_net_name,
                     net_param_dict,
                     eval_or_train_data):
    """
    Loads a pre-trained network of the shape/type according to net parameter dict. This is
    for formula information gain networks, so uses the appropriate network constructor.

    Parameters
    ----------
    trained_net_directory : string
        Name of the (sub) directory containing the neural network state dictionary
    trained_net_name : string
        Name of the pytorch state_dict() file
    net_param_dict : dictionary
        A dictionary with the following keys: 'activation', 'batch_norm', 'strategy'
        'num_layers', 'dropout'. The entries under these keys defines the
        shape and function of the fully connected network.
        Starting width is calculated.
    eval_or_train_data : LabelledMSSet object
        A dataset object for either training or evaluation, used to
        set the proper starting width for the neural network

    Returns
    -------
    A fully connected network according to the paramter dictionary, loaded with
    the trained weights according to the state dictionary

    """
    
    starting_net_width = 5 + len(eval_or_train_data.current_test_formula)
    
    new_network = FIGNet(net_param_dict['num_layers'],
                         starting_net_width,
                         net_param_dict['strategy'],
                         net_param_dict['batch_norm'],
                         net_param_dict['activation'],
                         net_param_dict['dropout'])
    
    dictionary_path = Path('.', trained_net_directory)
    train_dict = torch.load(dictionary_path / trained_net_name, map_location=torch.device('cpu'))
    
    new_train_dict = OrderedDict()
    for name, value in train_dict.items():
        if name[:6] == 'module':
            new_name = name[7:]
            new_train_dict[new_name] = value
        else:
            new_train_dict[name] = value
    
    new_network.load_state_dict(new_train_dict)
    
    return new_network

def load_extraction_network(trained_net_directory,
                            trained_net_name,
                            net_param_dict,
                            eval_or_train_data):
    """
    Loads a pre-trained network of the shape/type according to net parameter dict. This is
    for prior (before TIS) NN approach, so uses the appropriate network constructor.

    Parameters
    ----------
    trained_net_directory : string
        Name of the (sub) directory containing the neural network state dictionary
    trained_net_name : string
        Name of the pytorch state_dict() file
    net_param_dict : dictionary
        A dictionary with the following keys: 'activation', 'batch_norm', 'strategy'
        'num_layers', 'dropout'. The entries under these keys defines the
        shape and function of the fully connected network.
        Starting width is calculated.
    eval_or_train_data : LabelledMSSet object
        A dataset object for either training or evaluation, used to
        set the proper starting width for the neural network

    Returns
    -------
    A fully connected network according to the paramter dictionary, loaded with
    the trained weights according to the state dictionary

    """
    starting_net_width = 12 + len(eval_or_train_data.current_test_formula)
    print('Starting net width:', starting_net_width)

    
    new_network = FCExtNet(net_param_dict['num_layers'],
                           starting_net_width,
                           net_param_dict['strategy'],
                           net_param_dict['batch_norm'],
                           net_param_dict['activation'],
                           net_param_dict['dropout'])
    
    dictionary_path = Path('.', trained_net_directory)
    train_dict = torch.load(dictionary_path / trained_net_name, map_location=torch.device('cpu'))
    
    new_train_dict = OrderedDict()
    for name, value in train_dict.items():
        if name[:6] == 'module':
            new_name = name[7:]
            new_train_dict[new_name] = value
        else:
            new_train_dict[name] = value
    
    new_network.load_state_dict(new_train_dict)
    
    return new_network

def load_extraction_tis_network(trained_net_directory,
                                trained_net_name,
                                net_param_dict,
                                locked_formula_list):
    """
    Loads a pre-trained network of the shape/type according to net parameter dict. This is
    for the 'total ion set' approach, so uses the appropriate network constructor.

    Parameters
    ----------
    trained_net_directory : string
        Name of the (sub) directory containing the neural network state dictionary
    trained_net_name : string
        Name of the pytorch state_dict() file
    net_param_dict : dictionary
        A dictionary with the following keys: 'activation', 'batch_norm', 'strategy'
        'num_layers', 'dropout'. The entries under these keys defines the
        shape and function of the fully connected network.
        Starting width is calculated.
    eval_or_train_data : LabelledMSSet object
        A dataset object for either training or evaluation, used to
        set the proper starting width for the neural network

    Returns
    -------
    A fully connected network according to the paramter dictionary, loaded with
    the trained weights according to the state dictionary

    """
    starting_net_width = 12 + len(locked_formula_list)

    
    new_network = FCExtNetTIS(net_param_dict['layer_size_list'],
                              starting_net_width,
                              net_param_dict['batch_norm'],
                              nn.ReLU(),
                              net_param_dict['dropout_amt'])
    
    dictionary_path = Path('.', trained_net_directory)
    train_dict = torch.load(dictionary_path / trained_net_name, map_location=torch.device('cpu'))
    
    new_train_dict = OrderedDict()
    for name, value in train_dict.items():
        if name[:6] == 'module':
            new_name = name[7:]
            new_train_dict[new_name] = value
        else:
            new_train_dict[name] = value
    
    new_network.load_state_dict(new_train_dict)
    
    return new_network

class FIGTrainingData(Dataset):
    def __init__(self,
                 sm_file_directory,
                 ext_file_directory,
                 label_keys,
                 holdout_list,
                 locked_formula):
        """
        Creates Pytorch Dataset object for MS training based on relationship between domain-expertise
        identified MS formula adjustments.

        Parameters
        ----------
        sm_file_directory : string
            Name of directory containing starting materials .csv files
        ext_file_directory : string
            Name of directory containing .csv files for extracted materials
        label_keys : dictionary
            A dictionary that describes which file names are associated with which
            starting materials - used to label entries for future organization.
            Possible labels are 'L1', 'S2', etc. for extracted materials, and
            'L1_SM', 'S2_SM', etc. for starting materials themselves.
        holdout_list : list
            Name of MS .csv files (and, therefore, also .pkl files) that are held out
            from training (for testing/cross-validation)                    
        locked_formula : list
            A list of formula that have been locked in after being identified as significant. New formula
            masses to search are created by combining a possible_formula with locked_formula
        current_test_formula : list
            A list of formula that are being attempted in the current training iteration. Created below.
            
        Returns
        -------
        Creates training/validation data set (but not testing), with necessary .len and .getitem functions
        Getitem function will return values for identified formula peaks, and the starting material MS intensity
        always - but in cases of extraction, will also return the barcode for the extraction conditions,
        and the intensity of the target peak in the extracted fraction
        """

        self.sm_file_directory = sm_file_directory
        self.ext_file_directory = ext_file_directory
        self.label_keys = label_keys
        self.holdout_list = holdout_list
        
        if locked_formula is not None:
            self.locked_formula = locked_formula
            self.current_test_formula = locked_formula
        else:
            self.locked_formula = None
            self.current_test_formula = []
            
        self.training_dictionary = CTD.open_sum_training_dict(sm_file_directory,
                                                              ext_file_directory,
                                                              label_keys,
                                                              holdout_list)

        
        self.training_keys = CUD.enumerate_fig_dataset(self.training_dictionary)
        
    def __len__(self):
        return len(self.training_keys)
    
    def __getitem__(self, idx):
        top_level_key, file_label, sub_level_key = self.training_keys[idx]
        
        active_dictionary = self.training_dictionary[top_level_key]
        
        fig_example_tensor = CUD.create_fig_tensor(sub_level_key,
                                                   active_dictionary[1],
                                                   self.current_test_formula)

        example_target = np.float32(active_dictionary[1][sub_level_key])
                                
        return fig_example_tensor, example_target
                                     
    def set_test_formula(self,
                           formula_list):
        self.current_test_formula = formula_list
        return

    def return_curr_formula(self):
        return self.current_test_formula

class FIGTestData(Dataset):
    def __init__(self,
                 sm_file_directory,
                 ext_file_directory,
                 label_keys,
                 test_list,
                 locked_formula):
        """
        Creates Pytorch Dataset object for MS final testing based on relationship between domain-expertise
        identified MS formula.

        Parameters
        ----------
        sm_file_directory : string
            Name of directory containing starting materials .csv files
        ext_file_directory : string
            Name of directory containing .csv files for extracted materials
        label_keys : dictionary
            A dictionary that describes which file names are associated with which
            starting materials - used to label entries for future organization.
            Possible labels are 'L1', 'S2', etc. for extracted materials, and
            'L1_SM', 'S2_SM', etc. for starting materials themselves.
        test_list : list
            Name of MS .csv files that *will* be loaded for testing. Right now, do 1 at a time.                  
        locked_formula : list
            A list of formula that have been locked in after being identified as significant. New formula
            masses to search are created by combining a possible_formula with locked_formula
        current_test_formula : list
            A list of formula that are being attempted in the current training iteration. Created below.

        Returns
        -------
        Creates training/validation data set (but not testing), with necessary .len and .getitem functions
        Getitem function will return values for identified formula peaks, and the starting material MS intensity
        always - but in cases of extraction, will also return the barcode for the extraction conditions,
        and the intensity of the target peak in the extracted fraction
        """

        self.sm_file_directory = sm_file_directory
        self.ext_file_directory = ext_file_directory
        self.test_list = test_list
        self.label_keys = label_keys
        self.locked_formula = locked_formula
        
        self.test_dictionary = CTD.open_sum_test_dict(sm_file_directory,
                                                      ext_file_directory,
                                                      label_keys,
                                                      test_list)

        if locked_formula is not None:
            self.locked_formula = locked_formula
            self.current_test_formula = locked_formula
        else:
            self.locked_formula = None
            self.current_test_formula = []
        
        self.test_keys = CUD.enumerate_labelled_dataset(self.test_dictionary)
        
    def __len__(self):
        return len(self.test_keys)
    
    def __getitem__(self, idx):
        top_level_key, file_label, sub_level_key = self.test_keys[idx]
                
        #Detect starting material MS file in training dictionary
        top_level_key, file_label, sub_level_key = self.training_keys[idx]
        
        active_dictionary = self.training_dictionary[top_level_key]
        
        fig_example_tensor = CUD.create_fig_tensor(sub_level_key,
                                                   active_dictionary[1],
                                                   self.current_test_formula)

        example_target = active_dictionary[1][sub_level_key]
                                
        return fig_example_tensor, example_target
        
    def set_test_formula(self,
                           formula_list):
        self.current_test_formula = formula_list
        return

    def return_curr_formula(self):
        return self.current_test_formula

class DOMFIGDataset(Dataset):
    def __init__(self,
                 dom_file_directory,
                 holdout_list,
                 cv_splits=None,
                 log_int=False):
        """
        Creates Pytorch Dataset object for MS training based on relationship between domain-expertise
        identified MS formula adjustments. For dissolved organic matter dataset used in paper revisions.
        """
        self.dom_file_directory = dom_file_directory
        self.holdout_list = holdout_list
        self.cv_splits = cv_splits
        self.log_int = log_int

        self.dom_dictionary = CTD.create_full_dom_dict(dom_file_directory,
                                                    holdout_list)
        
        self.dom_keys = CUD.enumerate_dom_fig_dataset(self.dom_dictionary)

        self.locked_formula = None
        self.current_test_formula = []

        if self.cv_splits is not None:
            self.cv_splits = self.create_cv_splits(cv_splits)
        
    def create_cv_splits(self,
                            cv_splits):
        indices = list(range(len(self.dom_keys)))
        random.shuffle(indices)
        split_size = len(indices) // cv_splits
        splits = [indices[i * split_size: (i + 1) * split_size] for i in range(cv_splits)]

            # If there are remaining indices, add them to the last split
        if len(indices) % cv_splits != 0:
            splits[-1].extend(indices[cv_splits * split_size:])
    
        return splits
    
    def __len__(self):
        return len(self.dom_keys)
    
    def __getitem__(self, idx):
        top_level_key, sub_level_key = self.dom_keys[idx]
        
        active_dictionary = self.dom_dictionary[top_level_key]
        
        fig_example_tensor = CUD.create_dom_tensor(sub_level_key,
                                                   active_dictionary,
                                                   self.current_test_formula,
                                                   self.log_int)

        example_target = np.float32(active_dictionary[sub_level_key])
                                
        return fig_example_tensor, example_target

    def set_test_formula(self,
                           formula_list):
        self.current_test_formula = formula_list
        return

    def return_curr_formula(self):
        return self.current_test_formula

class BitumenExtTrainingData(Dataset):
    def __init__(self,
                 sm_file_directory,
                 ext_file_directory,
                 label_keys,
                 test_list,
                 locked_formula,
                 condition_dict,
                 output_name):
        """
        Creates Pytorch Dataset object for MS extraction training based on relationship between domain-expertise
        identified MS formula.

        Parameters
        ----------
        sm_file_directory : string
            Name of directory containing starting materials .csv files
        ext_file_directory : string
            Name of directory containing .csv files for extracted materials
        label_keys : dictionary
            A dictionary that describes which file names are associated with which
            starting materials - used to label entries for future organization.
            Possible labels are 'L1', 'S2', etc. for extracted materials, and
            'L1_SM', 'S2_SM', etc. for starting materials themselves.
        test_list : list
            Name of MS .csv files that *will* be held out for testing.                
        locked_formula : list
            A list of formula that have been locked in after being identified as significant. New formula
            masses to search are created by combining a possible_formula with locked_formula
        current_test_formula : list
            A list of formula that are being attempted in the current training iteration. Created below.
        condition_dict : dictionary
            A dictionary that relates a given .csv file name to the extraction conditions used to generate it.
            Provided in the top level function that calls FCSetNarrow for training/prediction
            Entries are of the format: [frac_toluene, frac_dcm, frac_iPrOH, frac_acetone, fract_HOAc, frac_NEt3]
        output_name : string
            A leading stamp that will be appended to .pkl outputs about training data ion composition
            
        Returns
        -------
        Creates training/validation data set (but not testing), with necessary .len and .getitem functions
        Getitem function will return values for identified formula peaks, and the starting material MS intensity
        always - but in cases of extraction, will also return the barcode for the extraction conditions,
        and the intensity of the target peak in the extracted fraction
        """

        self.sm_file_directory = sm_file_directory
        self.ext_file_directory = ext_file_directory
        self.test_list = test_list
        self.label_keys = label_keys
        self.condition_dict = condition_dict
        
        self.training_dictionary = CTD.open_sum_training_dict(sm_file_directory,
                                                              ext_file_directory,
                                                              label_keys,
                                                              test_list)
        
        self.training_keys, self.observed_ion_dictionary, self.total_ion_list = CUD.enumerate_extraction_dataset(self.training_dictionary,
                                                                                                                 output_name)
        
        if locked_formula is not None:
            self.current_test_formula = locked_formula
            self.locked_formula = locked_formula
        else:
            self.locked_formula = None
            self.current_test_formula = []

    def __len__(self):
        return len(self.training_keys)
    
    def __getitem__(self, idx):
        top_level_key, sm_file_label, sub_level_key, ion_type = self.training_keys[idx]
        
        ext_example_tensor = CUD.create_extraction_tensor(sm_file_label,
                                                          sub_level_key,
                                                          ion_type,
                                                          self.training_dictionary,
                                                          self.current_test_formula)
        
        if ion_type != 'dis':
            example_target = np.float32(self.training_dictionary[top_level_key][1][sub_level_key])
        else:
            example_target = np.float32(0.0)
        
        #Detect experimental example extraction conditions. Only ~17 conditions per SM, so inefficient
        #loop not currently a problem
        
        for extract_cond in self.condition_dict.keys():
            if extract_cond in top_level_key:
                example_extraction = self.condition_dict[extract_cond]
                condition_tensor = torch.FloatTensor(example_extraction)
                return ext_example_tensor, example_target, condition_tensor
        
        raise ValueError ('Extraction conditions not properly detected: training tensor not constructed')
        
    def set_test_formula(self,
                           formula_list):
        self.current_test_formula = formula_list
        return

    def return_curr_formula(self):
        return self.current_test_formula

class BitumenExtTISDataset(Dataset):
    def __init__(self,
                 sm_file_directory,
                 ext_file_directory,
                 label_keys,
                 test_list,
                 locked_formula,
                 condition_dict,
                 pickle_file,
                 output_name):
        """
        Creates Pytorch Dataset object for MS extraction training based on relationship between domain-expertise
        identified MS formula.
        For simplicity, training and test dictionaries are created at the outset and held in the same dataset object,
        with the training dictionary being accessed by the __getitem__ function. The test dictionary does not need
        to be accessed in training loops (obviously), so it doesn't need to be part of __getitem__.

        Parameters
        ----------
        sm_file_directory : string
            Name of directory containing starting materials .csv files
        ext_file_directory : string
            Name of directory containing .csv files for extracted materials
        label_keys : dictionary
            A dictionary that describes which file names are associated with which
            starting materials - used to label entries for future organization.
            Possible labels are 'L1', 'S2', etc. for extracted materials, and
            'L1_SM', 'S2_SM', etc. for starting materials themselves.
        test_list : list
            Name of MS .csv files that *will* be held out for testing.                
        locked_formula : list
            A list of formula that have been locked in after being identified as significant. New formula
            masses to search are created by combining a possible_formula with locked_formula
        current_test_formula : list
            A list of formula that are being attempted in the current training iteration. Created below.
        condition_dict : dictionary
            A dictionary that relates a given .csv file name to the extraction conditions used to generate it.
            Provided in the top level function that calls FCSetNarrow for training/prediction
            Entries are of the format: [frac_toluene, frac_dcm, frac_iPrOH, frac_acetone, fract_HOAc, frac_NEt3]
        output_name : string
            A leading stamp that will be appended to .pkl outputs about training data ion composition
            
        Returns
        -------
        Creates training/validation/internal test data set (true test via CV), with necessary .len and .getitem functions
        Getitem function will return values for formula peaks, and the starting material MS intensity
        always, the barcode for the extraction conditions, and the intensity of the target peak in the extracted fraction
        """

        self.sm_file_directory = sm_file_directory
        self.ext_file_directory = ext_file_directory
        self.test_list = test_list
        self.label_keys = label_keys
        self.condition_dict = condition_dict
        self.locked_formula = locked_formula
        
        self.training_dict = CTD.open_sum_training_dict(sm_file_directory,
                                                        ext_file_directory,
                                                        label_keys,
                                                        test_list)

        self.test_dict = CTD.open_sum_test_dict(sm_file_directory,
                                                ext_file_directory,
                                                label_keys,
                                                test_list)

        self.total_ion_set, self.normalization_tuple = CUD.create_total_ion_set(self.training_dict,
                                                                                output_name,
                                                                                pickle_file)
        
        self.training_keys, self.training_file_list = CUD.create_extraction_set_via_tis(self.training_dict,
                                                                                        self.total_ion_set)
        
        self.test_keys, self.test_targets, self.test_file_list = CUD.create_extraction_test_via_tis(self.training_dict,
                                                                                                    self.test_dict,
                                                                                                    self.total_ion_set)
        
        self.rectified_condition_dict = CUD.rectify_condition_dict(self.condition_dict,
                                                                   [self.training_file_list,
                                                                    self.test_file_list])
        
        self.getitem_list = CUD.expand_extraction_tis_dataset(self.training_dict,
                                                              self.training_keys,
                                                              self.rectified_condition_dict,
                                                              self.locked_formula,
                                                              self.normalization_tuple)
        
        self.test_getitem_list = CUD.expand_extraction_tis_dataset(self.test_dict,
                                                                   self.test_keys,
                                                                   self.rectified_condition_dict,
                                                                   self.locked_formula,
                                                                   self.normalization_tuple)
        
    def __len__(self):
        return len(self.training_keys)
    
    def __getitem__(self, idx):
        training_tensor = self.getitem_list[idx][4]
        training_target = self.getitem_list[idx][5]

        return training_tensor, training_target
        
    def set_test_formula(self,
                           formula_list):
        self.current_test_formula = formula_list
        return

    def return_curr_formula(self):
        return self.current_test_formula
    
    def report_dataset_metrics(self):
        print('Size of total ion set is:', len(self.total_ion_set))
        zero_ions = 0
        for target in self.test_targets:
            if target == 0:
                zero_ions += 1
        print('With', zero_ions, 'zero ion targets')
        return

class BitumenExtTestData(Dataset):
    def __init__(self,
                 sm_file_directory,
                 ext_file_directory,
                 label_keys,
                 test_list,
                 locked_formula,
                 condition_dict,
                 output_name):
        """
        Creates Pytorch Dataset object for MS extraction training based on relationship between domain-expertise
        identified MS formula.

        Parameters
        ----------
        sm_file_directory : string
            Name of directory containing starting materials .csv files
        ext_file_directory : string
            Name of directory containing .csv files for extracted materials
        label_keys : dictionary
            A dictionary that describes which file names are associated with which
            starting materials - used to label entries for future organization.
            Possible labels are 'L1', 'S2', etc. for extracted materials, and
            'L1_SM', 'S2_SM', etc. for starting materials themselves.
        test_list : list
            Name of MS .csv files that *will* be loaded for testing. Right now, do 1 at a time.                  
        locked_formula : list
            A list of formula that have been locked in after being identified as significant. New formula
            masses to search are created by combining a possible_formula with locked_formula
        current_test_formula : list
            A list of formula that are being attempted in the current training iteration. Created below.
        condition_dict : dictionary
            A dictionary that relates a given .csv file name to the extraction conditions used to generate it.
            Provided in the top level function that calls FCSetNarrow for training/prediction
            Entries are of the format: [frac_toluene, frac_dcm, frac_iPrOH, frac_acetone, fract_HOAc, frac_NEt3]
        output_name : string
            A leading stamp that will be appended to .pkl outputs about training data ion composition
            
        Returns
        -------
        Creates training/validation data set (but not testing), with necessary .len and .getitem functions
        Getitem function will return values for identified formula peaks, and the starting material MS intensity
        always - but in cases of extraction, will also return the barcode for the extraction conditions,
        and the intensity of the target peak in the extracted fraction
        """

        self.sm_file_directory = sm_file_directory
        self.ext_file_directory = ext_file_directory
        self.test_list = test_list
        self.label_keys = label_keys
        self.condition_dict = condition_dict
        
        self.test_dictionary = CTD.open_sum_test_dict(sm_file_directory,
                                                      ext_file_directory,
                                                      label_keys,
                                                      test_list)
        
        self.test_keys, self.observed_ion_dictionary, self.total_ion_list = CUD.enumerate_extraction_dataset(self.test_dictionary,
                                                                                                             output_name)
        
        if locked_formula is not None:
            self.current_test_formula = locked_formula
            self.locked_formula = locked_formula
        else:
            self.locked_formula = None
            self.current_test_formula = []

    def __len__(self):
        return len(self.test_keys)
    
    def __getitem__(self, idx):
        top_level_key, sm_file_label, sub_level_key, ion_type = self.test_keys[idx]
                
        
        ext_example_tensor = CUD.create_extraction_tensor(sm_file_label,
                                                          sub_level_key,
                                                          ion_type,
                                                          self.test_dictionary,
                                                          self.current_test_formula)
        
        example_target = np.float32(self.test_dictionary[top_level_key][1][sub_level_key])
        
        #Detect experimental example extraction conditions. Only ~17 conditions per SM, so inefficient
        #loop not currently a problem
        
        for extract_cond in self.condition_dict.keys():
            if extract_cond in top_level_key:
                example_extraction = self.condition_dict[extract_cond]
                condition_tensor = torch.FloatTensor(example_extraction)
                return ext_example_tensor, example_target, condition_tensor
        
        raise ValueError ('Extraction conditions not properly detected: training tensor not constructed')
        
    def set_test_formula(self,
                           formula_list):
        self.current_test_formula = formula_list
        return

    def return_curr_formula(self):
        return self.current_test_formula
    
def val_train_indices(dataset,
                      val_split,
                      shuffle=True):
    """
    Function to shuffle dataset and split into training/validation. Test set has
    already been removed in above functions that generate permutations.

    Parameters
    ----------
    dataset : Pytorch Dataset object
        Dataset generated in MSSetTraining
    val_split : float
        Number between 0-1 that represents amount of examples held for initial validation
    shuffle : Boolean, optional
        Whether or not to shuffle dataset before splitting. The default is True.

    Returns
    -------
    Two lists, one with training indices and one with validation indices

    """                    
    
    set_length = dataset.__len__()
    index_values = list(range(set_length))
    cutoff = int(np.floor(val_split * set_length))
    
    if shuffle == True:
        np.random.shuffle(index_values)
    
    train_index, val_index = index_values[cutoff:], index_values[:cutoff]
    
    return train_index, val_index

def test_train_val_split(dataset,
                         val_split,
                         test_split,
                         shuffle=True):
    """
    Function to shuffle dataset and split into training/testing/validation. For cases
    where test set is not split out during dataset generation (depreciated).

    Parameters
    ----------
    dataset : Pytorch Dataset object
        Dataset generated in formulaMSSet.
    val_split : float
        Fraction of the dataset that will be used for validation
    test_split : float
        Fraction of the dataset that will be used for testing
    shuffle : Boolean, optional
        Whether or not to shuffle dataset before splitting. The default is True.

    Returns
    -------
    Three lists, one each with training, validation, and testing indices.

    """
    set_length = dataset.__len__()
    index_values = list(range(set_length))
    cutoff_1 = int(np.floor((val_split * set_length)))
    
    if test_split is not None:
        cutoff_2 = int(np.floor(((val_split + test_split) * set_length)))
    else:
        cutoff_2 = cutoff_1
        
    if shuffle == True:
        np.random.shuffle(index_values)
        
    train_index = index_values[cutoff_2:]
    val_index = index_values[:cutoff_1]

    if test_split is not None:
        test_index = index_values[cutoff_1:cutoff_2]
    else:
        test_index = None
    
    return train_index, val_index, test_index

def loss_and_optim(network,
                   learning_rate,
                   lr_patience,
                   dom_dataset):
    """
    Creates Pytorch loss function and optimizer

    Parameters
    ----------
    network : FCNet object
        FC network generated as described above
    learning_rate : float
        Learning rate for training, typically around 0.0001 for adam optimizer
    lr_patience : integer
        Number of epoch with no improvement that are allowed before learning rate is updated.
    dom_dataset : boolean
        A check is performed to see if this is a DOM dataset, allows for using
        different parameters without changing the function
        
    Returns
    -------
    Loss function and optimizer.

    """
    if dom_dataset == False:
        loss_function = nn.MSELoss()

        optimize = optim.Adam(network.parameters(),
                            lr=learning_rate)
        
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimize,
                                                        mode='min',
                                                        factor=0.1,
                                                        patience=lr_patience,
                                                        threshold=0.0001,
                                                        min_lr=(learning_rate / 10001),
                                                        verbose=True)
    else:
        loss_function = nn.SmoothL1Loss(beta=1.0, reduction='mean')

        optimize = optim.Adam(network.parameters(),
                            lr=learning_rate)
        
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimize,
                                                        mode='min',
                                                        factor=0.1,
                                                        patience=lr_patience,
                                                        threshold=0.0001,
                                                        min_lr=(learning_rate / 10001),
                                                        verbose=True)
    return loss_function, optimize, lr_sched

def multi_loss_and_optim(network,
                      learning_rate,
                      lr_patience):
    """
    Creates Pytorch loss function and optimizer

    Parameters
    ----------
    network : FCNet object
        FC network generated as described above
    learning_rate : float
        Learning rate for training, typically around 0.0001 for adam optimizer
    lr_patience : integer
        Number of epoch with no improvement that are allowed before learning rate is updated.

    Returns
    -------
    Loss function and optimizer.

    """

    loss_function = nn.MSELoss()

    optimize = optim.Adam(network.parameters(),
                          lr=learning_rate)
    
    lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimize,
                                                    mode='min',
                                                    factor=0.1,
                                                    patience=lr_patience,
                                                    threshold=0.0001,
                                                    min_lr=(learning_rate / 10001),
                                                    verbose=True)
    
    return loss_function, optimize, lr_sched
        
def update_formula(training_dataset,
                   update_formula,
                   locked_formula):
    """
    A function that takes a training_dataset, and updates its current_formula
    list to include all locked_formula plus and minus the update_formula, and avoids
    duplicate formula, as well as (0,0,0,0,0), which would be cheating - it would
    allow the answer to be looked up directly.

    Parameters
    ----------
    training_dataset : formulaMSSet class object
        Object that contains all training data, and pytorch itemgetter that will return
        tensors with desired MS formula information
    update_formula : tuple
        A molecular formula that will be added and subtracted to each locked_formula
    locked_formula : list or None
        A list of already identified valuable mass formula. If none have yet been
        identified, use None

    Returns
    -------
    None, but updates the training_dataset in place

    """
    neg_frag = tuple([-1*x for x in update_formula])
    
    if locked_formula is None:
        training_dataset.set_test_formula([update_formula, neg_frag])
        return
    
    new_frag_list = list(locked_formula)
    
    if update_formula not in new_frag_list:
        new_frag_list.append(update_formula)
    
    if neg_frag not in new_frag_list:
        new_frag_list.append(neg_frag)

    update_array = np.asarray(update_formula)
    
    for known_formula in locked_formula:
        known_array = np.asarray(known_formula)
        pos_array = known_array + update_array
        pos_tuple = tuple(pos_array)
        if pos_tuple != (0, 0, 0, 0, 0):
            new_frag_list.append(pos_tuple)
        
        neg_array = known_array - update_array
        neg_tuple = tuple(neg_array)
        if neg_tuple != (0, 0, 0, 0, 0):
            new_frag_list.append(neg_tuple)
    
    #Remove duplicates via set
    final_frag_list = list(set(new_frag_list))
    
    training_dataset.set_test_formula(final_frag_list)
    return

# Old training loops below here. Updated/finalized ML workflows are not found in
# 'BitumenWorkflows.py'

# def train_network_end2end(network,
#                           dataset,
#                           val_split,
#                           test_split,
#                           training_epochs,
#                           batch_size,
#                           learning_rate,
#                           lr_patience,
#                           es_patience):
#     """
#     Training loop for fully connected bitumen MS data processing
#     Returns validation loss, state dictionary, and formula list
#     After full training loop, the state_dict and formula list are saved/pickled
#     This is for a single set of formula - formula are determined in outside loop

#     Parameters
#     ----------
#     network : FCExtNet as above
#         Fully connected neural network for training
#     dataset : LabelledMSSet as above
#         Dataset containing open-ended MS training values with starting material labels attached
#     possible_formula : list
#         A list of reasonable formula that have been identified as possibilities and are actively
#         searched for when building fully-connected networks. Each entry is a tuple
#         of the standard type (#C, #H, #N, #O, #S)
#     val_split : float
#         Number betwee 0-1, fraction of examples to be used in validation loop. Final test set
#         is held out by putting the .csv and .pkl files in separate folders from training/validation data
#     training_epochs : integer
#         Number of epochs of training to attempt
#     batch_size : integer
#         Batch size for training (and validation)
#     learning_rate : float
#         Initial learning rate
#     lr_patience : integer
#         Number of epochs to wait for improvement before reducing learning rate
#     es_patience : integer
#         Number of epochs to wait for improvement before early stopping

#     Returns
#     -------
#     The best validation loss, state dictionary, and MS formula being used

#     """        
    
#     train_index, val_index, test_index = test_train_val_split(dataset,
#                                                               val_split,
#                                                               test_split)

#     train_sample = SubsetRandomSampler(train_index)
#     val_sample = SubsetRandomSampler(val_index)
#     test_sample = SubsetRandomSampler(test_index)

#     train_loader = torch.utils.data.DataLoader(dataset,
#                                                batch_size=batch_size,
#                                                sampler=train_sample)

#     val_loader = torch.utils.data.DataLoader(dataset,
#                                              batch_size=batch_size,
#                                              sampler=val_sample)

#     test_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_size=batch_size,
#                                               sampler=test_sample)

#     num_batches = len(train_loader)
#     loss_func, optimize, lr_sched = loss_and_optim(network,
#                                                    learning_rate,
#                                                    lr_patience)

#     training_start = time.time()

#     best_val_loss = 1000.0
#     current_patience = es_patience
    
#     if torch.cuda.is_available() == True:
#         device = 'cuda'
#         if torch.cuda.device_count() > 1:
#             print('Multiple GPU Detected')
#             network = nn.DataParallel(network)
#     elif torch.backends.mps.is_available() == True:
#         device = 'mps'
#     else:
#         raise ValueError('No GPU available')
    
#     network.to(device)
    
#     for epoch in range(training_epochs):
#         current_loss = 0.0
#         epoch_time = time.time()
        
#         for index, data in enumerate(train_loader):
#             sm_example_tensor, example_target, condition_tensor = data
#             #Shape targets
#             example_target = example_target.view(-1, 1)
            
#             sm_example_tensor = sm_example_tensor.to(device)
#             example_target = example_target.to(device)
#             condition_tensor = condition_tensor.to(device)
            
#             optimize.zero_grad()
            
#             output = network(sm_example_tensor,
#                              condition_tensor)
#             loss_amount = loss_func(output, example_target.float())
#             loss_amount = loss_amount.float()
#             loss_amount.backward()
#             optimize.step()
            
#             current_loss += float(loss_amount.item())
            
#         print("Epoch {}, training_loss: {:.5f}, took: {:.2f}s".format(epoch+1, current_loss / num_batches, time.time() - epoch_time))

#         #At end of epoch, try validation set on GPU
        
#         total_val_loss = 0
#         network.eval()
#         with torch.no_grad():
#             for val_sm_tensor, val_target, val_cond_tensor in val_loader:
#                 val_target = val_target.view(-1, 1)

#                 #Send data to GPU
#                 val_sm_tensor = val_sm_tensor.to(device)
#                 val_target = val_target.to(device)
#                 val_cond_tensor = val_cond_tensor.to(device)
                
#                 #Forward pass only
#                 val_output = network(val_sm_tensor,
#                                      val_cond_tensor)
#                 val_loss_size = loss_func(val_output, val_target)
#                 total_val_loss += float(val_loss_size.item())
        
#             print("Val loss = {:.4f}".format(total_val_loss / len(val_loader)))
#             val_loss = total_val_loss / len(val_loader)
        
#         if val_loss <= best_val_loss:
#             best_val_loss = val_loss
#             current_best_state_dict = network.state_dict()
#             current_best_epoch = epoch+1
#             current_patience = es_patience
#         else:
#             current_patience = current_patience - 1
        
#         if current_patience == 0 or (time.time() - training_start) > 84600.0:
#             print('Early stopping/timeout engaged at epoch: ', epoch + 1)
#             print('Best results at epoch: ', current_best_epoch)
#             print("Training finished, took {:.2f}s".format(time.time() - training_start))
#             total_test_loss = 0
#             network.eval()
#             with torch.no_grad():
#                 for test_sm_tensor, test_target, test_cond_tensor in test_loader:
#                     test_target = test_target.view(-1, 1)
        
#                     #Send data to GPU
#                     test_sm_tensor = test_sm_tensor.to(device)
#                     test_target = test_target.to(device)
#                     test_cond_tensor = test_cond_tensor.to(device)
                    
#                     #Forward pass only
#                     test_output = network(test_sm_tensor,
#                                          test_cond_tensor)
#                     test_loss_size = loss_func(test_output, test_target)
#                     total_test_loss += float(test_loss_size.item())
                    
#                     test_loss = total_test_loss / len(test_loader)
        
#             return test_loss, current_best_state_dict
        
#         lr_sched.step(val_loss)
#         network.train()
    
#     print("Training finished, took {:.4f}s".format(time.time() - training_start))
#     print('Best results at epoch: ', current_best_epoch)

#     total_test_loss = 0
#     network.eval()
#     with torch.no_grad():
#         for test_sm_tensor, test_target, test_cond_tensor in test_loader:
#             test_target = test_target.view(-1, 1)

#             #Send data to GPU
#             test_sm_tensor = test_sm_tensor.to(device)
#             test_target = test_target.to(device)
#             test_cond_tensor = test_cond_tensor.to(device)
            
#             #Forward pass only
#             test_output = network(test_sm_tensor,
#                                  test_cond_tensor)
#             test_loss_size = loss_func(test_output, test_target)
#             total_test_loss += float(test_loss_size.item())
            
#             test_loss = total_test_loss / len(test_loader)

#     return test_loss, current_best_state_dict

# def end2end_locked_workflow(open_param_dict,
#                             sm_file_directory,
#                             ext_file_directory,
#                             label_keys,
#                             holdout_list,
#                             locked_formula,
#                             condition_dict,
#                             norm_strategy,
#                             val_split,
#                             training_epochs,
#                             batch_size,
#                             learning_rate,
#                             lr_patience,
#                             es_patience,
#                             output_name,
#                             preload_net,
#                             trained_net_directory,
#                             trained_net_name):
#     """
#     An end-to-end workflow for training a network **that has already had key molecular formula
#     pre-identified in another training process**, that attempts to predict the MS for a solvent-
#     extracted fraction.

#     Parameters
#     ----------
#     open_param_dict : dictionary
#         A dictionary file that contains information about the network depth, shape,
#         batch_normalization, and activation functions to be used.
#         Generates the fully connected network for training (or, to be loaded with state_dict)
#     sm_file_directory : string
#         Name of directory containing starting materials .csv files
#     ext_file_directory : string
#         Name of directory containing .csv files for extracted materials
#     label_keys : dictionary
#         A dictionary that describes which file names are associated with which
#         starting materials - used to label entries for future organization.
#         Possible labels are 'L1', 'S2', etc. for extracted materials, and
#         'L1_SM', 'S2_SM', etc. for starting materials themselves.
#     holdout_list : list
#         Name of MS .csv files (and, therefore, also .pkl files) that are held out
#         from training (for testing/cross-validation)
#     locked_formula : list
#         A list of formula that have been locked in after being identified as significant.         
#     val_split : float
#         Number betwee 0-1, fraction of examples to be used in validation loop. Final test set
#         is held out by putting the .csv and .pkl files in separate folders from training/validation data
#     condition_dict : dictionary
#         A dictionary that relates a given .csv file name to the extraction conditions used to generate it.
#         Provided in the top level function that calls FCSetNarrow for training/prediction
#         Entries are of the format: [frac_toluene, frac_dcm, frac_iPrOH, frac_acetone, fract_HOAc, frac_NEt3]
#     norm_strategy : string
#         A label that defines what kind of intensity normalization strategy is used.
#         'nor' = normalized to largest peak
#         'chg' = target is *change* in normalized intensity
#         'rel' = target is *change*/initial in normalized intensity - massively focuses on small high MW peaks
#         'sum' = peaks are described as fraction of the sum of total intensity
#         'schg' = target is *change* in fraction of sum intensity
#         'srel' = target is *change*/initial of fraction of sum intensity
#     training_epochs : integer
#         Number of epochs of training to attempt
#     batch_size : integer
#         Batch size for training (and validation)
#     learning_rate : float
#         Initial learning rate
#     lr_patience : integer
#         Number of epochs to wait for improvement before reducing learning rate
#     es_patience : integer
#         Number of epochs to wait for improvement before early stopping
#     output_name : string
#         Name to use for saving network state dictionaries
#     preload_net : Boolean
#         'True' indicates that a state_dictionary should be loaded for the network
#     trained_net_directory : string
#         Folder where the network state_dict() to be loaded can be found
#     trained_net_name : string
#         Name of the state_dict to load

#     Returns
#     -------
#     A record of training epoch vs. loss (which is also saved as a .csv)

#     """                
        
#     dataset = LabelledMSSet(sm_file_directory,
#                             ext_file_directory,
#                             label_keys,
#                             holdout_list,
#                             locked_formula,
#                             condition_dict,
#                             norm_strategy)
    
#     dataset.set_test_formula(locked_formula)

#     starting_net_width = (5 + len(dataset.current_test_formula))
    
#     network = FCExtNet(open_param_dict['num_layers'],
#                        starting_net_width,
#                        open_param_dict['strategy'],
#                        open_param_dict['batch_norm'],
#                        open_param_dict['activation'])
    
            
#     print('Training network with formula:', dataset.current_test_formula) 
    
#     val_loss, state_dict, test_frag = train_network_end2end(network,
#                                                             dataset,
#                                                             holdout_list,
#                                                             locked_formula,
#                                                             val_split,
#                                                             training_epochs,
#                                                             batch_size,
#                                                             learning_rate,
#                                                             lr_patience,
#                                                             es_patience)
        
#     dict_string = output_name + '.pt'
#     torch.save(state_dict, dict_string)
        
#     return 
 
# def nested_cross_val(open_param_dict,
#                        sm_file_directory,
#                        ext_file_directory,
#                        label_keys,
#                        holdout_list,
#                        locked_formula,
#                        learning_rates,
#                        dropout_rates,
#                        condition_dict,
#                        norm_strategy,
#                        val_split,
#                        test_split,
#                        training_epochs,
#                        batch_size,
#                        lr_patience,
#                        es_patience,
#                        output_name,
#                        preload_net,
#                        trained_net_directory,
#                        trained_net_name):
#     """
#     An end-to-end workflow for training a network with all possible combinations of possible + locked
#     formula, that attempts to predict the MS for a solvent-extracted fraction.

#     Parameters
#     ----------
#     open_param_dict : dictionary
#         A dictionary file that contains information about the network depth, shape,
#         batch_normalization, and activation functions to be used.
#         Generates the fully connected network for training (or, to be loaded with state_dict)
#     sm_file_directory : string
#         Name of directory containing starting materials .csv files
#     ext_file_directory : string
#         Name of directory containing .csv files for extracted materials
#     label_keys : dictionary
#         A dictionary that describes which file names are associated with which
#         starting materials - used to label entries for future organization.
#         Possible labels are 'L1', 'S2', etc. for extracted materials, and
#         'L1_SM', 'S2_SM', etc. for starting materials themselves.
#     holdout_list : list
#         Name of MS .csv files (and, therefore, also .pkl files) that are held out
#         from training (for testing/cross-validation)
#     possible_formula : list
#         List of all possible MS formula - identified with domain expertise - to attempt
#     locked_formula : list
#         A list of formula that have been locked in after being identified as significant.         
#     val_split : float
#         Number betwee 0-1, fraction of examples to be used in validation loop. Final test set
#         is held out by putting the .csv and .pkl files in separate folders from training/validation data
#     condition_dict : dictionary
#         A dictionary that relates a given .csv file name to the extraction conditions used to generate it.
#         Provided in the top level function that calls FCSetNarrow for training/prediction
#         Entries are of the format: [frac_toluene, frac_dcm, frac_iPrOH, frac_acetone, fract_HOAc, frac_NEt3]
#     norm_strategy : string
#         A label that defines what kind of intensity normalization strategy is used.
#         'sum' = peaks are described as fraction of the sum of total intensity
#         'schg' = target is *change* in fraction of sum intensity
#     training_epochs : integer
#         Number of epochs of training to attempt
#     batch_size : integer
#         Batch size for training (and validation)
#     learning_rate : float
#         Initial learning rate
#     lr_patience : integer
#         Number of epochs to wait for improvement before reducing learning rate
#     es_patience : integer
#         Number of epochs to wait for improvement before early stopping
#     output_name : string
#         Name to use for saving network state dictionaries
#     preload_net : Boolean
#         'True' indicates that a state_dictionary should be loaded for the network
#     trained_net_directory : string
#         Folder where the network state_dict() to be loaded can be found
#     trained_net_name : string
#         Name of the state_dict to load

#     Returns
#     -------
#     A record of best network and formula that led to it, which are also saved/pickled.

#     """                
    
#     result_list = []
    
#     current_best_test_loss = 100000.00
#     current_best_state_dict = None
    
#     dataset = LabelledMSSet(sm_file_directory,
#                             ext_file_directory,
#                             label_keys,
#                             holdout_list,
#                             locked_formula,
#                             condition_dict,
#                             norm_strategy)
    
#     dataset.set_test_formula(locked_formula)

    
#     for curr_learning_rate in learning_rates:
#         for curr_dropout_rate in dropout_rates:
#             print('Dropout rate:', curr_dropout_rate, 'Learning rate:', curr_learning_rate)
#             starting_net_width = (11 + len(dataset.current_test_formula))
#             print('Starting network is:', dataset.current_test_formula)    
#             if preload_net == False:
#                 network = FCExtNet(open_param_dict['num_layers'],
#                                    starting_net_width,
#                                    open_param_dict['strategy'],
#                                    open_param_dict['batch_norm'],
#                                    open_param_dict['activation'],
#                                    curr_dropout_rate)
#             else:
#                 network = load_e2e_network(trained_net_directory,
#                                            trained_net_name,
#                                            open_param_dict)
                
#             print('Training network with formula:', dataset.current_test_formula) 
            
#             test_loss, state_dict  = train_network_end2end(network,
#                                                                     dataset,
#                                                                     val_split,
#                                                                     test_split,
#                                                                     training_epochs,
#                                                                     batch_size,
#                                                                     curr_learning_rate,
#                                                                     lr_patience,
#                                                                     es_patience)
            
#             result_list.append((curr_dropout_rate, curr_learning_rate, test_loss))
            
#             if test_loss < current_best_test_loss:
#                 current_best_test_loss = test_loss
#                 current_best_state_dict = state_dict
#                 print('New best settings are:', curr_dropout_rate, curr_learning_rate)
    
#     result_list.sort(key=lambda x: x[1])
    
#     print('Sorted results:')
#     print(result_list)
    
#     dict_string = output_name + '.pt'
#     torch.save(current_best_state_dict, dict_string)
    
#     pickle_string = output_name + '.pkl'
#     pickle_file = open(pickle_string, 'wb')
#     pickle.dump(result_list, pickle_file)
#     pickle_file.close()
    
#     return 

