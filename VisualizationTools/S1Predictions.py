#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 09:10:04 2023

@author: jvh

Top level script for testing nearest neighbor trained networks

Solvent order for fingerprints is [toluene, DCM, iPrOH, acetone, AcOH, NEt3]
"""

import FCNets.BitumenWorkflows as BWF
import FCNets.BitumenFCNets as BFN
import torch.nn as nn
from pathlib import Path
import pandas as pd
from ast import literal_eval
import glob
import numpy as np

label_keys = {'S1P1_SM': 'S1P1_SM',
              'S1P2_SM': 'S1P2_SM',
              '20060': 'S1P1',
              '20061': 'S1P2',
              '20062': 'S1P1',
              '20063': 'S1P2',
              '20064': 'S1P1',
              '20065': 'S1P2',
              '20066': 'S1P1',
              '20067': 'S1P2'}

condition_dict = {'S1P1_SM': [0, 0, 0, 0, 0, 0],
                  'S1P2_SM': [0, 0, 0, 0, 0, 0],
                  '20060': [0.420, 0, 0, 0.580, 0, 0],
                  '20061': [0.420, 0, 0, 0.580, 0, 0],
                  '20062': [0.720, 0, 0.280, 0, 0, 0],
                  '20063': [0.720, 0, 0.280, 0, 0, 0],
                  '20064': [0, 0.620, 0, 0.380, 0, 0],
                  '20065': [0, 0.620, 0, 0.380, 0, 0],
                  '20066': [0, 0.630, 0.370, 0, 0, 0],
                  '20067': [0, 0.630, 0.370, 0, 0, 0]}

trained_net_directory = 'ExtTIS Full LOO'

trained_net_list_d11 = ['LOO_19208_D11.pt',
                        'LOO_19209_D11.pt',
                        'LOO_19210_D11.pt',
                        'LOO_19211_D11.pt',
                        'LOO_19212_D11.pt',
                        'LOO_19213_D11.pt',
                        'LOO_19214_D11.pt',
                        'LOO_19215_D11.pt',
                        'LOO_19216_D11.pt',
                        'LOO_19217_D11.pt',
                        'LOO_19218_D11.pt',
                        'LOO_19219_D11.pt',
                        'LOO_19220_D11.pt',
                        'LOO_19221_D11.pt',
                        'LOO_19222_D11.pt',
                        'LOO_19223_D11.pt',
                        'LOO_19224_D11.pt',
                        'LOO_19226_D11.pt',
                        'LOO_19227_D11.pt',
                        'LOO_19228_D11.pt',
                        'LOO_19229_D11.pt',
                        'LOO_19230_D11.pt',
                        'LOO_19231_D11.pt',
                        'LOO_19232_D11.pt',
                        'LOO_19233_D11.pt',
                        'LOO_19234_D11.pt',
                        'LOO_19235_D11.pt',
                        'LOO_19236_D11.pt',
                        'LOO_19237_D11.pt',
                        'LOO_19238_D11.pt',
                        'LOO_19239_D11.pt',
                        'LOO_19240_D11.pt',
                        'LOO_19241_D11.pt',
                        'LOO_19242_D11.pt',
                        'LOO_19243_D11.pt']

trained_net_param_dict = {'layer_size_list': [800,800,800,800,800],
                          'batch_norm': False,
                          'activation': nn.ReLU(),
                          'dropout_amt': 0.05}

dataset_param_dict = {'sm_file_directory': 'ExpCSV/S1_SM_folder',
                      'ext_file_directory': 'ExpCSV/S1_ext_folder',
                      'label_keys': label_keys,
                      'condition_dict': condition_dict,
                      'test_list': ['20061_nar.csv'],
                      'locked_formula_list': [(1,2,0,0,0),(-1,-2,0,0,0),
                                              (3,6,0,0,0),(-3,-6,0,0,0),
                                              (4,8,0,0,0),(-4,-8,0,0,0),
                                              (0,2,0,0,0),(0,-2,0,0,0),
                                              (2,2,0,0,0),(-2,-2,0,0,0),
                                              (4,2,0,0,0),(-4,-2,0,0,0),
                                              (2,0,0,0,1),(-2,0,0,0,-1),
                                              (2,2,0,0,-1),(-2,-2,0,0,1),
                                              (5,8,0,0,0),(-5,-8,0,0,0),
                                              (2,0,0,0,0),(-2,0,0,0,0),
                                              (3,2,0,0,0),(-3,-2,0,0,0)],
                      'pickle_output': False,
                      'output_name': '20061_mismatched_by_committee'}

def create_model_dictionary(trained_net_directory,
                            trained_net_name_list,
                            trained_net_parameters,
                            dataset_param_dict):
    """
    A function that takes a list of trained Pytorch networks, and creates a network for
    evaluation for each stored in a dictionary. For use in query by committee testing

    Parameters
    ----------
    trained_net_directory : string
        Name of the directory containing trained networks
    trained_net_name_list : list of string
        Name of each file that will be used for evaluation
    trained_net_parameters : dictionary
        A dictionary that contains all of the network information necessary to
        create the trained networks for evaluation
    dataset_param_dict : dictionary
        A dictionary that contains all of the information necessary to make the
        training/testing dataset for evaluation. Even in cases where no training is done,
        the training set (even if empty) is used to create the list of ions to predict

    Returns
    -------
    A dictionary of trained extraction models

    """
    trained_model_dictionary = {}
    
    for specific_net in trained_net_name_list:
        trained_network=BFN.load_extraction_tis_network(trained_net_directory,
                                                        specific_net,
                                                        trained_net_param_dict,
                                                        dataset_param_dict['locked_formula_list'])
        trained_model_dictionary[specific_net] = trained_network
    
    return trained_model_dictionary


output_name = 'test_mismatch_committee_20061'
trained_net_dictionary = create_model_dictionary(trained_net_directory=trained_net_directory,
                                                 trained_net_name_list=trained_net_list_d11,
                                                 trained_net_parameters=trained_net_param_dict,
                                                 dataset_param_dict=dataset_param_dict)

mse_dict, ppe_dict, ext_dataset = BWF.test_extraction_via_committee(trained_net_dictionary,
                                                                dataset_param_dict,
                                                                output_name,
                                                                csv_output=True,
                                                                dataset_pass=None)
