#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 11:35:43 2023

@author: jvh

Final scripts to take trained networks, makes predictions on held-out data (cross validation)
and compare those predictions vs 'persistence' and 'average' results as pred v. act as well as mse
violin plots. Use 2 colors for violin plots - from 'appearing' ions, as well as normal ions.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import time
from pathlib import Path
import DataTools.BitumenCreateUseDataset as CUD
import DataTools.BitumenCSVtoDict as CTD
import FCNets.BitumenFCNets as BFN
import numpy as np
import pickle
import pandas as pd
import FCNets.BitumenWorkflows as BWF
import DataTools.BitumenCreateUseDataset as CUD

def full_evaluation_of_trained_network(sm_file_directory,
                                       ext_file_directory,
                                       open_param_dict,
                                       label_keys,
                                       condition_dict,
                                       holdout_list,
                                       locked_formula,
                                       trained_net_directory,
                                       trained_net_name,
                                       calculate_pers_avg,
                                       pers_avg_output_name,
                                       trained_eval_output_name):
    """
    A top-level function that takes a trained network, and does the final evaluation between that trained network
    and a simple averaging of extractions, as well as the persistence of the starting material.
    During training and dataset generation, ions are sorted according to their properties (are the newly 'appearing'
    in the fraction, or 'persistent') and plotting can also maintain this sorting if desired

    Parameters
    ----------
    sm_file_directory : string
        Name of the folder that contains the MS for the extraction starting materials
    ext_file_directory : string
        Name of the folder that contains the MS for the extracted fractions. Each starting
        material should have a dedicated folder for its extracted fractions
    open_param_dict : dictionary
        A dictionary that gives the key network parameters to form a fully-connected network of correct form
    label_keys : dictionary
        A dictionary that relates the file name for an extraction to the starting material that was used
    condition_dict : dictionary
        A dictionary that contains the extraction conditions for every experiment
    holdout_list : list
        A list that describes which extractions (from the extraction file directory) were held out during training
        These files are still held out from the 'training' dictionary created, and are used to create the 'test'
        dictionary.
    locked_formula : list
        A list of the molecular fragment relationships that will be used in the network
    trained_net_directory : string
        Name of the folder that contains the trained network being evaluated
    trained_net_name : string
        Name of the specific .pt state_dictionary that will be loaded
    calculate_pers_avg : Boolean
        A Boolean flag - do you want the performance of the 'persistance' and 'average' prediction calculated
        and saved? In the case of cross-validation, where all CV slices have the same pers/avg, you would only
        want to have this done once ('True') and skipped every other time ('False')
    pers_avg_output_name : string
        If being calculated, a leading string to append to the pers/avg performance output
    trained_eval_output_name : string
        A leading string to be appended to the pred v. act and per-pred error output for this run

    Returns
    -------
    None, but saves the network performance as two .csv files (pred vs. act, and error distribution for violin plot)
    If desired, it also saves the performance of the 'average' and 'persistant' predictions.
    Last, the percent of observed ions that are predicted is calculated and saved.

    """
    
    #Create training and test datasets. Update testing molecular fragments.
    train_file_string = trained_eval_output_name + 'train_dict'
    test_file_string = trained_eval_output_name + 'test_dict'
    
    train_dataset = BFN.BitumenExtTrainingData(sm_file_directory=sm_file_directory,
                                               ext_file_directory=ext_file_directory,
                                               label_keys=label_keys,
                                               test_list=holdout_list,
                                               locked_formula=locked_formula,
                                               condition_dict=condition_dict,
                                               output_name=train_file_string)
    train_dataset.set_test_formula(locked_formula)    
    test_dataset = BFN.BitumenExtTestData(sm_file_directory=sm_file_directory,
                                          ext_file_directory=ext_file_directory,
                                          label_keys=label_keys,
                                          test_list=holdout_list,
                                          locked_formula=locked_formula,
                                          condition_dict=condition_dict,
                                          output_name=test_file_string)
    test_dataset.set_test_formula(locked_formula)
    #Load pre-trained network
    trained_network = BFN.load_extraction_network(trained_net_directory=trained_net_directory,
                                                  trained_net_name=trained_net_name,
                                                  net_param_dict=open_param_dict,
                                                  eval_or_train_data=train_dataset)

    #Calculate % 'coverage' of training vs. testing set and excess
    trained_ions = set(train_dataset.total_ion_list)
    testing_ions = set(test_dataset.total_ion_list)
    matched_ions = [x for x in testing_ions if x in trained_ions]
    coverage = len(matched_ions) / len(testing_ions)
    print('For evaluation of', trained_eval_output_name)
    print('Coverage of the sets is:', coverage)
    excess = len(trained_ions) / len(testing_ions)
    print('With an excess of:', excess)

    if calculate_pers_avg == True:
        #Calculate and return 'persistence' pred v. act and per-point error
        persist_pred, persist_act, persist_ppe, persist_stack = calculate_persistence_prediction(train_dataset,
                                                                                                 test_dataset)
        #Calculate and return 'average' pred v. act and per-point error
        average_pred, average_act, average_ppe, average_stack = calculate_average_prediction(train_dataset,
                                                                                             test_dataset)
        per_pickle_string = pers_avg_output_name + '_pers_predictions.pkl'
        per_pickle_file = open(per_pickle_string, 'wb')
        pickle.dump((persist_pred, persist_act, persist_ppe, persist_stack), per_pickle_file)
        per_pickle_file.close()
        
        avg_pickle_string = pers_avg_output_name + '_avg_predictions.pkl'
        avg_pickle_file = open(avg_pickle_string, 'wb')
        pickle.dump((average_pred, average_act, average_ppe, average_stack), avg_pickle_file)
        avg_pickle_file.close()
        
    #Predict all test samples, and return pred v. act and per-point error
    ml_pred, ml_act, ml_ppe, ml_stack = calculate_ml_error(train_dataset,
                                                           test_dataset,
                                                           trained_network)

    ml_pickle_string = trained_eval_output_name + '.pkl'
    ml_pickle_file = open(ml_pickle_string, 'wb')
    pickle.dump((ml_pred, ml_act, ml_ppe, ml_stack), ml_pickle_file)
    ml_pickle_file.close()    

    return 

def calculate_persistence_prediction(training_dataset,
                                     testing_dataset):
    """
    A function that takes a training & test dataset, and calculates (point by point)
    the accuracy of a model that simply predicts that nothing changes (baseline comparison)
    
    Unlike other evaluation functions, this approach can "predict" 100% of test peaks (predicts lots of zero values)
    
    Returns 3 pieces of data for creating the following: 
        a 'predicted vs. actual' plot,
        a 'predicted / actual / error' stacked plot by molecular weight and
        a list of per-point (raw) error for creating violin plots
    

    Parameters
    ----------
    training_dataset : BitumenExtTrainingData object as defined in BitumenFCNets
        The (sum-normalized) per-spectrum-per-point training data used when training a given network
    testing_dataset : BitumenExtTestData object as defined in BitumenFCNets
        The (sum-normalized) per-spectrum-per-point test data used to evaluate a given network

    Returns
    -------
    4 dictionary objects containing slightly different data for evaluating network performance

    """

    #'app' is for new ions that appear, 'smp' is for ions that are starting-material-present
    #act vs pred and ppe are ordered just by this ion type: for stacked plots, you need to separate
    #by file name, otherwise each ion will have a different value for each extraction.
    ordered_pred = {'app': [], 'smp': []}
    ordered_act = {'app': [], 'smp': []}
    per_point_error = {'app': [], 'smp': []}
    stacked_plot_lists = {}

    #All test extractions (at this point) will have common starting material file
    sm_file_label = testing_dataset.test_keys[0][1] + '_SM'    
    for possible_sm_file in testing_dataset.test_dictionary.keys():
        if sm_file_label in possible_sm_file:
            true_sm_name = possible_sm_file
            break

    for file_name, sm_file_name, formula_tuple, ion_type in testing_dataset.test_keys:
        if file_name not in stacked_plot_lists.keys():
            stacked_plot_lists[file_name] = {'app': [], 'smp': []}
            
        if ion_type != 'dis':
            actual_ion_value = np.float32(testing_dataset.test_dictionary[file_name][1][formula_tuple])
        else:
            actual_ion_value = np.float32(0.0)
            
            
        if ion_type == 'app':
            predicted_ion_value = 0
        else:
            predicted_ion_value = np.float32(testing_dataset.test_dictionary[true_sm_name][1][formula_tuple])            
        ppe = predicted_ion_value - actual_ion_value
        
        #Update all lists
        if ion_type == 'app':
            ordered_pred['app'].append(predicted_ion_value)
            ordered_act['app'].append(actual_ion_value)
            per_point_error['app'].append(ppe)
            stacked_plot_lists[file_name]['app'].append([CTD.formula_to_mass(formula_tuple), predicted_ion_value, actual_ion_value, ppe])

        else:
            ordered_pred['smp'].append(predicted_ion_value)
            ordered_act['smp'].append(actual_ion_value)
            per_point_error['smp'].append(ppe)
            stacked_plot_lists[file_name]['smp'].append([CTD.formula_to_mass(formula_tuple), predicted_ion_value, actual_ion_value, ppe])

    return ordered_pred, ordered_act, per_point_error, stacked_plot_lists
            
def calculate_average_prediction(training_dataset,
                                 testing_dataset):
    """
    A function that takes a training & test dataset, and calculates (point by point)
    the accuracy of a model that simply predicts the average observed extraction from the training set
    
    Cannot predict 100% of peaks, uses the training data to predict which doesn't have perfect overlap
    
    Returns 3 pieces of data for creating the following: 
        a 'predicted vs. actual' plot,
        a 'predicted / actual / error' stacked plot by molecular weight and
        a list of per-point (raw) error for creating violin plots
    

    Parameters
    ----------
    training_dataset : BitumenExtTrainingData object as defined in BitumenFCNets
        The (sum-normalized) per-spectrum-per-point training data used when training a given network
    testing_dataset : BitumenExtTestData object as defined in BitumenFCNets
        The (sum-normalized) per-spectrum-per-point test data used to evaluate a given network

    Returns
    -------
    4 dictionary objects containing slightly different data for evaluating network performance

    """

    #'app' is for new ions that appear, 'smp' is for ions that are starting-material-present
    #act vs pred and ppe are ordered just by this ion type: for stacked plots, you need to separate
    #by file name, otherwise each ion will have a different value for each extraction.

    observed_ion_list = training_dataset.total_ion_list
    test_ion_dict = testing_dataset.observed_ion_dictionary

    #Create file name list for testing and populate stacked plot dictionary
    test_file_list = []
    stacked_plot_lists = {}
    for poss_file in testing_dataset.test_dictionary.keys():
        if 'SM' not in poss_file:
            test_file_list.append(poss_file)
            stacked_plot_lists[poss_file] = {'app': [], 'smp': []}
    
    ordered_pred = {'app': [], 'smp': []}
    ordered_act = {'app': [], 'smp': []}
    per_point_error = {'app': [], 'smp': []}

    #All test extractions (at this point) will have common starting material file
    sm_file_short = testing_dataset.test_keys[0][1]
    sm_file_label = sm_file_short + '_SM'    
    for possible_sm_file in testing_dataset.test_dictionary.keys():
        if sm_file_label in possible_sm_file:
            true_sm_name = possible_sm_file
            break

    #Make predictions for every ion observed in training data, and per-file
    #Check per-file ion dictionary in test dataset to determine ion type
    
    for current_ion in observed_ion_list:
        #Gather/calculate average from all previous extractions, excluding the starting material spectrum
        running_avg = []
        for training_example in training_dataset.training_dictionary.keys():
            if 'SM' not in training_dataset.training_dictionary[training_example][0]:
                try:
                    obs_value = training_dataset.training_dictionary[training_example][1][current_ion]
                    running_avg.append(obs_value)
                except:
                    running_avg.append(0.0)
        predicted_ion_value = (sum(running_avg) / len(running_avg))

        #Check for presence in SM
        if current_ion in testing_dataset.test_dictionary[true_sm_name][1].keys():
            ion_type = 'smp'
        else:
            ion_type = 'app'
            
        for test_file in test_file_list:    
            if current_ion in test_ion_dict[test_file]:
                actual_ion_value = np.float32(testing_dataset.test_dictionary[test_file][1][current_ion])
                
            else:
                actual_ion_value = np.float32(0.0)    
                
            ppe = predicted_ion_value - actual_ion_value

            #Update all lists
            ordered_pred[ion_type].append(predicted_ion_value)
            ordered_act[ion_type].append(actual_ion_value)
            per_point_error[ion_type].append(ppe)
            stacked_plot_lists[test_file][ion_type].append([CTD.formula_to_mass(current_ion), predicted_ion_value, actual_ion_value, ppe])
    
    return ordered_pred, ordered_act, per_point_error, stacked_plot_lists

def calculate_ml_error(training_dataset,
                       testing_dataset,
                       trained_network):
    """
    

    Parameters
    ----------
    training_dataset : BitumenExtTrainingData object as defined in BitumenFCNets
        The (sum-normalized) per-spectrum-per-point training data used when training a given network
    testing_dataset : BitumenExtTestData object as defined in BitumenFCNets
        The (sum-normalized) per-spectrum-per-point test data used to evaluate a given network
    trained_network : FCExtNet as defined in BitumenFCNets
        The pre-trained ML network making predictions on extracted fractions

    Returns
    -------
    4 dictionary objects containing slightly different data for evaluating network performance

    """
    #Get trained network ready
    trained_network.eval()
    if torch.cuda.is_available() == True:
        device = 'cuda'
    elif torch.backends.mps.is_available() == True:
        device = 'mps'
    else:
        raise ValueError('No GPU available')
    
    trained_network.to(device)
    
    #'app' is for new ions that appear, 'smp' is for ions that are starting-material-present
    #act vs pred and ppe are ordered just by this ion type: for stacked plots, you need to separate
    #by file name, otherwise each ion will have a different value for each extraction.

    observed_ion_list = training_dataset.total_ion_list
    test_ion_dict = testing_dataset.observed_ion_dictionary

    #Create file name list for testing and populate stacked plot dictionary
    test_file_list = []
    stacked_plot_lists = {}
    for poss_file in testing_dataset.test_dictionary.keys():
        if 'SM' not in poss_file:
            test_file_list.append(poss_file)
            stacked_plot_lists[poss_file] = {'app': [], 'smp': []}
    
    ordered_pred = {'app': [], 'smp': []}
    ordered_act = {'app': [], 'smp': []}
    per_point_error = {'app': [], 'smp': []}

    #All test extractions (at this point) will have common starting material file
    sm_file_short = testing_dataset.test_keys[0][1]
    sm_file_label = sm_file_short + '_SM'    
    for possible_sm_file in testing_dataset.test_dictionary.keys():
        if sm_file_label in possible_sm_file:
            true_sm_name = possible_sm_file
            break

    #Make predictions for every ion observed in training data, and per-file
    #Check per-file ion dictionary in test dataset to determine ion type
    
    for current_ion in observed_ion_list:
        #Check for presence in SM
        if current_ion in testing_dataset.test_dictionary[true_sm_name][1].keys():
            ion_type = 'smp'
        else:
            ion_type = 'app'

        #Predict extracted ion intensity
        extraction_tensor = CUD.create_extraction_tensor(sm_file_short,
                                                         current_ion,
                                                         ion_type,
                                                         testing_dataset.test_dictionary,
                                                         testing_dataset.return_curr_formula())
        extraction_tensor = extraction_tensor.to(device)
        extraction_tensor = extraction_tensor.unsqueeze(0)
        
        for test_file in test_file_list:    
            if current_ion in test_ion_dict[test_file]:
                actual_ion_value = np.float32(testing_dataset.test_dictionary[test_file][1][current_ion])
                
            else:
                actual_ion_value = np.float32(0.0)    

            for extract_cond in testing_dataset.condition_dict.keys():
                if extract_cond in test_file:
                    example_extraction = testing_dataset.condition_dict[extract_cond]
                    condition_tensor = torch.FloatTensor(example_extraction)
                    condition_tensor = condition_tensor.to(device)
            
            #Add dimension to tensors, which are being predicted singularly
            condition_tensor=condition_tensor.unsqueeze(0)
            with torch.no_grad():
                predicted_ion_tensor = trained_network(extraction_tensor, condition_tensor)
                predicted_ion_tensor = predicted_ion_tensor.to('cpu')
                
            predicted_ion_value = float(predicted_ion_tensor)
            if predicted_ion_value < 0.0:
                predicted_ion_value = 0.0
            
            ppe = predicted_ion_value - actual_ion_value

            #Update all lists
            ordered_pred[ion_type].append(predicted_ion_value)
            ordered_act[ion_type].append(actual_ion_value)
            per_point_error[ion_type].append(ppe)
            stacked_plot_lists[test_file][ion_type].append([CTD.formula_to_mass(current_ion), predicted_ion_value, actual_ion_value, ppe])
    
    return ordered_pred, ordered_act, per_point_error, stacked_plot_lists
       