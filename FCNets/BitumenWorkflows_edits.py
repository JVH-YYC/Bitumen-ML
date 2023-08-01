#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:42:03 2022

@author: jvh

Training/testing/output functions for ML Bitumen workflows

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import time
from pathlib import Path
import DataTools.BitumenCreateUseDataset_edits as CUD
import DataTools.BitumenCSVtoDict as CTD
import FCNets.BitumenFCNets_edits as BFN
import numpy as np
import pickle
import pandas as pd
import random

def train_and_test_single_fig(network,
                              train_dataset,
                              test_dataset,
                              val_split,
                              test_split,
                              learning_rate,
                              lr_patience,
                              es_patience,
                              training_epochs,
                              batch_size):
    """
    
    A function for performing a single (full) training and testing cycle for
    measuring the 'formula information level' of a given formula.
    
    The network and dataset are constructed elsewhere for efficiency
    
    Parameters
    ----------
    network : FIGNet object created in BitumenFCNets
        A fully connected NN that is used to evaluated the level of
        'formula information gain' that comes from adding/subtracting
        small formula units from an ion
    train_dataset : FIGTrainingData object created in BitumenFCNets
        Training dataset that has had the formula formula updates made elsewhere
    test_dataset : FIGTestData object created in BitumenFCNets or None
        Test dataset that is created in parallel to training dataset
        Can be set to 'None' if a single training dataset is going to be
        split into training/validation/testing samples instead
    val_split : float
        Fraction of the training data to use for validation
    test_split : float or None
        Fraction of the training data to use for testing. Can be set to 'None' if entire
        .csv files will be held out for testing (as FIGTestData object)
    learning_rate : float
        Initial learning rate
    lr_patience : integer
        Number of training epochs with no improvement that will be allowed before the
        learning rate is reduced
    es_patience : integer
        Number of training epochs with no improvement that will be allowed before early
        stopping is engaged, and the best state_dict is reloaded
    training_epochs : integer
        The maximum number of training epochs allowed
    batch_size : integer
        Batch size for network training

    Returns
    -------
    A dictionary that contains several lists (that can/will be saved from higher level function), that
    show training/validation stats per epoch, predicted v. actual error for all 3 sets (for each point,
    so that violin plots can be made) as well as a singular test error for ranking during optimization

    """
    
    train_index, val_index, test_index = BFN.test_train_val_split(train_dataset,
                                                                  val_split,
                                                                  test_split)

    train_sample = SubsetRandomSampler(train_index)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sample)
    
    val_sample = SubsetRandomSampler(val_index)
    val_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=val_sample)
    
    if test_dataset is None:
        try:
            test_sample = SubsetRandomSampler(test_index)
            test_loader = torch.utils.data.DataLoader(train_dataset,
                                                      batch_size=batch_size,
                                                      sampler=test_sample)
        except:
            raise ValueError('test_split and test_dataset not properly matched for training loop')

    else:
        test_index = list(range(test_dataset.__len__()))
        test_sample = SubsetRandomSampler(test_index)
        test_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  sampler=test_sample)

    num_batches = len(train_loader)
    loss_func, optimize, lr_sched = BFN.loss_and_optim(network,
                                                       learning_rate,
                                                       lr_patience)

    training_start = time.time()

    performance_dict = {'train_act': [],                             
                        'train_pred': [],
                        'train_mse': [],
                        'val_act': [],
                        'val_pred': [],
                        'val_mse': [],
                        'test_act': [],
                        'test_pred': [],
                        'test_mse': [],
                        'train_loss_list': [],
                        'val_loss_list': []}

    best_val_loss = 1000.0
    current_patience = es_patience

    if torch.cuda.is_available() == True:
        device = torch.device('cuda')
    elif not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
        print('Using Apple MPS GPU')

    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        print('Multiple GPU Detected')
        network = nn.DataParallel(network)

    network.to(device)
    
    for epoch in range(training_epochs):
        current_loss = 0.0
        epoch_time = time.time()
        
        for example_tensor, example_target in train_loader:
            example_target = example_target.view(-1, 1)
            
            example_tensor = example_tensor.to(device)
            example_target = example_target.to(device)
            
            optimize.zero_grad()
            
            output = network(example_tensor)
            loss_amount = loss_func(output, example_target.float())
            loss_amount = loss_amount.float()
            loss_amount.backward()
            optimize.step()
            
            current_loss += float(loss_amount.item())

        if epoch % 10 == 0:
            print("Epoch {}, training_loss: {:.5f}, took: {:.2f}s".format(epoch+1, current_loss / num_batches, time.time() - epoch_time))

        performance_dict['train_loss_list'].append(current_loss / num_batches)
        #At end of epoch, try validation set on GPU

        total_val_loss = 0
        network.eval()
        with torch.no_grad():
            for val_tensor, val_target in val_loader:
                val_target = val_target.view(-1, 1)

                #Send data to GPU
                val_tensor = val_tensor.to(device)
                val_target = val_target.to(device)
                
                #Forward pass only
                val_output = network(val_tensor)
                val_loss_size = loss_func(val_output, val_target)
                total_val_loss += float(val_loss_size.item())
            
            if epoch % 10 == 0:
                print("Val loss = {:.4f}".format(total_val_loss / len(val_loader)))
            
            val_loss = total_val_loss / len(val_loader)
            performance_dict['val_loss_list'].append(val_loss)
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            current_best_state_dict = network.state_dict()
            current_best_epoch = epoch+1
            current_patience = es_patience
        else:
            current_patience = current_patience - 1
        
        if current_patience == 0 or (time.time() - training_start) > 84600.0:
            print('Early stopping/timeout engaged at epoch: ', epoch + 1)
            print('Best results at epoch: ', current_best_epoch)
            print("Training finished, took {:.2f}s".format(time.time() - training_start))
            network.load_state_dict(current_best_state_dict)
                        
            total_test_loss = 0
            network.eval()
            with torch.no_grad():
                for example_tensor, example_target in train_loader:
                    example_target = example_target.view(-1, 1)
                    
                    train_act_list = example_target.flatten().tolist()
                    performance_dict['train_act'] += train_act_list
                    
                    example_tensor = example_tensor.to(device)
                    train_pred = network(example_tensor)
                    
                    train_pred_list = train_pred.flatten().tolist()
                    performance_dict['train_pred'] += train_pred_list
                    
                    train_mse_list = [(x - y )**2 for x, y in zip(train_act_list, train_pred_list)]
                    performance_dict['train_mse'] += train_mse_list

                for val_tensor, val_target in val_loader:
                    val_target = val_target.view(-1, 1)
                    
                    val_act_list = val_target.flatten().tolist()
                    performance_dict['val_act'] += val_act_list
                    
                    val_tensor = val_tensor.to(device)
                    val_pred = network(val_tensor)
                    
                    val_pred_list = val_pred.flatten().tolist()
                    performance_dict['val_pred'] += val_pred_list
                    
                    val_mse_list = [(x - y)**2 for x, y in zip(val_act_list, val_pred_list)]
                    performance_dict['val_mse'] += val_mse_list
                
                for test_tensor, test_target in test_loader:
                    test_target = test_target.view(-1, 1)
                    
                    test_act_list = test_target.flatten().tolist()
                    performance_dict['test_act'] += test_act_list
        
                    test_tensor = test_tensor.to(device)
                    test_target = test_target.to(device)
                    
                    test_output = network(test_tensor)
                    test_pred_list = test_output.flatten().tolist()
                    performance_dict['test_pred'] += test_pred_list
                    
                    test_loss_size = loss_func(test_output, test_target)
                    total_test_loss += float(test_loss_size.item())
                    
                    test_loss = total_test_loss / len(test_loader)
                    performance_dict['test_loss'] = test_loss
        
                    test_mse_list = [(x - y)**2 for x, y in zip(test_act_list, test_pred_list)]
                    performance_dict['test_mse'] += test_mse_list
        
            return performance_dict, current_best_state_dict
        
        lr_sched.step(val_loss)
        network.train()
    
    print("Training finished, took {:.4f}s".format(time.time() - training_start))
    print('Best results at epoch: ', current_best_epoch)

    total_test_loss = 0
    network.eval()
    with torch.no_grad():
        for example_tensor, example_target in train_loader:
            example_target = example_target.view(-1, 1)
            
            train_act_list = example_target.flatten().tolist()
            performance_dict['train_act'] += train_act_list
            
            example_tensor = example_tensor.to(device)
            train_pred = network(example_tensor)
            
            train_pred_list = train_pred.flatten().tolist()
            performance_dict['train_pred'] += train_pred_list
            
            train_mse_list = [(x - y )**2 for x, y in zip(train_act_list, train_pred_list)]
            performance_dict['train_mse'] += train_mse_list

        for val_tensor, val_target in val_loader:
            val_target = val_target.view(-1, 1)
            
            val_act_list = val_target.flatten().tolist()
            performance_dict['val_act'] += val_act_list
            
            val_tensor = val_tensor.to(device)
            val_pred = network(val_tensor)
            
            val_pred_list = val_pred.flatten().tolist()
            performance_dict['val_pred'] += val_pred_list
            
            val_mse_list = [(x - y)**2 for x, y in zip(val_act_list, val_pred_list)]
            performance_dict['val_mse'] += val_mse_list
        
        for test_tensor, test_target in test_loader:
            test_target = test_target.view(-1, 1)
            
            test_act_list = test_target.flatten().tolist()
            performance_dict['test_act'] += test_act_list

            test_tensor = test_tensor.to(device)
            test_target = test_target.to(device)
            
            test_output = network(test_tensor)
            test_pred_list = test_output.flatten().tolist()
            performance_dict['test_pred'] += test_pred_list
            
            test_loss_size = loss_func(test_output, test_target)
            total_test_loss += float(test_loss_size.item())
            
            test_loss = total_test_loss / len(test_loader)
            performance_dict['test_loss'] = test_loss

            test_mse_list = [(x - y)**2 for x, y in zip(test_act_list, test_pred_list)]
            performance_dict['test_mse'] += test_mse_list

    return performance_dict, current_best_state_dict

def measure_all_fig_levels(sm_file_directory,
                           ext_file_directory,
                           open_param_dict,
                           label_keys,
                           holdout_list,
                           locked_formula,
                           possible_formula_list,
                           number_repeat,
                           csv_output_name,
                           val_split,
                           test_split,
                           training_epochs,
                           batch_size,
                           learning_rate,
                           lr_patience,
                           es_patience):
    """
    A function that examines all of the formula information gain levels for all files in a given
    extraction file directory, using all possible formula in the possible_formula_list, in combination
    with any locked_formula.

    As the extraction files are evaluated one-at-a-time for this analysis, a single file is split
    into training/validation/test sets. Given that some splits may be easier/harder to analyze,
    the analysis is repeated multiple times and averaged.

    Parameters
    ----------
    sm_file_directory : string
        Name of sub-directory that contains starting material MS files
    ext_file_directory : string
        Name of sub-directory that contains extraction MS files
    open_param_dict : dictionary
        A dictionary that contains the necessary description to create a fully connected NN
    label_keys : dictionary
        A dictionary that tells which extracted fraction came from which starting material
    holdout_list : list
        Name of files that should be excluded from creation of the training dataset
    locked_formula : list of tuples
        A list of formula that are included in every analysis, +/- the new formula being tested
    possible_formula_list : list of tuples
        A list of formula that will be evaluated individually in this workflow
    number_repeat : integer
        The number of times the training will be repeated/averaged per formula
    csv_output_name : string
        A leading string that will be added to all .csv output files
    val_split : float
        The fraction of MS peaks that will be used for validation
    test_split : float
        The fraction of MS peaks that will be used for final testing
    training_epochs : integer
        Maximum number of training epochs that will be used
    batch_size : integer
        Batch size for data loading
    learning_rate : float
        Initial learning rate used during ML training
    lr_patience : integer
        Number of training epochs with no improvement that will be allowed before learning rate is reduced
    es_patience : integer
        Number of training epochs with no improvement that will be allowed before early stopping is engaged

    Returns
    -------
    None, but saves results as .csv files

    """

    #Create starting dataframe that will be grown by appending results
    return_frame = pd.DataFrame(columns=possible_formula_list)
    
    #Gather list of .csv files to process
    file_process_list = CTD.define_sm_ext_train(sm_file_directory,
                                                ext_file_directory,
                                                label_keys,
                                                holdout_list)

    full_list_of_labels = []
    
    for current_label, current_file in file_process_list:
        full_list_of_labels.append(current_file)
    
    for current_file in full_list_of_labels:
        #Create individual Dataset object for single file
        print('Processing:', current_file)
        curr_holdout_list = [x.split('/')[-1] for x in full_list_of_labels if current_file != x]
        
        curr_dataset = BFN.FIGTrainingData(sm_file_directory,
                                           ext_file_directory,
                                           label_keys,
                                           curr_holdout_list,
                                           locked_formula)
        
        print('Full set size is:', len(curr_dataset))
        curr_file_losses = []

        for possible_formula in possible_formula_list:
            print('Attaching:', possible_formula)
            average_loss_list = []
            
            BFN.update_formula(curr_dataset,
                               possible_formula,
                               locked_formula)
            starting_net_width = 5 + len(curr_dataset.current_test_formula)
            
            for iteration in range(number_repeat):
                #Create fresh network
                network = BFN.FIGNet(open_param_dict['num_layers'],
                                     starting_net_width,
                                     open_param_dict['strategy'],
                                     open_param_dict['batch_norm'],
                                     open_param_dict['activation'],
                                     open_param_dict['dropout'])
                
                performance_dict, curr_best_state_dict = train_and_test_single_fig(network,
                                                                                   curr_dataset,
                                                                                   None,
                                                                                   val_split,
                                                                                   test_split,
                                                                                   learning_rate,
                                                                                   lr_patience,
                                                                                   es_patience,
                                                                                   training_epochs,
                                                                                   batch_size)
                
                average_loss_list.append(performance_dict['test_loss'])
               
            this_formula_loss = sum(average_loss_list) / len(average_loss_list)
            curr_file_losses.append(this_formula_loss)
            
        curr_frame = pd.DataFrame(data=[curr_file_losses],
                                  index=[current_file],
                                  columns=possible_formula_list)
        
        #Save intermediate .csv, in case of cluter time-out or other issues
        cf_save_string = csv_output_name + '_' + str(current_file) + '.csv'
        cf_save_string = cf_save_string.replace('/', '_')
        curr_frame.to_csv(cf_save_string)
        
        
        return_frame = return_frame.append(curr_frame)
    
    #If processing hasn't timed-out, return full frame
    return_string = csv_output_name + '_FULL.csv'
    return_frame.to_csv(return_string)
    
    return return_frame

def train_and_test_single_extraction_batch(network,
                                           train_dataset,
                                           test_dataset,
                                           condition_dict,
                                           val_split,
                                           test_split,
                                           learning_rate,
                                           lr_patience,
                                           es_patience,
                                           training_epochs,
                                           batch_size):
    """
    Function to train and test a single FCExtNet under a single set of conditions.
    For this workflow, which is eventually measured by cross-validation, a set of
    extraction MS are held out in a separate test dataset, unlike the single-file
    train/val/test splitting done for FormulaInformationGain
    
    Returns a performance dictionary with several important points for plotting results

    Parameters
    ----------
    network : FCExtNet created in BitumenFCNets
        A fully connected NN that includes extraction conditions to predict the
        value of a given MS ion in an extracted sample
    train_dataset : A BitumenExtTrainingData object as created in BitumenFCNets
        Dataset containing all training data for this part of workflow
    test_dataset : A BitumenExtTestData object as created in BitumenFCNets
        Dataset containing all testing data for this part of workflow. If using a split of
        the training database for testing, this can be None.
    condition_dict : TYPE
        DESCRIPTION.
    val_split : TYPE
        DESCRIPTION.
    test_split : TYPE
        DESCRIPTION.
    learning_rate : TYPE
        DESCRIPTION.
    lr_patience : TYPE
        DESCRIPTION.
    es_patience : TYPE
        DESCRIPTION.
    training_epochs : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.

    Returns
    -------
    A dictionary that contains several lists (that can/will be saved from higher level function), that
    show training/validation stats per epoch, predicted v. actual error for all 3 sets (for each point,
    so that violin plots can be made) as well as a singular test error for ranking during optimization

    """    

    if test_dataset is not None:    
        train_index, val_index = BFN.val_train_indices(train_dataset,
                                                           val_split)
    else:
        train_index, val_index, test_index = BFN.test_train_val_split(train_dataset,
                                                                      val_split,
                                                                      test_split)

    train_sample = SubsetRandomSampler(train_index)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sample)
    
    val_sample = SubsetRandomSampler(val_index)
    val_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             sampler=val_sample)
    
    if test_dataset is not None:
        test_index = list(range(test_dataset.__len__()))
        test_sample = SubsetRandomSampler(test_index)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  sampler=test_sample)
    else:
        test_sample = SubsetRandomSampler(test_index)
        test_loader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size=batch_size,
                                                  sampler=test_sample)

    num_batches = len(train_loader)
    print('Num batches is:', num_batches)

    loss_func, optimize, lr_sched = BFN.loss_and_optim(network,
                                                       learning_rate,
                                                       lr_patience)

    training_start = time.time()

    performance_dict = {'train_act': [],                             
                        'train_pred': [],
                        'train_mse': [],
                        'val_act': [],
                        'val_pred': [],
                        'val_mse': [],
                        'test_act': [],
                        'test_pred': [],
                        'test_mse': [],
                        'train_loss_list': [],
                        'val_loss_list': []}

    best_val_loss = 1000.0
    current_patience = es_patience

    if torch.cuda.is_available() == True:
        device = torch.device('cuda')
    elif not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
        print('Using Apple MPS GPU')

    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        print('Multiple GPU Detected')
        network = nn.DataParallel(network)

    network.to(device)

    for epoch in range(training_epochs):
        current_loss = 0.0
        epoch_time = time.time()
        
        for example_tensor, example_target, condition_tensor in train_loader:
            example_target = example_target.view(-1, 1)
            
            example_tensor = example_tensor.to(device)
            example_target = example_target.to(device)
            condition_tensor = condition_tensor.to(device)
            
            optimize.zero_grad()
            
            output = network(example_tensor, condition_tensor)
            loss_amount = loss_func(output, example_target.float())
            loss_amount = loss_amount.float()
            loss_amount.backward()
            optimize.step()
            
            current_loss += float(loss_amount.item())

        if epoch % 10 == 0:
            print("Epoch {}, training_loss: {:.5f}, took: {:.2f}s".format(epoch+1, current_loss / num_batches, time.time() - epoch_time))

        performance_dict['train_loss_list'].append(current_loss / num_batches)
        #At end of epoch, try validation set on GPU

        total_val_loss = 0
        network.eval()
        with torch.no_grad():
            for val_tensor, val_target, condition_tensor in val_loader:
                val_target = val_target.view(-1, 1)

                #Send data to GPU
                val_tensor = val_tensor.to(device)
                val_target = val_target.to(device)
                condition_tensor = condition_tensor.to(device)
                
                #Forward pass only
                val_output = network(val_tensor, condition_tensor)
                val_loss_size = loss_func(val_output, val_target)
                total_val_loss += float(val_loss_size.item())
            
            if epoch % 10 == 0:
                print("Val loss = {:.4f}".format(total_val_loss / len(val_loader)))
            
            val_loss = total_val_loss / len(val_loader)
            performance_dict['val_loss_list'].append(val_loss)
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            current_best_state_dict = network.state_dict()
            current_best_epoch = epoch+1
            current_patience = es_patience
        else:
            current_patience = current_patience - 1
        
        if current_patience == 0 or (time.time() - training_start) > 84600.0:
            print('Early stopping/timeout engaged at epoch: ', epoch + 1)
            print('Best results at epoch: ', current_best_epoch)
            print("Training finished, took {:.2f}s".format(time.time() - training_start))
            network.load_state_dict(current_best_state_dict)
                        
            total_test_loss = 0
            network.eval()
            with torch.no_grad():
                for example_tensor, example_target, condition_tensor in train_loader:
                    example_target = example_target.view(-1, 1)
                    
                    train_act_list = example_target.flatten().tolist()
                    performance_dict['train_act'] += train_act_list
                    
                    example_tensor = example_tensor.to(device)
                    condition_tensor = condition_tensor.to(device)
                    train_pred = network(example_tensor, condition_tensor)
                    
                    train_pred_list = train_pred.flatten().tolist()
                    performance_dict['train_pred'] += train_pred_list
                    
                    train_mse_list = [(x - y )**2 for x, y in zip(train_act_list, train_pred_list)]
                    performance_dict['train_mse'] += train_mse_list

                for val_tensor, val_target, condition_tensor in val_loader:
                    val_target = val_target.view(-1, 1)
                    
                    val_act_list = val_target.flatten().tolist()
                    performance_dict['val_act'] += val_act_list
                    
                    val_tensor = val_tensor.to(device)
                    condition_tensor = condition_tensor.to(device)
                    val_pred = network(val_tensor, condition_tensor)
                    
                    val_pred_list = val_pred.flatten().tolist()
                    performance_dict['val_pred'] += val_pred_list
                    
                    val_mse_list = [(x - y)**2 for x, y in zip(val_act_list, val_pred_list)]
                    performance_dict['val_mse'] += val_mse_list
                
                for test_tensor, test_target, condition_tensor in test_loader:
                    test_target = test_target.view(-1, 1)
                    
                    test_act_list = test_target.flatten().tolist()
                    performance_dict['test_act'] += test_act_list
        
                    test_tensor = test_tensor.to(device)
                    test_target = test_target.to(device)
                    condition_tensor = condition_tensor.to(device)
                    
                    test_output = network(test_tensor, condition_tensor)
                    test_pred_list = test_output.flatten().tolist()
                    performance_dict['test_pred'] += test_pred_list
                    
                    test_loss_size = loss_func(test_output, test_target)
                    total_test_loss += float(test_loss_size.item())
                    
                    test_loss = total_test_loss / len(test_loader)
                    performance_dict['test_loss'] = test_loss
        
                    test_mse_list = [(x - y)**2 for x, y in zip(test_act_list, test_pred_list)]
                    performance_dict['test_mse'] += test_mse_list
        
            return performance_dict, current_best_state_dict
        
        lr_sched.step(val_loss)
        network.train()
    
    print("Training finished, took {:.4f}s".format(time.time() - training_start))
    print('Best results at epoch: ', current_best_epoch)

    total_test_loss = 0
    network.eval()
    with torch.no_grad():
        for example_tensor, example_target, condition_tensor in train_loader:
            example_target = example_target.view(-1, 1)
            
            train_act_list = example_target.flatten().tolist()
            performance_dict['train_act'] += train_act_list
            
            example_tensor = example_tensor.to(device)
            condition_tensor = condition_tensor.to(device)
            train_pred = network(example_tensor, condition_tensor)
            
            train_pred_list = train_pred.flatten().tolist()
            performance_dict['train_pred'] += train_pred_list
            
            train_mse_list = [(x - y )**2 for x, y in zip(train_act_list, train_pred_list)]
            performance_dict['train_mse'] += train_mse_list

        for val_tensor, val_target, condition_tensor in val_loader:
            val_target = val_target.view(-1, 1)
            
            val_act_list = val_target.flatten().tolist()
            performance_dict['val_act'] += val_act_list
            
            val_tensor = val_tensor.to(device)
            condition_tensor = condition_tensor.to(device)
            val_pred = network(val_tensor, condition_tensor)
            
            val_pred_list = val_pred.flatten().tolist()
            performance_dict['val_pred'] += val_pred_list
            
            val_mse_list = [(x - y)**2 for x, y in zip(val_act_list, val_pred_list)]
            performance_dict['val_mse'] += val_mse_list
        
        for test_tensor, test_target, condition_tensor in test_loader:
            test_target = test_target.view(-1, 1)
            
            test_act_list = test_target.flatten().tolist()
            performance_dict['test_act'] += test_act_list

            test_tensor = test_tensor.to(device)
            test_target = test_target.to(device)
            condition_tensor = condition_tensor.to(device)
            
            test_output = network(test_tensor, condition_tensor)
            test_pred_list = test_output.flatten().tolist()
            performance_dict['test_pred'] += test_pred_list
            
            test_loss_size = loss_func(test_output, test_target)
            total_test_loss += float(test_loss_size.item())
            
            test_loss = total_test_loss / len(test_loader)
            performance_dict['test_loss'] = test_loss

            test_mse_list = [(x - y)**2 for x, y in zip(test_act_list, test_pred_list)]
            performance_dict['test_mse'] += test_mse_list

    return performance_dict, current_best_state_dict

def measure_and_test_extraction_batch(sm_file_directory,
                                      ext_file_directory,
                                      open_param_dict,
                                      label_keys,
                                      condition_dict,
                                      holdout_list,
                                      locked_formula,
                                      val_split,
                                      training_epochs,
                                      batch_size,
                                      learning_rate,
                                      lr_patience,
                                      es_patience,
                                      output_name):
    """
    A function that examines all of the formula information gain levels for all files in a given
    extraction file directory, using all possible formula in the possible_formula_list, in combination
    with any locked_formula.

    As the extraction files are evaluated one-at-a-time for this analysis, a single file is split
    into training/validation/test sets. Given that some splits may be easier/harder to analyze,
    the analysis is repeated multiple times and averaged.

    Parameters
    ----------
    sm_file_directory : string
        Name of sub-directory that contains starting material MS files
    ext_file_directory : string
        Name of sub-directory that contains extraction MS files
    open_param_dict : dictionary
        A dictionary that contains the necessary description to create a fully connected NN
    label_keys : dictionary
        A dictionary that tells which extracted fraction came from which starting material
    condition_dict : dictionary
        A dictionary that holds the solvent blend information for each experiment
    holdout_list : list
        Name of files that should be excluded from creation of the training dataset
    locked_formula : list of tuples
        A list of formula that are included in every analysis, +/- the new formula being tested
    val_split : float
        The fraction of MS peaks that will be used for validation
    training_epochs : integer
        Maximum number of training epochs that will be used
    batch_size : integer
        Batch size for data loading
    learning_rate : float
        Initial learning rate used during ML training
    lr_patience : integer
        Number of training epochs with no improvement that will be allowed before learning rate is reduced
    es_patience : integer
        Number of training epochs with no improvement that will be allowed before early stopping is engaged
    output_name : string
        Name of training/testing cycle that will be stamped on the saved state dictionary and performance dictionary
    Returns
    -------
    A state dictionary and perfomance dictionary are pickled for future use/processing

    """

    #Create datasets and network
    train_dataset = BFN.BitumenExtTrainingData(sm_file_directory=sm_file_directory,
                                               ext_file_directory=ext_file_directory,
                                               label_keys=label_keys,
                                               test_list=holdout_list,
                                               locked_formula=locked_formula,
                                               condition_dict=condition_dict,
                                               output_name=output_name)
    
    test_dataset = BFN.BitumenExtTestData(sm_file_directory=sm_file_directory,
                                          ext_file_directory=ext_file_directory,
                                          label_keys=label_keys,
                                          test_list=holdout_list,
                                          locked_formula=locked_formula,
                                          condition_dict=condition_dict,
                                          output_name=output_name)
    
    starting_net_width = 12 + len(train_dataset.current_test_formula)
    
    network = BFN.FCExtNet(open_param_dict['num_layers'],
                           starting_net_width,
                           open_param_dict['strategy'],
                           open_param_dict['batch_norm'],
                           open_param_dict['activation'],
                           open_param_dict['dropout'])

    performance_dict, best_state_dict = train_and_test_single_extraction_batch(network,
                                                                               train_dataset,
                                                                               test_dataset,
                                                                               condition_dict,
                                                                               val_split,
                                                                               learning_rate,
                                                                               lr_patience,
                                                                               es_patience,
                                                                               training_epochs,
                                                                               batch_size) 

    dict_string = output_name + 'best_state_dict.pt'
    torch.save(best_state_dict, dict_string)

    pickle_string = output_name + 'best_network_performance.pkl'
    pickle_file = open(pickle_string, 'wb')
    pickle.dump(performance_dict, pickle_file)
    pickle_file.close()
                       
    return

def test_possible_extraction_formula(sm_file_directory,
                                     ext_file_directory,
                                     open_param_dict,
                                     label_keys,
                                     condition_dict,
                                     holdout_list,
                                     locked_formula,
                                     possible_formula_list,
                                     val_split,
                                     test_split,
                                     training_epochs,
                                     batch_size,
                                     learning_rate,
                                     lr_patience,
                                     es_patience,
                                     output_name):
    """
    A function that examines all of the formula information gain levels for all files in a given
    extraction file directory, using all possible formula in the possible_formula_list, in combination
    with any locked_formula.

    As the extraction files are evaluated one-at-a-time for this analysis, a single file is split
    into training/validation/test sets. Given that some splits may be easier/harder to analyze,
    the analysis is repeated multiple times and averaged.

    Parameters
    ----------
    sm_file_directory : string
        Name of sub-directory that contains starting material MS files
    ext_file_directory : string
        Name of sub-directory that contains extraction MS files
    open_param_dict : dictionary
        A dictionary that contains the necessary description to create a fully connected NN
    label_keys : dictionary
        A dictionary that tells which extracted fraction came from which starting material
    condition_dict : dictionary
        A dictionary that holds the solvent blend information for each experiment
    holdout_list : list
        Name of files that should be excluded from creation of the training dataset
    locked_formula : list of tuples
        A list of formula that are included in every analysis, +/- the new formula being tested
    possible_formula : list of tuples
        Additional formula adjustments that will be tested individually for prediction performance
    val_split : float
        The fraction of MS peaks that will be used for validation
    test_split : float
        The fraction of MS peaks that will be used for testing. Testing different formula happens before
        final learning rate/CV testing, so use a split of the training data.
    training_epochs : integer
        Maximum number of training epochs that will be used
    batch_size : integer
        Batch size for data loading
    learning_rate : float
        Initial learning rate used during ML training
    lr_patience : integer
        Number of training epochs with no improvement that will be allowed before learning rate is reduced
    es_patience : integer
        Number of training epochs with no improvement that will be allowed before early stopping is engaged
    output_name : string
        Name of training/testing cycle that will be stamped on the saved state dictionary and performance dictionary
    Returns
    -------
    A state dictionary and perfomance dictionary are pickled for future use/processing

    """

    #Create datasets one time only
    train_dataset = BFN.BitumenExtTrainingData(sm_file_directory=sm_file_directory,
                                               ext_file_directory=ext_file_directory,
                                               label_keys=label_keys,
                                               test_list=holdout_list,
                                               locked_formula=locked_formula,
                                               condition_dict=condition_dict,
                                               output_name=output_name)
    
    #Final CV testing on held out MS happens after all formula have been tested, so a split of
    #the training dataset is used here rather than the final testing dataset.
    test_dataset = None

    current_best_test_loss = 100000.0
    current_best_state_dict = None
    current_best_performance_dict = None
    current_formula_losses = []
    
    for new_formula in possible_formula_list:
        BFN.update_formula(train_dataset,
                           new_formula,
                           locked_formula)
    
        starting_net_width = 12 + len(train_dataset.current_test_formula)
        
        network = BFN.FCExtNet(open_param_dict['num_layers'],
                               starting_net_width,
                               open_param_dict['strategy'],
                               open_param_dict['batch_norm'],
                               open_param_dict['activation'],
                               open_param_dict['dropout'])
    
        performance_dict, output_state_dict = train_and_test_single_extraction_batch(network,
                                                                                     train_dataset,
                                                                                     test_dataset,
                                                                                     condition_dict,
                                                                                     val_split,
                                                                                     test_split,
                                                                                     learning_rate,
                                                                                     lr_patience,
                                                                                     es_patience,
                                                                                     training_epochs,
                                                                                     batch_size) 

        current_formula_losses.append(performance_dict['test_loss'])
        
        if performance_dict['test_loss'] < current_best_test_loss:
            current_best_test_loss = performance_dict['test_loss']
            current_best_state_dict = output_state_dict
            current_best_performance_dict = performance_dict

    curr_frame = pd.DataFrame(data=[current_formula_losses],
                              index=[output_name],
                              columns=possible_formula_list)
    
    cf_save_string = output_name + '_per_formula_loss' + '.csv'
    cf_save_string = cf_save_string.replace('/', '_')
    curr_frame.to_csv(cf_save_string)
       
    dict_string = output_name + '.pt'
    torch.save(current_best_state_dict, dict_string)

    pickle_string = output_name + '.pkl'
    pickle_file = open(pickle_string, 'wb')
    pickle.dump(current_best_performance_dict, pickle_file)
    pickle_file.close()
                           
    return

def train_and_test_single_neighbor_cycle(network,
                                         dataset,
                                         val_split,
                                         test_split,
                                         training_epochs,
                                         batch_size,
                                         learning_rate,
                                         lr_patience,
                                         es_patience):
    """
    

    Parameters
    ----------
    network : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    val_split : TYPE
        DESCRIPTION.
    test_split : TYPE
        DESCRIPTION.
    training_epochs : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    learning_rate : TYPE
        DESCRIPTION.
    lr_patience : TYPE
        DESCRIPTION.
    es_patience : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    train_index, val_index, test_index = BFN.test_train_val_split(dataset,
                                                                  val_split,
                                                                  test_split)

    train_sample = SubsetRandomSampler(train_index)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=train_sample)
    
    val_sample = SubsetRandomSampler(val_index)
    val_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=val_sample)
    
    test_sample = SubsetRandomSampler(test_index)
    test_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=test_sample)
    
    device = mps_vs_cuda()
    
    loss_func, optimize, lr_sched = BFN.multi_loss_and_optim(network,
                                                             learning_rate,
                                                             lr_patience)

    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        print('Multiple GPU Detected')
        network = nn.DataParallel(network)

    network.to(device)

    training_start = time.time()


    best_val_loss = 1000.0
    current_patience = es_patience

    performance_dict = {'train_act': [],                             
                        'train_pred': [],
                        'train_mse': [],
                        'val_act': [],
                        'val_pred': [],
                        'val_mse': [],
                        'test_act': [],
                        'test_pred': [],
                        'test_mse': [],
                        'train_loss_list': [],
                        'val_loss_list': []}

    train_batches = len(train_loader)
    val_batches = len(val_loader)
    test_batches = len(test_loader)

    for epoch in range(training_epochs):
        current_loss = 0.0
        
        for index, data in enumerate(train_loader):
            neighbor_example_tensor, example_target = data
            example_target = example_target.to(device)
            # example_target = example_target.long()
            neighbor_example_tensor = neighbor_example_tensor.to(device)
            
            optimize.zero_grad()
            
            output = network(neighbor_example_tensor)
            loss_amount = loss_func(output, example_target)
            loss_amount = loss_amount.float()
            loss_amount.backward()
            optimize.step()
            
            current_loss += float(loss_amount.item())
            
        performance_dict['train_loss_list'].append(current_loss / train_batches)
        #At end of epoch, try validation set on GPU

        total_val_loss = 0
        network.eval()
        with torch.no_grad():
            for val_nn_tensor, val_target in val_loader:

                #Send data to GPU
                val_nn_tensor = val_nn_tensor.to(device)
                val_target = val_target.to(device)
                # val_target = val_target.long()
                
                #Forward pass only
                val_output = network(val_nn_tensor)
                val_loss_size = loss_func(val_output, val_target)
                total_val_loss += float(val_loss_size.item())
        
            val_loss = total_val_loss / val_batches
            
            performance_dict['val_loss_list'].append(val_loss)
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            current_best_state_dict = network.state_dict()
            current_best_epoch = epoch+1
            current_patience = es_patience
        else:
            current_patience = current_patience - 1
        
        if current_patience == 0 or (time.time() - training_start) > 84600.0:
            print('Early stopping/timeout engaged at epoch: ', epoch + 1)
            print('Best results at epoch: ', current_best_epoch)
            print("Training finished, took {:.2f}s".format(time.time() - training_start))
            network.load_state_dict(current_best_state_dict)
                        
            total_test_loss = 0
            network.eval()
            with torch.no_grad():
                for example_tensor, example_target in train_loader:
                    train_act_list = example_target.flatten().tolist()
                    performance_dict['train_act'] += train_act_list
                    
                    example_tensor = example_tensor.to(device)
                    train_pred = network(example_tensor)
                    
                    train_pred_list = train_pred.flatten().tolist()
                    performance_dict['train_pred'] += train_pred_list
                    
                    train_mse_list = [(x - y )**2 for x, y in zip(train_act_list, train_pred_list)]
                    performance_dict['train_mse'] += train_mse_list

                for val_tensor, val_target in val_loader:
                    val_act_list = val_target.flatten().tolist()
                    performance_dict['val_act'] += val_act_list
                    
                    val_tensor = val_tensor.to(device)
                    val_pred = network(val_tensor)
                    
                    val_pred_list = val_pred.flatten().tolist()
                    performance_dict['val_pred'] += val_pred_list
                    
                    val_mse_list = [(x - y)**2 for x, y in zip(val_act_list, val_pred_list)]
                    performance_dict['val_mse'] += val_mse_list
                
                for test_tensor, test_target in test_loader:
                    test_act_list = test_target.flatten().tolist()
                    performance_dict['test_act'] += test_act_list
        
                    test_tensor = test_tensor.to(device)
                    test_target = test_target.to(device)
                    
                    test_output = network(test_tensor)
                    test_pred_list = test_output.flatten().tolist()
                    performance_dict['test_pred'] += test_pred_list
                    
                    test_loss_size = loss_func(test_output, test_target)
                    total_test_loss += float(test_loss_size.item())
                    
                    test_loss = total_test_loss / len(test_loader)
                    performance_dict['test_loss'] = test_loss
        
                    test_mse_list = [(x - y)**2 for x, y in zip(test_act_list, test_pred_list)]
                    performance_dict['test_mse'] += test_mse_list
        
            return performance_dict, current_best_state_dict
        
        lr_sched.step(val_loss)
        network.train()
    
    print("Training finished, took {:.4f}s".format(time.time() - training_start))
    print('Best results at epoch: ', current_best_epoch)

    total_test_loss = 0
    network.eval()
    with torch.no_grad():
        for example_tensor, example_target in train_loader:         
            train_act_list = example_target.flatten().tolist()
            performance_dict['train_act'] += train_act_list
            
            example_tensor = example_tensor.to(device)
            train_pred = network(example_tensor)
            
            train_pred_list = train_pred.flatten().tolist()
            performance_dict['train_pred'] += train_pred_list
            
            train_mse_list = [(x - y )**2 for x, y in zip(train_act_list, train_pred_list)]
            performance_dict['train_mse'] += train_mse_list

        for val_tensor, val_target in val_loader:            
            val_act_list = val_target.flatten().tolist()
            performance_dict['val_act'] += val_act_list
            
            val_tensor = val_tensor.to(device)
            val_pred = network(val_tensor)
            
            val_pred_list = val_pred.flatten().tolist()
            performance_dict['val_pred'] += val_pred_list
            
            val_mse_list = [(x - y)**2 for x, y in zip(val_act_list, val_pred_list)]
            performance_dict['val_mse'] += val_mse_list
        
        for test_tensor, test_target in test_loader:            
            test_act_list = test_target.flatten().tolist()
            performance_dict['test_act'] += test_act_list

            test_tensor = test_tensor.to(device)
            test_target = test_target.to(device)
            
            test_output = network(test_tensor)
            test_pred_list = test_output.flatten().tolist()
            performance_dict['test_pred'] += test_pred_list
            
            test_loss_size = loss_func(test_output, test_target)
            total_test_loss += float(test_loss_size.item())
            
            test_loss = total_test_loss / test_batches
            performance_dict['test_loss'] = test_loss

            test_mse_list = [(x - y)**2 for x, y in zip(test_act_list, test_pred_list)]
            performance_dict['test_mse'] += test_mse_list

    return performance_dict, current_best_state_dict

def optimize_final_ext_network(sm_file_directory,
                               ext_file_directory,
                               open_param_dict,
                               label_keys,
                               condition_dict,
                               holdout_list,
                               locked_formula,
                               val_split,
                               test_split,
                               training_epochs,
                               batch_size,
                               possible_learning_rates,
                               lr_patience,
                               es_patience,
                               output_name):
    """
    A final workflow item to optimize Depth3 cross-validated networks. Lots of previous work
    found very consistent results for network shape/size and dropout levels, so these are
    no longer optimized. Learning rate only.
    
    Parameters
    ----------
    sm_file_directory : string
        Name of sub-directory that contains starting material MS files
    ext_file_directory : string
        Name of sub-directory that contains extraction MS files
    open_param_dict : dictionary
        A dictionary that contains the necessary description to create a fully connected NN
    label_keys : dictionary
        A dictionary that tells which extracted fraction came from which starting material
    condition_dict : dictionary
        A dictionary that holds the solvent blend information for each experiment
    holdout_list : list
        Name of files that should be excluded from creation of the training dataset
    locked_formula : list of tuples
        A list of formula that are included in every analysis, +/- the new formula being tested
    val_split : float
        The fraction of MS peaks that will be used for validation
    test_split : float
        The fraction of MS peaks that will be used for testing. Testing different formula happens before
        final learning rate/CV testing, so use a split of the training data.
    training_epochs : integer
        Maximum number of training epochs that will be used
    batch_size : integer
        Batch size for data loading
    possible_learning_rates : list of floats
        Initial learning rate used during ML training, parameter being optimized
    lr_patience : integer
        Number of training epochs with no improvement that will be allowed before learning rate is reduced
    es_patience : integer
        Number of training epochs with no improvement that will be allowed before early stopping is engaged
    output_name : string
        Name of training/testing cycle that will be stamped on the saved state dictionary and performance dictionary
    Returns
    -------
    A state dictionary and perfomance dictionary are pickled for future use/processing

    """

    #Create datasets one time only
    train_dataset = BFN.BitumenExtTrainingData(sm_file_directory=sm_file_directory,
                                               ext_file_directory=ext_file_directory,
                                               label_keys=label_keys,
                                               test_list=holdout_list,
                                               locked_formula=locked_formula,
                                               condition_dict=condition_dict,
                                               output_name=output_name)
    
    #Final CV testing on held out MS happens after all formula have been tested, so a split of
    #the training dataset is used here rather than the final testing dataset.
    test_dataset = None

    #Only need to update formula one time

    current_best_test_loss = 100000.0
    current_best_state_dict = None
    current_best_performance_dict = None
    current_formula_losses = []
    
    for learning_rate in possible_learning_rates:
        starting_net_width = 12 + len(train_dataset.current_test_formula)
        
        network = BFN.FCExtNet(open_param_dict['num_layers'],
                               starting_net_width,
                               open_param_dict['strategy'],
                               open_param_dict['batch_norm'],
                               open_param_dict['activation'],
                               open_param_dict['dropout'])
    
        performance_dict, output_state_dict = train_and_test_single_extraction_batch(network,
                                                                                     train_dataset,
                                                                                     test_dataset,
                                                                                     condition_dict,
                                                                                     val_split,
                                                                                     test_split,
                                                                                     learning_rate,
                                                                                     lr_patience,
                                                                                     es_patience,
                                                                                     training_epochs,
                                                                                     batch_size) 

        current_formula_losses.append(performance_dict['test_loss'])
        
        if performance_dict['test_loss'] < current_best_test_loss:
            current_best_test_loss = performance_dict['test_loss']
            current_best_state_dict = output_state_dict
            current_best_performance_dict = performance_dict

    curr_frame = pd.DataFrame(data=[current_formula_losses],
                              index=[output_name],
                              columns=possible_learning_rates)
    
    cf_save_string = output_name + '_lr_optimized' + '.csv'
    cf_save_string = cf_save_string.replace('/', '_')
    curr_frame.to_csv(cf_save_string)
       
    dict_string = output_name + '.pt'
    torch.save(current_best_state_dict, dict_string)

    pickle_string = output_name + '.pkl'
    pickle_file = open(pickle_string, 'wb')
    pickle.dump(current_best_performance_dict, pickle_file)
    pickle_file.close()
                           
    return

def random_search_opt_neighbor(sm_file_directory,
                               ext_file_directory,
                               search_param_dict,
                               label_keys,
                               condition_dict,
                               test_list,
                               num_neighbors,
                               val_split,
                               test_split,
                               training_epochs,
                               batch_size,
                               lr_patience,
                               es_patience,
                               num_searches,
                               output_name):
    """
    An optimization strategy that uses random search to find the best nearest-neighbor
    search strategy from the list of conditions provided in search_param_dict

    Parameters
    ----------
    sm_file_directory : TYPE
        DESCRIPTION.
    ext_file_directory : TYPE
        DESCRIPTION.
    search_param_dict : TYPE
        DESCRIPTION.
    label_keys : TYPE
        DESCRIPTION.
    condition_dict : TYPE
        DESCRIPTION.
    test_list : TYPE
        DESCRIPTION.
    num_neighbors : interger
    val_split : TYPE
        DESCRIPTION.
    test_split : TYPE
        DESCRIPTION.
    training_epochs : TYPE
        DESCRIPTION.
    batch_size : TYPE
        DESCRIPTION.
    lr_patience : TYPE
        DESCRIPTION.
    es_patience : TYPE
        DESCRIPTION.
    num_searches : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    tested_combinations = []
    
    #Create full dataset
    dataset = BFN.NeighborExtFullDataset(sm_file_directory,
                                         ext_file_directory,
                                         label_keys,
                                         condition_dict,
                                         test_list,
                                         num_neighbors,
                                         output_name)

    current_best_test_loss = 100000.0
    current_best_state_dict = None
    current_best_performance_dict = None
    current_loss_per_fingerprint = []

    
    for current_opt_run in range(num_searches):
        possible_curr_conditions, cond_fingerprint = generate_search_tuple(search_param_dict)
        while cond_fingerprint in tested_combinations:
            print('Found repeat! At run: ', current_opt_run)
            possible_curr_conditions, cond_fingerprint = generate_search_tuple(search_param_dict)
        
        tested_combinations.append(cond_fingerprint) 
        print('Launching with:', possible_curr_conditions)
        
        network = BFN.NeighborExtNet(layer_size_list=possible_curr_conditions['layer_list'],
                                     file_processing_list=dataset.file_processing_list,
                                     batch_norm=possible_curr_conditions['batch_norm'],
                                     softmax=possible_curr_conditions['softmax'],
                                     activation=nn.ReLU(),
                                     dropout_amt=possible_curr_conditions['dropout'])
    
        performance_dict, output_state_dict = train_and_test_single_neighbor_cycle(network,
                                                                                   dataset,
                                                                                   val_split,
                                                                                   test_split,
                                                                                   training_epochs,
                                                                                   batch_size,
                                                                                   possible_curr_conditions['learning_rate'],
                                                                                   lr_patience,
                                                                                   es_patience)
    
        current_loss_per_fingerprint.append(performance_dict['test_loss'])
        
        if performance_dict['test_loss'] < current_best_test_loss:
            current_best_test_loss = performance_dict['test_loss']
            current_best_state_dict = output_state_dict
            current_best_performance_dict = performance_dict

    curr_frame = pd.DataFrame(data=[current_loss_per_fingerprint],
                              index=[output_name],
                              columns=tested_combinations)
    
    cf_save_string = output_name + '_random_optimized' + '.csv'
    cf_save_string = cf_save_string.replace('/', '_')
    curr_frame.to_csv(cf_save_string)
       
    dict_string = output_name + '.pt'
    torch.save(current_best_state_dict, dict_string)

    pickle_string = output_name + '.pkl'
    pickle_file = open(pickle_string, 'wb')
    pickle.dump(current_best_performance_dict, pickle_file)
    pickle_file.close()
                           
    return

def generate_search_tuple(search_param_dict):
    """
    A function that takes a search parameter dictionary, which includes a leading entry
    that locks the parameter order, and returns a tuple of random numbers that correspond
    to the parameter settings to be used in a given optimization run.
    
    *Special processing is needed for num_layers + layer_width such that each layer is
    not forced to be identical, and these variables always need to be present/absent together*
    
    *If a set of locked layer depth/width combinations is going to be tested, locked_layers takes
    precedence over num_layers and layer_width*

    Parameters
    ----------
    search_param_dict : dictionary
        A dictionary that contains a leading entry called 'opt_params', which sets the order
        of optimization parameters, and then has a key for each parameter that points to a
        tuple which holds possible values.

    Returns
    -------
    A dictionary that contains the settings for each specific parameter of interest, and a 
    fingerprint tuple to prevent searching identical conditions

    """

    parameter_list = search_param_dict['opt_params']
    setting_dictionary = {}
    
    for specific_param in parameter_list:
        if specific_param == 'layer_list':
            if 'locked_layers' in search_param_dict.keys():
                layer_choice = random.randint(0, (len(search_param_dict['locked_layers']) - 1))
                layer_tuple = tuple(search_param_dict['locked_layers'][layer_choice])
                setting_dictionary['layer_list'] = layer_tuple
            else:
                num_layer_choice = random.randint(0, (len(search_param_dict['num_layers']) - 1))
                set_num_layers = search_param_dict['num_layers'][num_layer_choice]
                setting_dictionary['num_layers'] = set_num_layers
                layer_list = []
                for specific_layer in range(set_num_layers):
                    this_layer_choice = random.randint(0, (len(search_param_dict['layer_width']) - 1))
                    layer_list.append(search_param_dict['layer_width'][this_layer_choice])
                    layer_tuple = tuple(layer_list)
                    setting_dictionary['layer_list'] = layer_tuple
        
        else:
            param_length = len(search_param_dict[specific_param])
            this_run_int = random.randint(0, (param_length - 1))
            setting_dictionary[specific_param] = search_param_dict[specific_param][this_run_int]

    fingerprint_tuple = create_opt_run_fingerprint(setting_dictionary,
                                                   parameter_list)

    return setting_dictionary, fingerprint_tuple

def create_opt_run_fingerprint(setting_dictionary,
                               parameter_list):
    """
    Helper function which takes a dictionary for an optimization run, and an ordered tuple of variables,
    and returns a fingerprint tuple of conditions that can be used during random search to ensure that
    duplicate settings are not tested.

    Parameters
    ----------
    setting_dictionary : dictionary
        A dictionary containing a specific set of network conditions for an optimization run
    parameter_list : tuple
        An ordered tuple of parameters that sets the order of entries for the fingerprint

    Returns
    -------
    A tuple that is a fingerprint of the optimization conditions

    """

    fingerprint_list = []
    for specific_param in parameter_list:
        fingerprint_list.append(setting_dictionary[specific_param])
    
    fingerprint_tuple = tuple(fingerprint_list)
    
    return fingerprint_tuple

def mps_vs_cuda():
    """
    Helper function that determines whether to use 'mps' or 'cuda' backend

    Returns
    -------
    'cuda' is available, 'mps' if available, otherwise raises error.

    """
    if torch.cuda.is_available() == True:
        device = torch.device('cuda')
    elif not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        device = torch.device("mps")
        print('Using Apple MPS GPU')

    return device
    
def all_neighbor_avg_test_from_file(trained_net_directory,
                                    trained_net_name,
                                    trained_net_param_dict,
                                    dataset_param_dict,
                                    output_name,
                                    csv_output):
    """
    Function for evaluating the performance of a trained network from its saved .state_dict()

    Parameters
    ----------
    trained_net_directory : TYPE
        DESCRIPTION.
    trained_net_name : TYPE
        DESCRIPTION.
    trained_net_param_dict : TYPE
        DESCRIPTION.
    dataset_param_dict : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.
    csv_output : Boolean
        If 'True', a .csv file with mse results is created
    Returns
    -------
    The top level mse and ppe dictionaries, and also saves these as pickle files with output_name

    """
    neighbor_dataset = BFN.NeighborExtFullDataset(dataset_param_dict['sm_file_directory'],
                                                  dataset_param_dict['ext_file_directory'],
                                                  dataset_param_dict['label_keys'],
                                                  dataset_param_dict['condition_dict'],
                                                  dataset_param_dict['test_list'],
                                                  dataset_param_dict['num_neighbors'],
                                                  output_name)
    
    trained_network = BFN.load_neighbor_network(trained_net_directory,
                                                trained_net_name,
                                                trained_net_param_dict,
                                                neighbor_dataset)
    
    mse_dict, ppe_dict = all_neighbor_avg_test(trained_network,
                                               neighbor_dataset)
    
    mse_string = output_name + 'mse_dict.pkl'
    mse_file = open(mse_string, 'wb')
    pickle.dump(mse_dict, mse_file)
    mse_file.close()

    ppe_string = output_name + 'ppe_dict.pkl'
    ppe_file = open(ppe_string, 'wb')
    pickle.dump(ppe_dict, ppe_file)
    ppe_file.close()

    if csv_output == True:
        mse_dict_to_csv_output(mse_dict,
                               output_name)

    return mse_dict, ppe_dict    

def all_neighbor_avg_test(network,
                          dataset):
    """
    A function that described the MSE and PPE for every different neighbor distance for a
    trained network on a test dataset
    
    Parameters
    ----------
    network : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    output_name : string
        Leading string that will be used when exporting exhaustive neighbor measurement as .csv file

    Returns
    -------
    The performance dictionaries and saves the performance output as a .pkl

    """

    #Load state dictionary outside of this function
    avg_result_dict = {}
    per_point_result_dict = {}
    test_file_list = list(dataset.test_dict.keys())
    num_possible_neighbors = len(dataset.file_processing_list)
    
    for curr_test_file in test_file_list:
        avg_result_dict[curr_test_file] = {}
        per_point_result_dict[curr_test_file] = {}
        single_file_avg, single_file_ppe = single_file_neigh_dist_test(network,
                                                                       dataset,
                                                                       curr_test_file,
                                                                       num_possible_neighbors)
        for result_entry in single_file_avg.keys():
            avg_result_dict[curr_test_file][result_entry] = single_file_avg[result_entry]
            per_point_result_dict[curr_test_file][result_entry] = single_file_ppe[result_entry]

    return avg_result_dict, per_point_result_dict

def single_file_neigh_dist_test(network,
                                dataset,
                                curr_test_file,
                                num_possible_neighbors):
    """
    A function that takes a single test extraction file, and tests the accuracy
    of all possible num_neighbor choices, and returns 2 things: the MSE of each
    num_neighbors for the whole file, as well as a list of per-point errors (for box/violin plot)

    Parameters
    ----------
    network : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.
    curr_test_file : TYPE
        DESCRIPTION.
    num_possible_neighbors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    avg_result_dict = {}
    per_point_result_dict = {}
    network.eval()
    
    #Generate blank lists for PPE dict
    for specific_pred in range(num_possible_neighbors):
        key_string = 'num_neigh=' + str(specific_pred + 1)
        per_point_result_dict[key_string] = []
        
    for possible_point in dataset.full_test_getitem:
        if possible_point[0] == curr_test_file:
            #Get actual value
            if possible_point[3] == 'app' or possible_point[3] == 'per':
                act_val = dataset.test_dict[curr_test_file][2][possible_point[2]]
            else:
                act_val = 0.0
            
            #Assemble ordered list of predition values. Need to use try/except
            train_list = []
            for train_file in dataset.file_processing_list:
                try:
                    train_list.append(dataset.training_dict[train_file][2][possible_point[2]])
                except:
                    train_list.append(0.0)
            
            #Get tensor prediction
            try:
                pred_neighbors = network(possible_point[4]).tolist()
            except:
                pred_neighbors = network(possible_point[4].unsqueeze(0)).tolist()[0]
            seq = sorted(pred_neighbors)
            ranks = [seq.index(val) for val in pred_neighbors]
            for specific_pred in range(num_possible_neighbors):
                #Assemble list of looked-up nearest neighbor values from ranked list
                key_string = 'num_neigh=' + str(specific_pred + 1)
                target_list = [train_list[ranks.index(rank)] if rank >= (num_possible_neighbors - specific_pred - 1) else 0.0 for rank in ranks]
                total_pred = sum(target_list)
                avg_pred = total_pred / (specific_pred + 1)
                sq_error = (act_val - avg_pred)**2
                per_point_result_dict[key_string].append(sq_error)
    
    #Generate averages
    for num_neighbor_test in per_point_result_dict.keys():
        average_e = sum(per_point_result_dict[num_neighbor_test]) / len(per_point_result_dict[num_neighbor_test])
        avg_result_dict[num_neighbor_test] = average_e
    
    return avg_result_dict, per_point_result_dict

def mse_dict_to_csv_output(mse_dict,
                           output_name):
    """
    Function for recording data that turns a mse_dict output into a .csv file
    Done as a separate function so that data isn't always output

    Parameters
    ----------
    mse_dict : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.

    Returns
    -------
    None, but saves desired .csv output

    """                

    return_frame = pd.DataFrame.from_dict(mse_dict, orient='index')
    mse_save_string = output_name + '_mse_perf.csv'
    return_frame.to_csv(mse_save_string)

    return
