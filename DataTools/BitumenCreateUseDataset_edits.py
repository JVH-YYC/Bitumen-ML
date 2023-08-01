#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:47:17 2021

@author: jvh

Tools for working with MS Data (excluding creating dictionaries)


"""

from operator import itemgetter
import random
import torch
import numpy as np
from pathlib import Path
import pickle

### New getitem_list creation functions. First, create the total_ion_list, then create a full getitem list that
### includes looking for each ion in total_ion_list in each processing file. This introduces a new ion type
### which is the 'oth' ion - an ion that is in another extraction, but not in the current extraction or SM

def create_total_ion_set(training_dict,
                          output_name):
    """
    A function that looks at all extraction files in the training dictionary, and creates a set of
    all ions that are present in the entire training dictionary. It saves the set as a pickled object
    with the output_name as the leading string, to which is appended '_total_ion_set.pkl'
    
    Also creates a normalization tuple-of-tuples, which is formatted as: ((min #C, max #C), (min #H, max #H), etc)
    
    Parameters
    ----------
    training_dict : dictionary
        A dictionary of dictionaries. The top level keys are file names of the original
        .csv files of MS data. Each key points to a dictionary, which has tuple keys of
        the form (#C, #H, #N, #O, #S). Those keys have a single entry,
        the normalized MS intensity of the peak corresponding to that formula.
    output_name : dictionary
        A leading string that will be used to stamp the pickled output file,
        containing the total ion set

    Returns
    -------
    The total ion set

    """
    
    total_ion_set = set()
    
    for file_key in training_dict.keys():
        if 'SM' not in training_dict[file_key][0]:
            curr_sm_label = training_dict[file_key][0] + '_SM'
            for tuple_key in training_dict[file_key][2].keys():
                if tuple_key not in total_ion_set:
                    total_ion_set.add(tuple_key)
    
    #One pass through starting material to detect any ions that disappear during extraction
    for try_top_level in training_dict.keys():
        if curr_sm_label in try_top_level:
            sm_location = try_top_level
    
    for sm_tuple_key in training_dict[sm_location][2].keys():
        if sm_tuple_key not in total_ion_set:
            total_ion_set.add(sm_tuple_key)
    
    total_ion_string = output_name + 'total_ion_set.pkl'
    pickle_file = open(total_ion_string, 'wb')
    pickle.dump(total_ion_set, pickle_file)
    pickle_file.close()
    
    #Create min/max double tuple
    #Grab random ion
    
    random_example = random.sample(total_ion_set, 1)[0]
    normalization_list = []
    for atom_position in range(len(random_example)):
        min_val = min(total_ion_set,key=itemgetter(atom_position))[atom_position]
        max_val = max(total_ion_set,key=itemgetter(atom_position))[atom_position]
        norm_tup = (min_val, max_val)
        normalization_list.append(norm_tup)
    
    normalization_tuple = tuple(normalization_list)
    
    return total_ion_set, normalization_tuple

def create_extraction_set_via_tis(training_dict,
                                  total_ion_set,
                                  output_name):
    """
    An updated function to create the full __getitem__() list that uses the
    total_ion_set as a starting point. As compared to previous versions of this
    function, the ML approach will now be trained on many ions that were absent
    in the starting material *and* in the extraction of interest - now labelled
    as 'oth' aka other ions.

    Parameters
    ----------
    training_dict : dictionary
        A dictionary of dictionaries. The top level keys are file names of the original
        .csv files of MS data. Each key points to a dictionary, which has tuple keys of
        the form (#C, #H, #N, #O, #S). Those keys have a single entry,
        the normalized MS intensity of the peak corresponding to that formula.
    total_ion_set : set
        A set that contains all observed tuples in the entire training set
    output_name : string
        A string that will become the leading part of the filename for the
        total_ion_dictionary, and will be appended with '_total_ion_dict.pkl'

    Returns
    -------
    The getitem_list that will be used for training and the observed ion dictionary. 

    """

    getitem_list = []
    observed_ion_dict = {}
    
    for file_key in training_dict.keys():
        observed_tuples = set()
        if 'SM' not in training_dict[file_key][0]:
            curr_sm_label = training_dict[file_key][0] + '_SM'
            #Inefficient loop over file names, but there are currently never more than 50. Figure out
            #how to do better if nececssary in future
            for try_top_level in training_dict.keys():
                if curr_sm_label in try_top_level:
                    sm_location = try_top_level
  
            for possible_ion in total_ion_set:
                if possible_ion in training_dict[file_key][1].keys():
                    observed_tuples.add(possible_ion)
                    if possible_ion in training_dict[sm_location][1].keys():
                        getitem_list.append((file_key, training_dict[file_key][0], possible_ion, 'per'))
                    else:
                        getitem_list.append((file_key, training_dict[file_key][0], possible_ion, 'app'))
                else:
                    if possible_ion in training_dict[sm_location][1].keys():
                        getitem_list.append((file_key, training_dict[file_key][0], possible_ion, 'dis'))
                    else:
                        getitem_list.append((file_key, training_dict[file_key][0], possible_ion, 'oth'))
                        
        observed_ion_dict[file_key] = observed_tuples

    ion_dict_string = output_name + 'observed_ion_dict.pkl'
    pickle_file = open(ion_dict_string, 'wb')
    pickle.dump(observed_ion_dict, pickle_file)
    pickle_file.close()

    return getitem_list, observed_ion_dict

def create_neighbor_set_via_tis(dataset_dict,
                                total_ion_set,
                                output_name):
    """
    Essentially identical to create_extraction_set_via_tis above, just with a different
    position in the dataset_dict for reading the ions.
    During the creation of the dataset, the order in which training files was read is
    locked into the file_processing_list, which is critical for further functions.    

    Parameters
    ----------
    dataset_dict : dictionary
        A dictionary of dictionaries. The top level keys are file names of the original
        .csv files of MS data. Each key points to a dictionary, which has tuple keys of
        the form (#C, #H, #N, #O, #S). Those keys have a single entry,
        the normalized MS intensity of the peak corresponding to that formula.
    total_ion_set : set
        A set that contains all observed tuples in the entire training set
    normalization_tuple : tuple of tuples
        Of the format ((min #C, max #C), (min #H, max #H), etc.)
    output_name : string
        A string that will become the leading part of the filename for the
        total_ion_dictionary, and will be appended with '_total_ion_dict.pkl'

    Returns
    -------
    The getitem_list that will be used for training (after expansion), 
    the observed ion dictionary and the file_processing_list. 

    """

    getitem_list = []
    observed_ion_dict = {}
    file_processing_list = []
    
    for file_key in dataset_dict.keys():
        observed_tuples = set()
        if 'SM' not in dataset_dict[file_key][0]:
            file_processing_list.append(file_key)
            curr_sm_label = dataset_dict[file_key][0] + '_SM'
            #Inefficient loop over file names, but there are currently never more than 50. Figure out
            #how to do better if nececssary in future
            for try_top_level in dataset_dict.keys():
                if curr_sm_label in try_top_level:
                    sm_location = try_top_level
  
            for possible_ion in total_ion_set:
                if possible_ion in dataset_dict[file_key][2].keys():
                    observed_tuples.add(possible_ion)
                    if possible_ion in dataset_dict[sm_location][2].keys():
                        getitem_list.append((file_key, dataset_dict[file_key][0], possible_ion, 'per'))
                    else:
                        getitem_list.append((file_key, dataset_dict[file_key][0], possible_ion, 'app'))
                else:
                    if possible_ion in dataset_dict[sm_location][2].keys():
                        getitem_list.append((file_key, dataset_dict[file_key][0], possible_ion, 'dis'))
                    else:
                        getitem_list.append((file_key, dataset_dict[file_key][0], possible_ion, 'oth'))
                        
            observed_ion_dict[file_key] = observed_tuples

    ion_dict_string = output_name + 'observed_ion_dict.pkl'
    pickle_file = open(ion_dict_string, 'wb')
    pickle.dump(observed_ion_dict, pickle_file)
    pickle_file.close()

    return getitem_list, file_processing_list, observed_ion_dict

def create_neighbor_test_via_tis(training_dict,
                                 test_dict,
                                 total_ion_set):
    """
    A function that creates the test __getitem__ list along with the test targets for
    the learned-nearest-neighbor approach

    Parameters
    ----------
    training_dict : TYPE
        DESCRIPTION.
    test_dict : TYPE
        DESCRIPTION.
    total_ion_set : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    getitem_list = []
    test_targets = []
    test_file_list = []
    
    for test_file in test_dict.keys():
        test_file_list.append(test_file)
        curr_sm_label = test_dict[test_file][0] + '_SM'
        for try_top_level in training_dict.keys():
            if curr_sm_label in try_top_level:
                sm_location = try_top_level
        
        for possible_ion in total_ion_set:
            if possible_ion in test_dict[test_file][2].keys():
                if possible_ion in training_dict[sm_location][2].keys():
                    getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'per'))
                    test_targets.append(test_dict[test_file][2][possible_ion])
                else:
                    getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'app'))
                    test_targets.append(test_dict[test_file][2][possible_ion])
            else:
                if possible_ion in training_dict[sm_location][2].keys():
                    getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'dis'))
                    test_targets.append(0.0)
                else:
                    getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'oth'))
                    test_targets.append(0.0)
    
    return getitem_list, test_targets, test_file_list
    

def enumerate_extraction_dataset(training_dict,
                                 output_name):
    """
    A function that takes a training dataset dictionary (created in CSVtoDict),
    and generates a full length list of tuples: where each tuple is a
    (filename, SM label, formula_tuple, ion_type) series. Pytorch itemgetter will use these pairs
    to look up the given normalized mass spec intensity from the training dictionary
    
    There are 4 distinct kinds of 'ion_type'. There are ions that are present in both
    the starting material, and an extracted fraction that are labelled 'per' for persistent.
    There are some ions that appear in an extracted fraction, but not in the starting fraction
    that are labelled as 'app' for appear. Some ions from the starting material do not appear
    in the extracted fraction, and are labelled as 'dis' for disappear. If the file being processed
    is a starting material file itself, these ions are ignored as they are never predicted in
    the desired process - SM information is always available before extraction, no need to predict.

    For cross-validation/testing the identity of all unique ions identified *must* be recorded to
    simulate real world situation. Can't just predict observed peaks - this requires knowledge of
    result ahead of time. Can't predict all 'possible' ions, as what does that mean? Can only
    predict all previously observed ions in training set.
    

    Parameters
    ----------
    training_dict : dictionary
        A dictionary of dictionaries. The top level keys are file names of the original
        .csv files of MS data. Each key points to a dictionary, which has tuple keys of
        the form (#C, #H, #N, #O, #S). Those keys have a single entry,
        the normalized MS intensity of the peak corresponding to that formula.
    output_name : dictionary
        A leading string that will be used to stamp the two pickled output files,
        containing the observed ion dictionary (a per-file dictionary) and the total ion list

    Returns
    -------
    An organized lists of tuples. The first tuple entry has the filename
    dictionary key; the second has the tuple dictionary key; these will allow
    the mass spec intensity to be returned when looked up with the previous
    two keys.
    
    Also returns a dictionary that has descriptions of which ions are observed per file, and in total.

    """
    
    getitem_list = []
    observed_ion_dict = {}
    total_ion_list = []
    
    for file_key in training_dict.keys():
        observed_tuples = []
        if 'SM' not in training_dict[file_key][0]:
            curr_sm_label = training_dict[file_key][0] + '_SM'
            #Inefficient loop over file names, but there are currently never more than 50. Figure out
            #how to do better if nececssary in future
            for try_top_level in training_dict.keys():
                if curr_sm_label in try_top_level:
                    sm_location = try_top_level
                    
            #Create per and app ion types
            for tuple_key in training_dict[file_key][1].keys():
                if tuple_key in training_dict[sm_location][1].keys():
                    getitem_list.append((file_key, training_dict[file_key][0], tuple_key, 'per'))
                else:
                    getitem_list.append((file_key, training_dict[file_key][0], tuple_key, 'app'))
                if tuple_key not in observed_tuples:
                    observed_tuples.append(tuple_key)
                if tuple_key not in total_ion_list:
                    total_ion_list.append(tuple_key)

            #Single pass through SM file to create 'dis' ion types
            for sm_tuple_key in training_dict[sm_location][1].keys():
                if sm_tuple_key not in observed_tuples:
                    getitem_list.append((file_key, training_dict[file_key][0], sm_tuple_key, 'dis'))
                if sm_tuple_key not in total_ion_list:
                    total_ion_list.append(sm_tuple_key)
                    
            observed_ion_dict[file_key] = observed_tuples
    
    ion_dict_string = output_name + 'observed_ion_dictionary.pkl'
    pickle_file = open(ion_dict_string, 'wb')
    pickle.dump(observed_ion_dict, pickle_file)
    pickle_file.close()
    
    total_ion_string = output_name + 'total_ion_list.pkl'
    pickle_file = open(total_ion_string, 'wb')
    pickle.dump(total_ion_list, pickle_file)
    pickle_file.close()
                            
    return getitem_list, observed_ion_dict, total_ion_list

def enumerate_neighbor_dataset(dataset_dict,
                               output_name):
    """
    A function that takes a training neighbor dataset dictionary (created in CSVtoDict),
    and generates a full length list of tuples: where each tuple is a
    (filename, SM label, formula_tuple, ion_type) series. Pytorch itemgetter will use these pairs
    to look up the given normalized mass spec intensity from the training dictionary
    
    There are 4 distinct kinds of 'ion_type'. There are ions that are present in both
    the starting material, and an extracted fraction that are labelled 'per' for persistent.
    There are some ions that appear in an extracted fraction, but not in the starting fraction
    that are labelled as 'app' for appear. Some ions from the starting material do not appear
    in the extracted fraction, and are labelled as 'dis' for disappear. If the file being processed
    is a starting material file itself, these ions are ignored as they are never predicted in
    the desired process - SM information is always available before extraction, no need to predict.

    For cross-validation/testing the identity of all unique ions identified *must* be recorded to
    simulate real world situation. Can't just predict observed peaks - this requires knowledge of
    result ahead of time. Can't predict all 'possible' ions, as what does that mean? Can only
    predict all previously observed ions in training set.
    
    Neural networks based on this approach are also *completely* sensitive to the order in
    which the tensors are constructed - also pickle an ordered list of file names - for testing,
    neighbors must also be constructed in exactly this order.
    

    Parameters
    ----------
    dataset_dict : dictionary
        A dictionary of dictionaries. The top level keys are file names of the original
        .csv files of MS data. Each key points to a dictionary, which has tuple keys of
        the form (#C, #H, #N, #O, #S). Those keys have a single entry,
        the normalized MS intensity of the peak corresponding to that formula.
    output_name : dictionary
        A leading string that will be used to stamp the two pickled output files,
        containing the observed ion dictionary (a per-file dictionary) and the total ion list

    Returns
    -------
    An organized lists of tuples. The first tuple entry has the filename
    dictionary key; the second has the tuple dictionary key; these will allow
    the mass spec intensity to be returned when looked up with the previous
    two keys.
    
    Also returns a dictionary that has descriptions of which ions are observed per file, and in total.

    """
    
    getitem_list = []
    observed_ion_dict = {}
    total_ion_list = []
    file_processing_list = []
    
    for file_key in dataset_dict.keys():
        observed_tuples = []
        if 'SM' not in dataset_dict[file_key][0]:
            file_processing_list.append(file_key)
            curr_sm_label = dataset_dict[file_key][0] + '_SM'
            #Inefficient loop over file names, but there are currently never more than 50. Figure out
            #how to do better if nececssary in future
            for try_top_level in dataset_dict.keys():
                if curr_sm_label in try_top_level:
                    sm_location = try_top_level
                    
            #Create per and app ion types
            for tuple_key in dataset_dict[file_key][2].keys():
                if tuple_key in dataset_dict[sm_location][2].keys():
                    getitem_list.append((file_key, dataset_dict[file_key][0], tuple_key, 'per'))
                else:
                    getitem_list.append((file_key, dataset_dict[file_key][0], tuple_key, 'app'))
                if tuple_key not in observed_tuples:
                    observed_tuples.append(tuple_key)
                if tuple_key not in total_ion_list:
                    total_ion_list.append(tuple_key)

            #Single pass through SM file to create 'dis' ion types
            for sm_tuple_key in dataset_dict[sm_location][2].keys():
                if sm_tuple_key not in observed_tuples:
                    getitem_list.append((file_key, dataset_dict[file_key][0], sm_tuple_key, 'dis'))
                if sm_tuple_key not in total_ion_list:
                    total_ion_list.append(sm_tuple_key)
                    
            observed_ion_dict[file_key] = observed_tuples
    
    ion_dict_string = output_name + 'observed_neigh_ion_dict.pkl'
    pickle_file = open(ion_dict_string, 'wb')
    pickle.dump(observed_ion_dict, pickle_file)
    pickle_file.close()
    
    total_ion_string = output_name + 'total_neigh_ion_list.pkl'
    pickle_file = open(total_ion_string, 'wb')
    pickle.dump(total_ion_list, pickle_file)
    pickle_file.close()
                            
    return getitem_list, file_processing_list, observed_ion_dict, total_ion_list

def expand_neighbor_getitem(dataset_dict,
                            getitem_list,
                            rectified_condition_dict,
                            file_processing_list,
                            normalization_tuple,
                            num_neighbors,
                            training_style):
    """
    The above enumerate_neighbor_dataset is based on previous dataset creation methods,
    and was kept so that other analysis tools could be used. For this more complex
    data presentation, it will be important not to be constantly querying the dataset.
    This function will expand the getitem list to include both the training tensor
    and the training target.
    The size of the tensors and the weighting of the target depends on the
    number of desired neighbors.

    Needs a helper file that rectifies the file_processing_list and condition_dict such that
    functions further down don't need to loop over the condition dictionary multiple times

    Parameters
    ----------
    training_dict : TYPE
        DESCRIPTION.
    getitem_list : TYPE
        DESCRIPTION
    condition_dict :
        
    file_processing_list : list
        An ordered list of (training) files for consistent tensor generation
    num_neighbors : integer
        Number of nearest neighbors to consider
    training_style : string
        If 'absolute' is used, then the dataset is looking to *minimize* the predicted variance
        between a given point and a given neighbor - as opposed to the normal training, where we
        are trying to *maximize* the chances that a given neighbor is the most accurate.
    Returns
    -------
    An expanded version of the getitem_list from above, with the new entries being
    a Pytorch tensor, as well as a target with a length identical to the number of
    files in the training set, for cross-entropy loss target

    """

    expanded_getitem_list = []
    
    if training_style != 'absolute':
        for list_entry in getitem_list:
            entry_filename = list_entry[0]
            entry_formula_tuple = list_entry[2]
            target_tensor = create_neighbor_target(entry_filename,
                                                   entry_formula_tuple,
                                                   dataset_dict,
                                                   file_processing_list,
                                                   num_neighbors)
            neighbor_tensor = create_neighbor_tensor(entry_filename,
                                                     entry_formula_tuple,
                                                     rectified_condition_dict,
                                                     file_processing_list,
                                                     normalization_tuple)
            expanded_getitem_list.append((list_entry[0], list_entry[1], list_entry[2], list_entry[3], neighbor_tensor, target_tensor))
    else:
        for list_entry in getitem_list:
            entry_filename = list_entry[0]
            entry_formula_tuple = list_entry[2]
            target_tensor = create_absolute_neighbor_target(entry_filename,
                                                            entry_formula_tuple,
                                                            dataset_dict,
                                                            file_processing_list)
            
            neighbor_tensor = create_absolute_neighbor_tensor(entry_filename,
                                                              entry_formula_tuple,
                                                              rectified_condition_dict,
                                                              file_processing_list,
                                                              normalization_tuple)

            expanded_getitem_list.append((list_entry[0], list_entry[1], list_entry[2], list_entry[3], neighbor_tensor, target_tensor))

        
    return expanded_getitem_list

def expand_neighbor_test_getitem(training_dict,
                                 test_getitem_list,
                                 rectified_condition_dict,
                                 file_processing_list,
                                 normalization_tuple):
    """
    

    Parameters
    ----------
    training_dict : TYPE
        DESCRIPTION.
    test_getitem_list : TYPE
        DESCRIPTION.
    condition_dict : TYPE
        DESCRIPTION.
    file_processing_list : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    expanded_test_getitem_list = []

    for list_entry in test_getitem_list:
        entry_filename = list_entry[0]
        entry_formula_tuple = list_entry[2]
        
        neighbor_tensor = create_neighbor_tensor(entry_filename,
                                                 entry_formula_tuple,
                                                 rectified_condition_dict,
                                                 file_processing_list,
                                                 normalization_tuple)
    
        expanded_test_getitem_list.append((list_entry[0], list_entry[1], list_entry[2], list_entry[3], neighbor_tensor))
    
    return expanded_test_getitem_list

def rectify_condition_dict(condition_dictionary,
                           file_processing_lists):
    """
    Awful double loop - but only has to be done once - saves a lot of time in the actual creation
    of the dataset. This is true because the length of condition_dict/file_processing_list are
    several thousand times smaller than the enumerated list for __getitem()

    Parameters
    ----------
    condition_dictionary : TYPE
        DESCRIPTION.
    file_processing_list : TYPE
        DESCRIPTION.

    Returns
    -------
    A new condition dictionary that has the actual file processing labels as dictionary keys

    """

    new_condition_dict = {}
    for single_processing_list in file_processing_lists:
        for full_length_entry in single_processing_list:
            for old_dict_key in condition_dictionary:
                if old_dict_key in full_length_entry:
                    new_condition_dict[full_length_entry] = condition_dictionary[old_dict_key]
    
    return new_condition_dict


def create_neighbor_target(target_file_label,
                           formula_tuple,
                           full_extraction_dictionary,
                           file_processing_list,
                           num_neighbors):
    """
    To streamline training, this function is only called once per ion during the creation
    of the dataset. Following the order of file_processing_list.
    
    For the target file that is being considered, this is filled with dummy values of +1
    and is never the nearest neighbor (give ppt dif of 1000 to be safe)

    Need to use try/except to account for the fact that not all ions will be observed
    in every training dictionary file
    
    In cases where there are (multiple) ties - distribute cross entropy target even
    further
    
    Parameters
    ----------
    target_file_label : string
        The name of the file for the data point being targeted during training/testing
    formula_tuple : tuple
        A five position tuple that contains (#C, #H, #N, #O, #S)
    full_extraction_dictionary : dictionary
        As created in BitumenCSVtoDict 'open_neighbor_(training/test)_dict'
    file_processing_list : list
        An ordered list that contains all of the training files to be compared against for
        nearest neighbor calculation
    num_neighbors : integer
        The desired number of neighbors for comparison

    Returns
    -------
    A pytorch tensor with a length equal to the number of files in the processing list, with
    the true nearest neighbors presented as values of 1/num_neighbors, and all others as zero

    """
    
    try:
        target_actual_value = full_extraction_dictionary[target_file_label][2][formula_tuple]
    except:
        target_actual_value = 0.0
        
    neighbor_act_distance = []
    
    for train_file in file_processing_list:
        if train_file != target_file_label:
            try:
                curr_act_val = full_extraction_dictionary[train_file][2][formula_tuple]
            except:
                curr_act_val = 0.0
                
            neighbor_act_distance.append(abs(curr_act_val - target_actual_value))
        else:
            neighbor_act_distance.append(1000)
    
    seq = sorted(neighbor_act_distance)
    ranks = [seq.index(val) for val in neighbor_act_distance]
    
    #Count number of winners including ties
    
    act_neighbors = 0
    for entry in ranks:
        if entry <= (num_neighbors - 1):
            act_neighbors = act_neighbors + 1
        
    pred_val = (1 / act_neighbors)

    target_list = [pred_val if rank <= (num_neighbors - 1) else 0.0 for rank in ranks ]
        
    target_np = np.asarray(target_list)
    target_tensor = torch.from_numpy(target_np)
    target_tensor = target_tensor.to(torch.float32)

    return target_tensor    

def create_absolute_neighbor_target(target_file_label,
                                    formula_tuple,
                                    full_extraction_dictionary,
                                    file_processing_list):
    """
    To streamline training, this function is only called once per ion during the creation
    of the dataset. Following the order of file_processing_list.
    
    For the target file that is being considered, this is filled with dummy values of +1
    and is never the nearest neighbor (give ppt dif of 1000 to be safe)

    Need to use try/except to account for the fact that not all ions will be observed
    in every training dictionary file
    
    In cases where there are (multiple) ties - distribute cross entropy target even
    further
    
    Parameters
    ----------
    target_file_label : string
        The name of the file for the data point being targeted during training/testing
    formula_tuple : tuple
        A five position tuple that contains (#C, #H, #N, #O, #S)
    full_extraction_dictionary : dictionary
        As created in BitumenCSVtoDict 'open_neighbor_(training/test)_dict'
    file_processing_list : list
        An ordered list that contains all of the training files to be compared against for
        nearest neighbor calculation
    num_neighbors : integer
        The desired number of neighbors for comparison

    Returns
    -------
    A pytorch tensor with a length equal to the number of files in the processing list, with
    the true nearest neighbors presented as values of 1/num_neighbors, and all others as zero

    """
    
    try:
        target_actual_value = full_extraction_dictionary[target_file_label][2][formula_tuple]
    except:
        target_actual_value = 0.0
        
    neighbor_act_distance = []
    
    for train_file in file_processing_list:
        if train_file != target_file_label:
            try:
                curr_act_val = full_extraction_dictionary[train_file][2][formula_tuple]
            except:
                curr_act_val = 0.0
                
            neighbor_act_distance.append(abs(curr_act_val - target_actual_value))
        else:
            neighbor_act_distance.append(10)
            
    target_np = np.asarray(neighbor_act_distance)
    target_tensor = torch.from_numpy(target_np)
    target_tensor = target_tensor.to(torch.float32)

    return target_tensor    

def create_neighbor_tensor(target_file_label,
                           formula_tuple,
                           condition_dict,
                           file_processing_list,
                           normalization_tuple):
    """
    To streamline training, this function is only called once per ion during the creation
    of the dataset. Following the order of file_processing_list.
    
    Creates a tensor of length (5 + 6x the length of the file_processing_list)
    
    For each entry in the file processing list, it compares the extraction conditions to
    those of the data point being predicted. For the current file under investigation,
    its entry is given a placeholder value of max distance (i.e. [1, 1, 1, 1, 1, 1]), as it
    will always be the 'further' neighbor as designed in create_neighbor_target
    
    For network training, we also want to know what ion formula we are looking at: perhaps
    low MW vs high MW, or low sulfur vs high sulfur have a difference sense of 'near'

    Parameters
    ----------
    target_file_label : TYPE
        DESCRIPTION.
    formula_tuple : TYPE
        DESCRIPTION.
    condition_dict : TYPE
        MUST USE RECTIFIED CONDITION DICTIONARY CREATED IN HELPER FUNCTION ABOVE!
    full_extraction_dictionary : TYPE
        DESCRIPTION.
    file_processing_list : TYPE
        DESCRIPTION.
    num_neighbors : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    min_val_list = []
    max_val_list = []
    
    for entry in normalization_tuple:
        min_val_list.append(entry[0])
        max_val_list.append(entry[1])
    
    min_val_array = np.asarray(min_val_list)
    max_val_array = np.asarray(max_val_list)
    
    starting_tuple_array = np.asarray(formula_tuple)
    adjusted_tuple_array = starting_tuple_array - min_val_array
    normalized_tuple_array = adjusted_tuple_array / max_val_array
    
    starting_condition_array = np.asarray(condition_dict[target_file_label])
    
    for poss_neighbor in file_processing_list:
        if poss_neighbor != target_file_label:
            curr_condition = np.asarray(condition_dict[poss_neighbor])
            condition_diff = curr_condition - starting_condition_array
            normalized_tuple_array = np.append(normalized_tuple_array, condition_diff)
        else:
            condition_diff_placeholder = np.asarray([1,1,1,1,1,1])
            normalized_tuple_array = np.append(normalized_tuple_array, condition_diff_placeholder)
    
    neighbor_tensor = torch.from_numpy(normalized_tuple_array)
    neighbor_tensor = neighbor_tensor.to(torch.float32)
    
    return neighbor_tensor

        
def report_dataset_distribution(getitem_list):
    """
    A function that takes the list to be used as a __getitem__() call, and reports on
    how many 'per', 'app', and 'dis' ions are observed for each file in the training set,
    as well as the total distribution

    Parameters
    ----------
    getitem_list : list generated by enumerate_extraction_training_dataset() above
        DESCRIPTION.

    Returns
    -------
    None, but prints results.

    """
    
    report_dict = {}
    for ion in getitem_list:
        #ion[0] is the filename for the extraction, ion[3] is the ion type
        if ion[0] not in report_dict.keys():
            report_dict[ion[0]] = {'app': 0, 'per': 0, 'dis': 0}
        report_dict[ion[0]][ion[3]] = report_dict[ion[0]][ion[3]] + 1

    print(report_dict)
    return

def enumerate_fig_dataset(labelled_training_dict):
    """
    A function that takes a labelled training dataset dictionary (created in CSVtoDict),
    and generates a full length list of tuples: where each tuple is a
    (filename, type_label, formula_tuple) triplet. Pytorch itemgetter will use these triplets
    to look up mass spec intensities from the correct dictionaries

    Parameters
    ----------
    labelled_training_dict : dictionary
        A dictionary of dictionaries. The top level keys are the file names of the original
        .csv files of MS data. Each key points to a (label, dictionary) pair, where the label
        which starting material led to this sample after extraction 'L1', etc. The appended dictionary has
        keys of the usual form (#C, #H, #N, #O, #S)

    Returns
    -------
    An organized list of tuples. The first tuple entry has the filename dictionary key;
    the second has the starting material label; the third has the tuple key of the form
    (#C, #H, #N, #O, #S). These will allow the correct mass spec intensities to be looked
     up during training/testing.

    """

    getitem_list = []
    
    for file_key in labelled_training_dict.keys():
        for tuple_key in labelled_training_dict[file_key][1].keys():
            getitem_list.append((file_key, labelled_training_dict[file_key][0], tuple_key))
    
    return getitem_list

def create_fig_tensor(target_formula,
                      single_experiment_dictionary,
                      current_formula_adjustments):
    """
    A function that takes a target formula (dictionary key), as well as a list of fragments
    that are currently being used in FIL searching. It returns a pytorch tensor that *does not*
    include the intensity of the target formula - only includes (target + adjustment 1)[intensity],
    (target - adjustment 1)[intensity], etc.
    
    When the tensor is created, the target formula is the first 5 positions. Therefore, the formula of
    all fragments is *implied* - as the formulae are not linearly independent, it could be harmful
    to add multiple CHNOS values across the tensor.

    Parameters
    ----------
    target_formula : tuple
        A tuple of the usual formula for the unknown target ion (#C, #H, #N, #O, #S)
    single_experiment_dictionary : dictionary
        A dictionary with keys = formula tuples, entries = sum normalized MS intensity
    current_formula_adjustments : list
        A list of all fragments to search for.

    Returns
    -------
    A pytorch tensor of length(current_test_fragments)

    """

    starting_formula_array = np.asarray(target_formula)
    
    running_return_array = np.asarray(target_formula)
    
    for target_fragment in current_formula_adjustments:
        target_array = np.asarray(target_fragment)
        curr_formula_target = starting_formula_array + target_array
        curr_formula_tuple = tuple(curr_formula_target)

        if curr_formula_tuple not in single_experiment_dictionary.keys():
            running_return_array = np.append(running_return_array, 0.0)
        
        else:
            running_return_array = np.append(running_return_array, single_experiment_dictionary[curr_formula_tuple])
    
    numpy_tensor = torch.from_numpy(running_return_array)
    
    numpy_tensor = numpy_tensor.to(torch.float32)

    return numpy_tensor 
        
def create_extraction_tensor(sm_label,
                             formula_tuple,
                             ion_type,
                             full_extraction_dictionary,
                             current_formula_adjustments):
    """
    A function that takes 3 of the __getitem__() set of objects, which includes the dictionary key for
    the starting material, the target molecular formula, and the ion type -
    and from this, as well as the list of current formula adjustments - creates the input tensor
    for ML training.
    
    The tensor containing the extraction condition information is appended by the Dataset object

    Parameters
    ----------
    target_getitem_tuple : tuple
        A tuple extracted from a list by __getitem__() that contains, in order, these 4
        pieces of information: [0] is the extraction file key, [1] is the starting material
        file key, [2] is the target ion formula tuple, [3] is the ion type
    full_extraction_dictionary : dictionary of dictionaries
        A full training or testing dictionary (created in BitumenCSVtoDict) that contains both the MS data
        for the exctracted fraction and for the starting material
    current_formula_adjustments : list of tuples
        A list of all the formula adjustments being included in the current ML analysis

    Returns
    -------
    A pytorch float tensor for use

    """

    starting_formula_array = np.float32(np.asarray(formula_tuple))
    
    running_return_array = np.float32(np.asarray(formula_tuple))

    sm_file_name = sm_label + '_SM'
    if ion_type == 'app':
        starting_ion_intensity = np.float32(0.0)
        for possible_sm_file in full_extraction_dictionary.keys():
            if sm_file_name in possible_sm_file:
                true_sm_name = possible_sm_file
                break
    else:
        for possible_sm_file in full_extraction_dictionary.keys():
            if sm_file_name in possible_sm_file:
                true_sm_name = possible_sm_file
                starting_ion_intensity = np.float32(full_extraction_dictionary[true_sm_name][1][formula_tuple])
                break
    #First update
    running_return_array = np.append(running_return_array, starting_ion_intensity)
    
    for formula_update in current_formula_adjustments:
        new_ion_lookup = starting_formula_array + np.asarray(formula_update)
        new_ion_lookup = tuple((np.rint(new_ion_lookup)).astype(int))
        try:
            this_ion_intensity = np.float32(full_extraction_dictionary[true_sm_name][1][new_ion_lookup])
        except:
            this_ion_intensity = np.float32(0.0)
        running_return_array = np.append(running_return_array, this_ion_intensity)
        
    complete_tensor = torch.FloatTensor(running_return_array)
    return complete_tensor
