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
                          output_name,
                          pickle_output):
    """
    A function that looks at all extraction files in the training dictionary, and creates a set of
    all ions that are present in the entire training dictionary. It saves the set as a pickled object
    with the output_name as the leading string, to which is appended '_total_ion_set.pkl'
    
    Also creates a normalization tuple-of-tuples, which is formatted as: ((min #C, max #C), (min #H, max #H), etc)
    
    Depending on the input dictionary, the index of the training data can be different - do initial try/except
    to set proper indexing
    
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
    pickle_output : Boolean
        If 'True', pickle a copy of the total ion set

    Returns
    -------
    The total ion set

    """
    
    total_ion_set = set()
    starting_material_names = set()
    
    try:
        file_key = list(training_dict.keys())[0]
        test = training_dict[file_key][2].keys()
        target_index = 2
    except:
        target_index = 1
        
    for file_key in training_dict.keys():
        if 'SM' not in training_dict[file_key][0]:
            curr_sm_label = training_dict[file_key][0] + '_SM'
            starting_material_names.add(curr_sm_label)
            for tuple_key in training_dict[file_key][target_index].keys():
                if tuple_key not in total_ion_set:
                    total_ion_set.add(tuple_key)
    
    #One pass through starting material(s) to detect any ions that disappear during extraction
    for specific_sm in starting_material_names:
        for try_top_level in training_dict.keys():
            if specific_sm in try_top_level:
                sm_location = try_top_level
                for sm_tuple_key in training_dict[sm_location][target_index].keys():
                    if sm_tuple_key not in total_ion_set:
                        total_ion_set.add(sm_tuple_key)
    
    if pickle_output == True:
        total_ion_string = output_name + 'total_ion_set.pkl'
        to_pickle_file = open(total_ion_string, 'wb')
        pickle.dump(total_ion_set, to_pickle_file)
        to_pickle_file.close()
    
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
                                  total_ion_set):
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
    file_name_list = []
    
    for file_key in training_dict.keys():
        file_name_list.append(file_key)
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

    return getitem_list, file_name_list

def create_extraction_test_via_tis(training_dict,
                                   test_dict,
                                   total_ion_set):
    """
    A function that creates the necessary __getitem__ list for a test loop. For further statistics,
    ions are split into their types: 'per' ions are those that persist from the starting material
    into the extraction; 'app' ions are those that appear in the extraction, even though they were
    not present in the starting material; 'dis' ions are those that disappear from the starting
    material upon performing an extraction; oth are ions that are not present in either the starting
    material, or the test extraction, but were present in the total ion set.

    Returns
    -------
    An oriented set of lists - getitem_list and test_targets are aligned, where the ion targeted
    in getitem_list[idx] has an actual value of test_targets[idx]. test_file_list is a list of
    the file names that were used in the test set.

    """

    getitem_list = []
    test_targets = []
    test_file_list = []
    
    for test_file in test_dict.keys():
        if 'SM' not in test_dict[test_file][0]:
            test_file_list.append(test_file)
            curr_sm_label = test_dict[test_file][0] + '_SM'
            for try_top_level in training_dict.keys():
                if curr_sm_label in try_top_level:
                    sm_location = try_top_level
            
            for possible_ion in total_ion_set:
                if possible_ion in test_dict[test_file][1].keys():
                    if possible_ion in training_dict[sm_location][1].keys():
                        getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'per'))
                        test_targets.append(test_dict[test_file][1][possible_ion])
                    else:
                        getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'app'))
                        test_targets.append(test_dict[test_file][1][possible_ion])
                else:
                    if possible_ion in training_dict[sm_location][1].keys():
                        getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'dis'))
                        test_targets.append(0.0)
                    else:
                        getitem_list.append((test_file, test_dict[test_file][0], possible_ion, 'oth'))
                        test_targets.append(0.0)
    
    return getitem_list, test_targets, test_file_list

def expand_extraction_tis_dataset(dataset_dict,
                                  getitem_list,
                                  rectified_condition_dict,
                                  locked_formula_list,
                                  normalization_tuple):
    """
    An intermediate function that expands the getitem_list to include tensors necessary for the FCNet.
    This makes training loops much more efficient rather than creating tensors on the fly.

    Parameters
    ----------
    dataset_dict : dictionary
        A dictionary of dictionaries. The top level keys are file names of the original
        .csv files of MS data. Each key points to a dictionary, which has tuple keys of
        the form (#C, #H, #N, #O, #S). Those keys have a single entry,
        the normalized MS intensity of the peak corresponding to that formula.
    getitem_list : list of tuples
        A list of tuples as created in the functions above, that will hold the correct
        dictionary keys for looking up the mass spec intensity
    rectified_condition_dict : dictionary
        A dictionary containing a link between the extraction filename (key) and a barcode
        that indicates the solvent blend used in extraction
    additional_formula_list : list of tuples
        A list of tuples that contains the additional formula modifications that are to be
        looked up in the starting material HRMS, and included in the tensor
    normalization_tuple : tuple
        A normalization tuple generated during the creation of the dataset, that contains
        values for C/H/N/O/S to normalize the given input formula

    Returns
    -------
    An expanded version of the input getitem_list, with the new entries being a tensor than includes
    information about related formulae, the starting material intensity, and the extraction conditions,
    and the target ion value as a tensor

    """

    expanded_getitem_list = []

    for list_entry in getitem_list:
        entry_filename = list_entry[0]
        entry_sm = list_entry[1]
        entry_formula_tuple = list_entry[2]
        ion_type = list_entry[3]
        target_tensor = create_tis_extraction_target(entry_filename,
                                                     entry_formula_tuple,
                                                     ion_type,
                                                     dataset_dict)
        neighbor_tensor = create_tis_extraction_tensor(entry_filename,
                                                       entry_sm,
                                                       entry_formula_tuple,
                                                       rectified_condition_dict,
                                                       locked_formula_list,
                                                       normalization_tuple,
                                                       dataset_dict)
        expanded_getitem_list.append((list_entry[0], list_entry[1], list_entry[2], list_entry[3], neighbor_tensor, target_tensor))
        
    return expanded_getitem_list
    
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
        
    return getitem_list, observed_ion_dict, total_ion_list

def rectify_condition_dict(condition_dictionary,
                           file_processing_lists):
    """
    Awful double loop - but only has to be done once - saves a lot of time in the actual creation
    of the dataset. This is true because the length of condition_dict/file_processing_list are
    several thousand times smaller than the enumerated list for __getitem()

    Parameters
    ----------
    condition_dictionary : dictionary
        A dictionary that connects the extraction file names to the solvent blend used in the experiment
    file_processing_list : list
        A list of extraction files being considered in the current dataset

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

def create_tis_extraction_target(entry_filename,
                                 entry_formula_tuple,
                                 ion_type,
                                 dataset_dict):
    """
    A function that creates the target value for prediction, using the 'total ion set' method.

    Parameters
    ----------
    entry_filename : string
        The name of the .csv file that holds the current point of interest; this is a key for the
        compiled dataset dictionary
    entry_formula_tuple : tuple
        A tuple of the usual format [#C, #H, #N, #O, #S] that is used as a lookup dictionary key
    dataset_dict : dictionary
        A dictionary of dictionaries. The top level keys are file names of the original
        .csv files of MS data. Each key points to a dictionary, which has tuple keys of
        the form (#C, #H, #N, #O, #S). Those keys have a single entry,
        the normalized MS intensity of the peak corresponding to that formula.  
    
    Returns
    -------
    Pytorch tensor that is the target for prediction/training

    """

    if ion_type == 'oth' or ion_type == 'dis':
        np_target = np.asarray([0.0])
    
    else:
        np_target = np.asarray([dataset_dict[entry_filename][1][entry_formula_tuple]])    
    
    target_tensor = torch.from_numpy(np_target)
    target_tensor = target_tensor.to(torch.float32)
    
    return target_tensor

def create_tis_extraction_tensor(entry_filename,
                                 entry_sm,
                                 entry_formula_tuple,
                                 rectified_condition_dict,
                                 additional_formula_list,
                                 normalization_tuple,
                                 dataset_dict):
    """
    A function that takes the key extraction parameters, the value of a given ion in the *starting material HRMS*,
    and creates the Pytorch tensor that is the input for FCNet. This tensor is normalized, and includes the
    SM value, the extraction conditions, and the intensities of the locked ions in the order they are presented.

    Parameters
    ----------
    entry_filename : string
        The name of the .csv file that holds the current point of interest; this is a key for the
        compiled dataset dictionary
    entry_formula_tuple : tuple
        A tuple of the usual format [#C, #H, #N, #O, #S] that is used as a lookup dictionary key
    rectified_condition_dict : dictionary
        A dictionary that connects the file names to standardized extraction conditions.
    additional_formula_list : list of tuples
        A list of the formula modifications (i.e. 'formula information gain') that are to be added/substrated
        from the target ion in the *starting material HRMS*.
    normalization_tuple : tuple
        During creation of the dataset, the min/max values for each atom type are calculated and stored
        in a tuple, this is used to normalize input formula.

    Returns
    -------
    The pytorch tensor that is used for an FCNet input. The length of the tensor depends on the number
    of additional formula included.

    """

    #Find SM value
    for poss_sm in dataset_dict.keys():
        sm_string = entry_sm + '_SM'
        if sm_string in poss_sm:
            true_sm_file = poss_sm
            break
    
    #Format for training tuple = normalized formula tuple, then SM value, then extraction conditions,
    #then locked formula in order
    
    min_val_list = []
    max_val_list = []
    
    for entry in normalization_tuple:
        min_val_list.append(entry[0])
        max_val_list.append(entry[1])
    
    min_val_array = np.asarray(min_val_list)
    max_val_array = np.asarray(max_val_list)
    
    starting_formula_array = np.asarray(entry_formula_tuple)
    adjusted_formula_array = starting_formula_array - min_val_array
    normalized_formula_array = adjusted_formula_array / max_val_array    
    
    try:
        starting_material_val = np.float32(dataset_dict[true_sm_file][1][entry_formula_tuple])
    except:
        starting_material_val = np.float32(0.0)
    
    normalized_formula_array = np.append(normalized_formula_array, starting_material_val)

    example_extraction = rectified_condition_dict[entry_filename]
    condition_tensor = torch.FloatTensor(example_extraction)
    normalized_formula_array = np.append(normalized_formula_array, condition_tensor)    
    
    for formula_update in additional_formula_list:
        if len(formula_update) != 5:
            print('Short formula update: ', formula_update)
        new_ion_lookup = starting_formula_array + np.asarray(formula_update)
        new_ion_lookup = tuple((np.rint(new_ion_lookup)).astype(int))
        try:
            this_ion_intensity = np.float32(dataset_dict[true_sm_file][1][new_ion_lookup])
        except:
            this_ion_intensity = np.float32(0.0)
        normalized_formula_array = np.append(normalized_formula_array, this_ion_intensity)
        
    complete_tensor = torch.FloatTensor(normalized_formula_array)
    
    return complete_tensor
       
def report_dataset_distribution(getitem_list):
    """
    A function that takes the list to be used as a __getitem__() call, and reports on
    how many 'per', 'app', and 'dis' ions are observed for each file in the training set,
    as well as the total distribution

    Parameters
    ----------
    getitem_list : list generated by enumerate_extraction_training_dataset() above
        A list that includes the filename, starting material label, formula tuple, and ion type

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
    that are currently being used in FIG searching. It returns a pytorch tensor that *does not*
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
