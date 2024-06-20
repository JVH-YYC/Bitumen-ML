#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:58:51 2021

@author: jvh

Series of functions that turns .csv files of mass spec data into dictionaries
with tuples of (C, H, N, O, S)

Simplified to only use peak area / sum of peak areas, only open-ended.
"""

import pandas as pd
from pathlib import Path
import pickle
import glob
# import DataTools.CreateUseDatasetNarrow as CUD
import time
import numpy as np

def create_csv_list(csv_file_directory,
                    holdout_list):
    """
    Creates a list of .csv files in a given directory

    Parameters
    ----------
    csv_file_directory : string
        Name of directory containing .csv files
    holdout_list : list
        Name of MS .csv files (and, therefore, also .pkl files) that are held out
        from training (for testing/cross-validation)        
    
    Returns
    -------
    List of .csv files names in given directory

    """
    holdout_strings = [(csv_file_directory + '/' + x) for x in holdout_list]
    
    file_path = Path('.', csv_file_directory)
    
    csv_list = list(file_path.glob('*.csv'))
    
    string_list = [str(x) for x in csv_list]
    
    return_strings = [x for x in string_list if x not in holdout_strings]        

    return return_strings

def define_sm_ext_train(sm_file_directory,
                        ext_file_directory,
                        label_keys,
                        test_list):
    """
    Creates a list of .csv files that are associated with MS spectra
    Each list entry is a list that is going to have a first entry that is the key
    label for creating a future dictionary ('L1', 'S2', 'L1_SM', 'S2_SM', etc.), while
    the second entry is the full file name for opening .csv in other functions

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
        Name of MS .csv files that are held out of training (either SM or extracted),
        but, must include all SM that are required for extracted fractions.

    Returns
    -------
    A list with the structure described above.

    """
    
    holdout_strings_sm = [(sm_file_directory + '/' + x) for x in test_list]
    holdout_strings_ext = [(ext_file_directory + '/' + x) for x in test_list]
    
    holdout_strings = holdout_strings_sm + holdout_strings_ext
    
    sm_path = Path('.', sm_file_directory)
    ext_path = Path('.', ext_file_directory)
    
    sm_csv_list = list(sm_path.glob('*.csv'))
    ext_csv_list = list(ext_path.glob('*.csv'))
       
    comb_csv_list = sm_csv_list + ext_csv_list
    
    comb_str_list = [str(x) for x in comb_csv_list]
    
    final_comb_list = [x for x in comb_str_list if x not in holdout_strings]
    
    key_string_list = []
    
    #Double loop to match keys OK because there are usually only <50 files. Need to change
    #if every have needs in very large datasets
    
    for specific_file in final_comb_list:
        for specific_key in label_keys.keys():
            if specific_key in specific_file:
                key_string_list.append([label_keys[specific_key], specific_file])

    return key_string_list

def define_sm_ext_test(sm_file_directory,
                       ext_file_directory,
                       label_keys,
                       test_list):
    """
    Creates a list of .csv files that are associated with MS spectra
    Each list entry is a list that is going to have a first entry that is the key
    label for creating a future dictionary ('L1', 'S2', 'L1_SM', 'S2_SM', etc.), while
    the second entry is the full file name for opening .csv in other functions
    
    This is the complementary function to that described above. The function above loads
    and organized all files MINUS a holdout list. This function loads ONLY the passed list.

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
        Name of MS .csv files that are being loaded for final testing.

    Returns
    -------
    A list with the structure described above.

    """
    #Hold out no starting materials
    
    test_strings_ext = [(ext_file_directory + '/' + x) for x in test_list]
        
    sm_path = Path('.', sm_file_directory)
    ext_path = Path('.', ext_file_directory)
    
    sm_csv_list = list(sm_path.glob('*.csv'))
    ext_csv_list = list(ext_path.glob('*.csv'))
    
    sm_str_list = [str(x) for x in sm_csv_list]
    ext_test_str = [str(x) for x in ext_csv_list]
    
    ext_test_list = [x for x in ext_test_str if x in test_strings_ext]
       
    comb_str_list = sm_str_list + ext_test_list
            
    key_string_list = []
    
    #Double loop to match keys OK because there are usually only <50 files. Need to change
    #if every have needs in very large datasets
    
    for specific_file in comb_str_list:
        for specific_key in label_keys.keys():
            if specific_key in specific_file:
                key_string_list.append([label_keys[specific_key], specific_file])
        
    return key_string_list
   
def load_from_list_nar(csv_name_incl_dir):
    """
    A function that loads .csv files that already include the directory name,
    as this is included in create_csv_list function.
    'Narrow' form of the function: .csv files will only have 'Formula' and 'Mono inty'

    Parameters
    ----------
    csv_name_incl_dir : string
        Full directory/file_name string for .csv file

    Returns
    -------
    Dataframe with only rows containing molecular formula returned

    """    
    
    column_names = ['Formula',
                    'Mono Inty']
    
    starting_frame = pd.read_csv(csv_name_incl_dir, names=column_names)
    formula_frame = starting_frame[starting_frame['Formula'].notna()]
    formula_frame = formula_frame[[x[0] == 'C' for x in formula_frame['Formula']]]
    
    formula_frame = formula_frame.reset_index(drop=True)
        
    return formula_frame

def formula_to_tuple(formula_string):
    """
    A function that takes in a molecular formula, and converts it into
    a tuple of the structure (#C, #H, #N, #O, #S)

    Parameters
    ----------
    formula_string : string
        A molecular formula read from a dataframe

    Returns
    -------
    A tuple with values as given above

    """

    formula_tuple = (return_atom_value('C', formula_string),
                    return_atom_value('H', formula_string),
                    return_atom_value('N', formula_string),
                    return_atom_value('O', formula_string),
                    return_atom_value('S', formula_string))
    
    return formula_tuple

def return_atom_value(target_atom,
                      formula_string):
    """
    A function that reads a string of the type C12H20NS2, etc. and for any
    given element of (C, H, N, O, S), returns the number of those atoms in the formula.
    Main complications are when the element is missing (val = 0), or when there is only
    1 of it - e.g. C12H20ON2, the 'O' has no '1' after it. Parsing along string needs
    a try/except in case you run off the end of the string, which happens when the
    final atom in the formula has a value of 1.
    
    A little ugly, but given that there are no more than 3 possible digits (e.g. <999 atoms)
    do a triple try/except loop.

    Parameters
    ----------
    target_atom : string
        Defines the atom being targeted
    formula_string : string
        The entire formula string being parsed

    Returns
    -------
    An integer, which is the number of the requested atom type in a given string

    """
    
    atom_val = 0
    integer_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    
    #First, check for absence
    if target_atom not in formula_string:
        return atom_val
    
    for character in range(len(formula_string)):
        if formula_string[character] == target_atom:
            try:
                if formula_string[character + 1] in integer_list:
                    run_string = formula_string[character + 1]
                else:
                    #Top level else, no integer after atom so value is 1
                    atom_val = 1
                    return atom_val
            except:
                #Top level except, value must be 1
                atom_val = 1
                return atom_val

            try:
                if formula_string[character + 2] in integer_list:
                    run_string = run_string + formula_string[character + 2]
                else:
                    #Level 2 else, must be single integer atom value
                    atom_val = int(run_string)
                    return atom_val
            except:
                #Level 2 except, must be a single integer atom value
                atom_val = int(run_string)
                return atom_val

            try:
                if formula_string[character + 3] in integer_list:
                    run_string = run_string + formula_string[character + 3]
                    atom_val = int(run_string)
                    return atom_val
                else:
                    atom_val = int(run_string)
                    return atom_val
            except:
                atom_val = int(run_string)
                return atom_val

def formula_to_mass(formula_tuple):
    """
    A function that converts a formula tuple back into a molecular mass

    Parameters
    ----------
    formula_tuple : tuple
        Tuple of the precise formula (#C, #H, #N, #O, #S)

    Returns
    -------
    A float value - the molecular mass of the given formula

    """
    
    exact_mass = ((formula_tuple[0] * 12.0) +
                  (formula_tuple[1] * 1.007825) +
                  (formula_tuple[2] * 14.003074) +
                  (formula_tuple[3] * 15.994915) +
                  (formula_tuple[4] * 31.972072))
    
    return exact_mass

def nominal_kendrick_mass(formula_tuple):
    """
    A function that converts a formula tuple back into a nominal mass,
    useful for KMD calculations according to Mass. Spec. Rev. 2009,
    28, 121-134.

    Parameters
    ----------
    formula_tuple : tuple
        Tuple of the precise formula (#C, #H, #N, #O, #S)

    Returns
    -------
    A float value - the molecular mass of the given formula

    """
    
    nkm = ((formula_tuple[0] * 12) +
           (formula_tuple[1] * 1) +
           (formula_tuple[2] * 14) +
           (formula_tuple[3] * 16) +
           (formula_tuple[4] * 32))
    
    return nkm

#### -----------------
#### Open-Ended dictionaries
#### -----------------     


def single_sum_dict(mass_spec_frame):
    """
    A function that creates a dictionary from a MS .csv file, with keys that
    are tuples of the usual form (#C, #H, #N, #O, #S), each holding
    a single entry: the normalized intensity of the peak observed.
    In this case, normalization is done with the total observed intensity,
    rather than the size of the largest peak.
    
    MW Cutoffs of 200-1000 used for a reasonable range
    
    Essentially, the units for graphs plotted with this dictionary will be 'Parts-per-thousand'

    Parameters
    ----------
    mass_spec_frame : pandas Dataframe
        Dataframe of MS results that has been stripped down to only rows that
        contain actual molecular formulae

    Returns
    -------
    A dictionary of MS intensities for a single file

    """

    ms_dict = {}
       
    mass_spec_frame['Mono Inty'] = pd.to_numeric(mass_spec_frame['Mono Inty'])
    mass_spec_frame = mass_spec_frame.set_index('Formula')
    mass_spec_frame['Total'] = mass_spec_frame.groupby(['Formula']).transform('sum')
    mass_spec_frame = mass_spec_frame[~mass_spec_frame.index.duplicated(keep='first')]
    
    normalization_sum = mass_spec_frame['Total'].sum()
    
    for formula, current_row in mass_spec_frame.iterrows():
        formula_tuple = formula_to_tuple(formula)
        if formula_to_mass(formula_tuple) >= 200.0 and formula_to_mass(formula_tuple) <=1000.0:    
            ms_dict[formula_tuple] = (current_row['Total'] / normalization_sum) * 1000
        
    return ms_dict
    
def print_single_file_size(csv_file_directory,
                           csv_file_name):
    """
    A function that prints the size of a single file, in terms of number of
    molecular formulae observed.
    """
    column_names = ['Formula',
                    'Mono Inty']
    csv_location = csv_file_directory + '/' + csv_file_name
    starting_frame = pd.read_csv(csv_location, names=column_names)
    formula_frame = starting_frame[starting_frame['Formula'].notna()]
    formula_frame = formula_frame[[x[0] == 'C' for x in formula_frame['Formula']]]
    
    formula_dict = single_sum_dict(formula_frame)
    print('Size of dict is:', len(formula_dict))
    
    return

def print_full_sm_set_size(csv_file_directory,
                           list_of_csv_files):
    """
    A function that combines the SM files and counts their size
    """
    column_names = ['Formula',
                    'Mono Inty']
    ion_set = set()
    for entry in list_of_csv_files:
        csv_location = csv_file_directory + '/' + entry
        starting_frame = pd.read_csv(csv_location, names=column_names)
        formula_frame = starting_frame[starting_frame['Formula'].notna()]
        formula_frame = formula_frame[[x[0] == 'C' for x in formula_frame['Formula']]]
        
        formula_dict = single_sum_dict(formula_frame)
        curr_set = set(formula_dict.keys())
        ion_set = ion_set.union(curr_set)
    
    print('Size of total ion set is:', len(ion_set))

def open_sum_training_dict(sm_file_directory,
                           ext_file_directory,
                           label_keys,
                           holdout_list):
    """
    A function that creates a dictionary for all .csv files in two directories, with
    top-level keys being the file name, each holding a tuple with two entries.
    The first entry is the label corresponding to either the material itself
    'L1_SM', 'S2_SM', etc. for starting materials, or what starting material
    an extracted fraction comes from 'L1', 'S2', etc. The second entry is
    the processed MS dictionary. Uses the sum of all observed intensities for
    normalization, rather than the size of the largest peak.

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
        Name of MS .csv files that are held out of training (either SM or extracted),
        but, must include all SM that are required for extracted fractions.

    Returns
    -------
    A dictionary of (key, dictionary) for training extraction predictions

    """                
    
    label_ms_list = define_sm_ext_train(sm_file_directory,
                                        ext_file_directory,
                                        label_keys,
                                        holdout_list)
    key_ms_dict = {}

    for label, file in label_ms_list:
        mass_spec_frame = load_from_list_nar(file)
        curr_dict = single_sum_dict(mass_spec_frame)
        key_ms_dict[file] = (label, curr_dict)

    return key_ms_dict

def open_sum_test_dict(sm_file_directory,
                       ext_file_directory,
                       label_keys,
                       test_list):
    """
    A function that creates a dictionary for all .csv files in two directories, with
    top-level keys being the file name, each holding a tuple with two entries.
    The first entry is the label corresponding to either the material itself
    'L1_SM', 'S2_SM', etc. for starting materials, or what starting material
    an extracted fraction comes from 'L1', 'S2', etc. The second entry is
    the processed MS dictionary.

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
        Name of MS .csv files that are being loaded for final testing.

    Returns
    -------
    A dictionary of (key, dictionary) for training extraction predictions

    """                

    label_ms_list = define_sm_ext_test(sm_file_directory,
                                       ext_file_directory,
                                       label_keys,
                                       test_list)

    key_ms_dict = {}

    for label, file in label_ms_list:
        mass_spec_frame = load_from_list_nar(file)
        curr_dict = single_sum_dict(mass_spec_frame)
        key_ms_dict[file] = (label, curr_dict)

    return key_ms_dict

def get_sm_sum(test_dataset,
               sm_name,
               formula_tuple):
    """
    A function that takes a given LabelledMSSetTest object, and for a given
    formula tuple, returns that value of the MS intensity of the starting material,
    in 'sum' of peaks normalization mode
    
    sm_name call must include '_SM' in call otherwise an extraction MS may be called

    Parameters
    ----------
    test_dataset : LabelledMSSetTest(Dataset) object
        Test dataset object as created in FCNarrowCut
    sm_name : string
        Name of the starting material being called
    formula_tuple : tuple
        A tuple of the usual formula (#C, #H, #N, #O, #S)

    Returns
    -------
    The value of the given formula in the starting material

    """
    
    for try_top_level in test_dataset.test_dictionary.keys():
        if sm_name in try_top_level:
            formula_value = test_dataset.test_dictionary[try_top_level][1][formula_tuple]
    
    return formula_value  
    
label_keys = {'L1_SM': 'L1_SM',
              'S2_SM': 'S2_SM',
              '19208': 'S2',
              '19209': 'S2',
              '19210': 'S2',
              '19211': 'S2',
              '19212': 'S2',
              '19213': 'S2',
              '19214': 'S2',
              '19215': 'S2',
              '19216': 'S2',
              '19217': 'S2',
              '19218': 'S2',
              '19219': 'S2',
              '19220': 'S2',
              '19221': 'S2',
              '19222': 'S2',
              '19223': 'S2',
              '19224': 'S2',
              '19226': 'L1',
              '19227': 'L1',
              '19228': 'L1',
              '19229': 'L1',
              '19230': 'L1',
              '19231': 'L1',
              '19232': 'L1',
              '19233': 'L1',
              '19234': 'L1',
              '19235': 'L1',
              '19236': 'L1',
              '19237': 'L1',
              '19238': 'L1',
              '19239': 'L1',
              '19240': 'L1',
              '19241': 'L1',
              '19242': 'L1',
              '19243': 'L1'}

condition_dict = {'L1_SM': [0, 0, 0, 0, 0, 0],
                  'S2_SM': [0, 0, 0, 0, 0, 0],
                  '19208': [0, 0.900, 0.100, 0, 0, 0],
                  '19209': [0, 0.900, 0, 0.100, 0, 0],
                  '19210': [0, 0.587, 0, 0.413, 0, 0],
                  '19211': [0, 0.757, 0, 0, 0.243, 0],
                  '19212': [0, 0.689, 0, 0.311, 0, 0],
                  '19213': [0.900, 0, 0.100, 0, 0, 0],
                  '19214': [0.851, 0, 0.149, 0, 0, 0],
                  '19215': [0.780, 0, 0.220, 0, 0, 0],
                  '19216': [0.900, 0, 0, 0.100, 0, 0],
                  '19217': [0.645, 0, 0, 0, 0.355, 0],
                  '19218': [0.720, 0, 0, 0.280, 0, 0],
                  '19219': [0.570, 0, 0, 0.430, 0, 0],
                  '19220': [0.820, 0, 0, 0, 0.180, 0],
                  '19221': [0.730, 0, 0, 0, 0.270, 0],
                  '19222': [0.890, 0, 0, 0, 0, 0.110],
                  '19223': [0.838, 0, 0, 0, 0, 0.162],
                  '19224': [0.800, 0, 0, 0, 0, 0.200],
                  '19226': [0, 0.502, 0.498, 0, 0, 0],
                  '19227': [0, 0.500, 0, 0.500, 0, 0],
                  '19228': [0, 0.450, 0, 0, 0.550, 0],
                  '19229': [0, 0.388, 0.612, 0, 0, 0],
                  '19230': [0, 0.340, 0, 0.660, 0, 0],
                  '19231': [0, 0.230, 0, 0.770, 0, 0],
                  '19232': [0, 0.360, 0, 0, 0.640, 0],
                  '19233': [0, 0.300, 0, 0, 0.700, 0],
                  '19234': [0, 0.600, 0.400, 0, 0, 0],
                  '19235': [0.583, 0, 0.417, 0, 0, 0],
                  '19236': [0.313, 0, 0.687, 0, 0, 0],
                  '19237': [0.606, 0, 0, 0.394, 0, 0],
                  '19238': [0.540, 0, 0, 0, 0.460, 0],
                  '19239': [0.175, 0, 0.825, 0, 0, 0],
                  '19240': [0.460, 0, 0, 0.540, 0, 0],
                  '19241': [0.170, 0, 0, 0.830, 0, 0],
                  '19242': [0.350, 0, 0, 0, 0.650, 0],
                  '19243': [0.250, 0, 0, 0, 0.750, 0]}
    
