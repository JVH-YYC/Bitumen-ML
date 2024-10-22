#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 10:37:37 2023

@author: jvh

A quick file to check the 'average' and 'persistence' calculations to ensure
they are correct
"""
import FCNets.BitumenFCNets as BFN
import VisualizationTools.MSPlots as MSP

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
#Depth 11 locked formula for LOO optimization

locked_formula=[(1,2,0,0,0),(-1,-2,0,0,0),
                (3,6,0,0,0),(-3,-6,0,0,0),
                (4,8,0,0,0),(-4,-8,0,0,0),
                (0,2,0,0,0),(0,-2,0,0,0),
                (2,2,0,0,0),(-2,-2,0,0,0),
                (4,2,0,0,0),(-4,-2,0,0,0),
                (2,0,0,0,1),(-2,0,0,0,-1),
                (2,2,0,0,-1),(-2,-2,0,0,1),
                (5,8,0,0,0),(-5,-8,0,0,0),
                (2,0,0,0,0),(-2,0,0,0,0),
                (3,2,0,0,0),(-3,-2,0,0,0)]

def print_full_cfgc_dataset_stats():
    for single_file in label_keys:
        if 'SM' not in single_file:
            curr_file_name = single_file + '_nar.csv'
            dataset = BFN.BitumenExtTISDataset(sm_file_directory='SMCSV',
                                            ext_file_directory='ExtCSV',
                                            label_keys=label_keys,
                                            test_list=[curr_file_name,],
                                            locked_formula=[],
                                            condition_dict=condition_dict,
                                            pickle_file=False,
                                            output_name='placeholder')
            print('For extractions file:', curr_file_name)
            dataset.report_dataset_metrics()
    return

test_dict = MSP.combine_pickled_data(['Split Average Pickles/',],
                                     ['19208_split_average_predact.pkl'])

def print_out_raw_data(formula,
                       dataset):
    """
    

    Parameters
    ----------
    formula : TYPE
        DESCRIPTION.
    dataset : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    print('For formula:', formula, 'the training data shows:')
    for training_file in dataset.training_dict.keys():
        try:
            print('File', training_file, ':', dataset.training_dict[training_file][1][formula])
        except:
            print('File', training_file, ': 0.0')
    
    return

