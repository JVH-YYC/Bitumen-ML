#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:20:26 2023

@author: jvh

Another data visualization set of functions - this one focused
on clustering using UMAP.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap

def import_csv_and_craft(csv_file_directory,
                         csv_file_name,
                         ordered_target_column_list,
                         ordered_data_column_list,
                         target_mode,
                         data_mode):
    """
    A function that opens a .csv file and populates a pandas DataFrame with the desired
    descriptive ('target') columns along with the desired data columns. For both the target
    and columns, there are 3 modes for describing what to include or exclude.
    In 'exact', the column_list names are used precisely. In 'exclude', the columns included
    are everything except those listed. In 'range', an inclusive range of positions is used
    to avoid creating very long exact lists. In 'all' mode, the starting .csv is returned 
    with no editing performed.

    When using the 'exclude' or 'all' approach, *only* the target_column list is read - as the .csv
    file doesn't know which columns are target vs. data

    Parameters
    ----------
    csv_file_directory : TYPE
        DESCRIPTION.
    csv_file_name : TYPE
        DESCRIPTION.
    ordered_target_column_list : TYPE
        DESCRIPTION.
    ordered_data_column_list : TYPE
        DESCRIPTION.
    target_mode : TYPE
        DESCRIPTION.
    data_mode : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    starting_frame = pd.read_csv(csv_file_directory + csv_file_name, header=0, index_col=0)
    
    if target_mode == 'all':
        return starting_frame
    
    populated_frame = pd.DataFrame(index=starting_frame.index)
    
    if target_mode == 'exact':
        for target_title in ordered_target_column_list:
            populated_frame[target_title] = starting_frame[target_title]
    elif target_mode == 'range':
        target_titles = parse_range_list(ordered_target_column_list,
                                         starting_frame)
        for single_title in target_titles:
            populated_frame[single_title] = starting_frame[single_title]
    elif target_mode == 'exclude':
        for possible_title in starting_frame.columns:
            if possible_title not in ordered_target_column_list:
                populated_frame[possible_title] = starting_frame[possible_title]
        return populated_frame
    else:
        raise ValueError('Incorrect target mode used')

    if data_mode == 'exact':
        for target_title in ordered_data_column_list:
            populated_frame[target_title] = starting_frame[target_title]
    elif data_mode == 'range':
        target_titles = parse_range_list(ordered_data_column_list,
                                         starting_frame)
        for single_title in target_titles:
            populated_frame[single_title] = starting_frame[single_title]
    else:
        raise ValueError('Incorrect data mode used')
        
    return populated_frame

def parse_range_list(ordered_list,
                     full_dataframe):
    """
    A function that takes a list of numerical ranges and single entries ([1, 3, 5-8, 11, 17-19, etc.])
    and a dataframe that will be processed with these ranges, and creates and new list of the column
    titles that correspond to these ranges.

    Parameters
    ----------
    ordered_list : TYPE
        DESCRIPTION.
    full_dataframe : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Create a list to hold the column names
    column_names = []
    # Iterate through the ordered list
    for item in ordered_list:
        # Try to convert the item to an integer. This will be successful if it is a single entry
        try:
            curr_int = int(item)
            # If successful, append the column name to the list
            column_names.append(full_dataframe.columns[curr_int])
        # If it is not a single entry, it will be a range
        except ValueError:
            # Split the range into a list of two integers
            start, end = item.split('-')
            # Convert the integers to integers
            start = int(start)
            end = int(end)
            # Append the column names to the list
            column_names.extend(full_dataframe.columns[start:end+1])
    # Return the list of column names
    return column_names

# csv_file_directory='ExpCSV/Clustering/'
# csv_file_name='Just Avg for Clustering No SM.csv'
# ordered_target_column_list=['Label_0', 'Label_1', 'Label_2']
# ordered_data_column_list=[7,10,'15-20']
# target_mode='exact'
# data_mode='range'

# test_frame = import_csv_and_craft(csv_file_directory,
#                          csv_file_name,
#                          ordered_target_column_list,
#                          ordered_data_column_list,
#                          target_mode,
#                          data_mode)

# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# bit_data = pd.read_csv('Just Avg for Clustering No SM.csv')
# bit_num = bit_data.iloc[:,7:] #This is the actual data to fit, previous columns are type labels

# reducer = umap.UMAP()
# embed = reducer.fit_transform(bit_num)

# Plotting attempt #1 - no colors per sample
# plt.scatter(embed[:, 0],
#             embed[:, 1])


#Plotting attempt #2 - just L1 vs S2, with starting materials same color
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_0.map({'L1':0,
#                                                                     'S2':1})])

#Plotting attempt #3 - just toluene vs DCM, with starting materials a different color 
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_1.map({'toluene':0,
#                                                                     'DCM':1})])

#Plotting attempt #4 - polar additive each different color (and SM different color)
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_2.map({'AcOH':0,
#                                                                     'acetone':1,
#                                                                     'iPrOH':2,
#                                                                     'NEt3':3})])

#Plotting attempt #5 - toluene vs DCM including starting material information
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_3.map({'L1_tol':0,
#                                                                     'L1_DCM':1,
#                                                                     'S2_tol':2,
#                                                                     'S2_DCM':3})])

#Plotting attempt #4 - polar fractions, with starting materials a different color 
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_4.map({'L1_AcOH':0,
#                                                                     'L1_ace':1,
#                                                                     'L1_SM':2,
#                                                                     'S2_SM':3,
#                                                                     'L1_ipa':4,
#                                                                     'S2_net':5,
#                                                                     'S2_AcOH':6,
#                                                                     'S2_ace':7,
#                                                                     'S2_ipa':8})])

#Plotting with different UMAP settings
# reducer=umap.UMAP(n_neighbors=35,
#                   min_dist=0.999)
# embed=reducer.fit_transform(bit_num)
# print(embed)
#Plotting attempt #2 - just L1 vs S2, with starting materials same color
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_0.map({'L1':0,
#                                                                     'S2':1,})],
#             s=100)
# plt.show()

# Plotting attempt #5 - toluene vs DCM including starting material information
# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_3.map({'L1_tol':0,
#                                                                     'L1_DCM':1,
#                                                                     'S2_tol':2,
#                                                                     'S2_DCM':3})],
#             s=100)
# plt.show()

# plt.scatter(embed[:, 0],
#             embed[:, 1],
#             c=[sns.color_palette()[x] for x in bit_data.Label_2.map({'AcOH':0,
#                                                                     'acetone':1,
#                                                                     'iPrOH':2,
#                                                                     'NEt3':3})],
#             s=100)
# plt.show()


#Plotting with different UMAP settings in 3D
# reducer=umap.UMAP(n_neighbors=7,
#                   min_dist=0.5,
#                   n_components=3)
# embed=reducer.fit_transform(bit_num)

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')

#Plotting attempt #2 - just L1 vs S2, with starting materials same color
# ax.scatter(embed[:, 0],
#             embed[:, 1],
#             embed[:, 2],
#             c=[sns.color_palette()[x] for x in bit_data.Label_0.map({'L1':0,
#                                                                     'S2':1,})])
# plt.show()

# Plotting attempt #5 - toluene vs DCM including starting material information
# ax.scatter(embed[:, 0],
#             embed[:, 1],
#             embed[:, 2],
#             c=[sns.color_palette()[x] for x in bit_data.Label_3.map({'L1_tol':0,
#                                                                     'L1_DCM':1,
#                                                                     'S2_tol':2,
#                                                                     'S2_DCM':3})])
# plt.show()

# ax.scatter(embed[:, 0],
#             embed[:, 1],
#             embed[:, 2],
#             c=[sns.color_palette()[x] for x in bit_data.Label_2.map({'AcOH':0,
#                                                                     'acetone':1,
#                                                                     'iPrOH':2,
#                                                                     'NEt3':3})])
# plt.show()


