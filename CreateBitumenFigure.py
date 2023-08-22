#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:24:30 2023

@author: jvh
"""
import VisualizationTools.UMAPPlots as UMP
import VisualizationTools.MSPlots as MSP
import torch.nn as nn

# dataset_param_dict = {'sm_file_directory': 'ExpCSV/SM_nar',
#                       'ext_file_directory': 'ExpCSV/working_ext',
#                       'label_keys': label_keys,
#                       'test_list': ['19233_nar.csv',],
#                       'locked_formula': locked_formula,
#                       'condition_dict': condition_dict,
#                       'pickle_file': False,
#                       'output_name': 'placeholder'}
                        
# MSP.create_ms_difference_stack(list_of_csv_file_directories=csv_folder_list,
#                                list_of_csv_file_names=csv_file_list,
#                                list_of_plot_labels=plot_label_list,
#                                new_plot_label='Absolute Difference',
#                                list_of_yaxis_labels=yaxis_label_list,
#                                new_yaxis_label='Absolute Diff (ppt)',
#                                list_of_palette_dicts=list_of_pal_dict,
#                                new_palette_dict=diff_pal_dict,
#                                hrms_plot_dict=hrms_plot_dict,
#                                difference_mode='absolute')

# MSP.create_predicted_vs_actual_stack(dataset_param_dict=dataset_param_dict,
#                                      network_load_dict=network_load_dict,
#                                      network_param_dict=network_param_dict,
#                                      file_to_test='19213_nar.csv',
#                                      list_of_plot_labels=plot_label_list,
#                                      list_of_yaxis_labels=yaxis_label_list,
#                                      list_of_palette_dicts=list_of_pal_dict,
#                                      hrms_plot_dict=hrms_plot_dict,
#                                      difference_mode='raw')


# MSP.single_predact_plot_from_pickle(pickle_file_directory_list=['S1 PredAct Pickles/']*4,
#                                     pickle_file_name_list=['20060_20061_pair_ML_predact.pkl',
#                                                            '20062_20063_pair_ML_predact.pkl',
#                                                            '20064_20065_pair_ML_predact.pkl',
#                                                            '20066_20067_pair_ML_predact.pkl'],
#                                     predact_plot_dict=predact_plot_dict)

# MSP.multiple_predact_scatter_from_pickel(list_of_pickle_dict=list_of_scatter_dict,
#                                          scatter_plot_dict=scatter_plot_dict)

# MSP.compare_mse_for_scatters(list_of_pickle_dict=list_of_scatter_dict)

# MSP.single_ppe_violin_plot_from_pickle(pickle_file_directory_list=predact_combined_dir_list,
#                                         pickle_file_name_list=pers_ppe_list,
#                                         violin_plot_dict=violin_plot_dict)

# for entry in [20061, 20063, 20065, 20067]:
#     file_label = str(entry) + '_nar.csv'
#     output_name = str(entry) + '_split_by_committee'
#     dataset_param_dict = {'sm_file_directory': 'ExpCSV/S1P2 SM',
#                           'ext_file_directory': 'ExpCSV/S1P2 Split Ext',
#                           'label_keys': committee_label_keys,
#                           'test_list': [file_label,],
#                           'locked_formula': locked_formula,
#                           'condition_dict': committee_condition_dict,
#                           'pickle_file': False,
#                           'output_name': 'placeholder'}
#     MSP.calc_and_save_ml_by_committee(trained_net_directory=committee_trained_net_directory,
#                                       trained_net_name_list=committee_trained_net_list_d11,
#                                       trained_net_param_dict=network_param_dict,
#                                       dataset_param_dict=dataset_param_dict,
#                                       file_to_test=file_label,
#                                       difference_mode='absolute',
#                                       output_name=output_name,
#                                       dataset_pass=None)

    
# long_combined_dataframe = MSP.multiple_ppe_violin_plot_from_pickle(list_of_pickle_dict=list_of_pickle_dict,
#                                           violin_plot_dict=violin_plot_dict)

# dataset_param_dict = {'sm_file_directory': 'ExpCSV/S1_SM_folder',
#                       'ext_file_directory': 'ExpCSV/S1_ext_folder',
#                       'label_keys': committee_label_keys,
#                       'test_list': ['20062_nar.csv', '20063_nar.csv'],
#                       'locked_formula': locked_formula,
#                       'condition_dict': committee_condition_dict,
#                       'pickle_file': False,
#                       'output_name': 'placeholder'}

# MSP.calc_cfgc_SM_only_difference(dataset_param_dict=dataset_param_dict,
#                                   paired_files_to_test=['20062_nar.csv', '20063_nar.csv'],
#                                   paired_file_names=['20062', '20063'],
#                                   paired_sm_labels=['S1P1', 'S1P2'],
#                                   difference_mode='raw',
#                                   output_name='20062_20063_pair_SM_only',
#                                   dataset_pass=None)

# dataset_param_dict = {'sm_file_directory': 'ExpCSV/S1_SM_folder',
#                       'ext_file_directory': 'ExpCSV/S1_ext_folder',
#                       'label_keys': committee_label_keys,
#                       'test_list': ['20064_nar.csv', '20065_nar.csv'],
#                       'locked_formula': locked_formula,
#                       'condition_dict': committee_condition_dict,
#                       'pickle_file': False,
#                       'output_name': 'placeholder'}

# MSP.calc_cfgc_SM_only_difference(dataset_param_dict=dataset_param_dict,
#                                   paired_files_to_test=['20064_nar.csv', '20065_nar.csv'],
#                                   paired_file_names=['20064', '20065'],
#                                   paired_sm_labels=['S1P1', 'S1P2'],
#                                   difference_mode='raw',
#                                   output_name='20064_20065_pair_SM_only',
#                                   dataset_pass=None)

# dataset_param_dict = {'sm_file_directory': 'ExpCSV/S1_SM_folder',
#                       'ext_file_directory': 'ExpCSV/S1_ext_folder',
#                       'label_keys': committee_label_keys,
#                       'test_list': ['20066_nar.csv', '20067_nar.csv'],
#                       'locked_formula': locked_formula,
#                       'condition_dict': committee_condition_dict,
#                       'pickle_file': False,
#                       'output_name': 'placeholder'}

# MSP.calc_cfgc_SM_only_difference(dataset_param_dict=dataset_param_dict,
#                                   paired_files_to_test=['20066_nar.csv', '20067_nar.csv'],
#                                   paired_file_names=['20066', '20067'],
#                                   paired_sm_labels=['S1P1', 'S1P2'],
#                                   difference_mode='raw',
#                                   output_name='20066_20067_pair_SM_only',
#                                   dataset_pass=None)

# dataset_param_dict = {'sm_file_directory': 'ExpCSV/S1_SM_folder',
#                       'ext_file_directory': 'ExpCSV/S1_ext_folder',
#                       'label_keys': committee_label_keys,
#                       'test_list': ['20060_nar.csv', '20061_nar.csv'],
#                       'locked_formula': locked_formula,
#                       'condition_dict': committee_condition_dict,
#                       'pickle_file': False,
#                       'output_name': 'placeholder'}

# MSP.calc_cfgc_SM_only_difference(dataset_param_dict=dataset_param_dict,
#                                   paired_files_to_test=['20060_nar.csv', '20061_nar.csv'],
#                                   paired_file_names=['20060', '20061'],
#                                   paired_sm_labels=['S1P1', 'S1P2'],
#                                   difference_mode='raw',
#                                   output_name='20060_20061_pair_SM_only',
#                                   dataset_pass=None)



