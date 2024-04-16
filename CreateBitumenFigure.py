#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:24:30 2023

@author: jvh
"""
import VisualizationTools.UMAPPlots as UMP
import VisualizationTools.MSPlots as MSP
import VisualizationTools.visualization_settings as VS
import torch.nn as nn

#MSP.create_multiple_UMAP_scatter(multi_scatter_plot_dict = VS.cfgc_publication_multi_UMAP_dict,
#                                 umap_scatter_dict = VS.cfgc_publication_umap_scatter_dict,
#                                 constant_umap_settings = VS.cfgc_publication_consistent_UMAP_dict,
#                                 list_of_cluster_targets = VS.cfgc_publication_multiscatter_data_list,
#                                 output_file_name = 'test_multiscatter')

#MSP.create_consistent_multiUMAP_scatter(multi_scatter_plot_dict = VS.cfgc_publication_multi_UMAP_dict,
#                                        umap_scatter_dict = VS.cfgc_publication_umap_scatter_dict,
#                                        constant_umap_settings = VS.cfgc_publication_consistent_UMAP_dict,
#                                        nested_multiscatter_dict = VS.cfgc_publication_highlight_multiscatter,
#                                        list_of_cluster_targets = VS.cfgc_publication_single_highlight_multiscatter_list)

# MSP.all_solvent_UMAP(multi_scatter_plot_dict = VS.cfgc_publication_multi_multisolvent_scatter_dict,
#                      constant_umap_settings = VS.cfgc_publication_consistent_UMAP_dict,
#                      multisolvent_umap_dict=VS.cfgc_publication_multisolvent_dict,
#                      csv_file_column_list=VS.cfgc_publication_single_highlight_multiscatter_list,
#                      column_dictionary_key=VS.cfgc_publication_multisolvent_label_key,
#                      output_file_name='test_allsolv')

# dataset_param_dict = {'sm_file_directory': 'AllCSV',
#                       'ext_file_directory': 'AllCSV',
#                       'label_keys': VS.cfgc_bitumen_label_keys,
#                       'test_list': ['19242_nar.csv',],
#                       'locked_formula': VS.cfgc_publication_locked_formula,
#                       'condition_dict': VS.cfgc_bitumen_condition_dict,
#                       'pickle_file': False,
#                       'output_name': 'placeholder'}

# SM_csv_folder_list = ['./AllCSV/',
#                   './AllCSV/']

# SM_csv_file_list = ['L1_SM_nar.csv',
#                 'S2_SM_nar.csv']

# SM_plot_label_list = ['A1 Starting Mixture',
#                   'A2 Starting Mixture']

# SM_yaxis_label_list = ['Ion Intensity (ppt)',
#                       'Ion Intensity (ppt)']

# SM_list_of_pal_dict = [VS.standard_blue_pal_dict,
#                         VS.standard_orange_pal_dict]

# SM_hrms_plot_dict = VS.cfgc_publication_hrms_plot_dict

# MSP.create_ms_bar_stack_from_file_list(list_of_csv_file_directories=SM_csv_folder_list,
#                                       list_of_csv_file_names=SM_csv_file_list,
#                                       list_of_plot_labels=SM_plot_label_list,
#                                       list_of_yaxis_labels=SM_yaxis_label_list,
#                                       list_of_palette_dicts=SM_list_of_pal_dict,
#                                       hrms_plot_dict=SM_hrms_plot_dict,
#                                       output_name='Starting Material HRMS.png',
#                                       output_file_type='png')

# fig_2_csv_folder_list = ['./AllCSV/',
#                         './AllCSV/']

# fig_2_csv_file_list = ['S2_SM_nar.csv',
#                       '19224_nar.csv']

# fig_2_plot_label_list = ['A2 SM',
#                         'After extraction with\n20% NEt3/PhMe',
#                         'Absolute Difference']

# fig_2_yaxis_label_list = ['Intensity (ppt)',
#                          'Intensity (ppt)',
#                          '']

# fig_2_list_of_pal_dict = [VS.standard_blue_pal_dict,
#                         VS.standard_orange_pal_dict,
#                         VS.standard_green_pal_dict]

# fig_2_hrms_plot_dict = VS.cfgc_publication_hrms_plot_dict

# MSP.create_ms_difference_stack(list_of_csv_file_directories=fig_2_csv_folder_list,
#                                list_of_csv_file_names=fig_2_csv_file_list,
#                                list_of_plot_labels=fig_2_plot_label_list,
#                                list_of_yaxis_labels=fig_2_yaxis_label_list,
#                                list_of_palette_dicts=fig_2_list_of_pal_dict,
#                                hrms_plot_dict=fig_2_hrms_plot_dict,
#                                difference_mode='absolute',
#                                output_name='19224_extraction_diff.png',
#                                output_file_type='png')

# diff_plot_labels = ['A1 + 65% AcOH/PhMe',
#                     'ML Predicted',
#                     'Abs Error']

# diff_plot_yaxis = ['Actual MS (ppt)',
#                    'Predicted',
#                    '']

# diff_plot_pal_dict = [VS.standard_blue_pal_dict,
#                       VS.standard_orange_pal_dict,
#                       VS.standard_green_pal_dict]

# predact_network_load_dict = {'trained_net_directory': 'ExtTIS LOO Trained Models',
#                              'trained_net_name': 'LOO_19242_D11.pt'}

# MSP.create_predicted_vs_actual_stack(dataset_param_dict=dataset_param_dict,
#                                      network_load_dict=predact_network_load_dict,
#                                      network_param_dict=VS.cfgc_publication_network_param_dict,
#                                      file_to_test='19242_nar.csv',
#                                      list_of_plot_labels=diff_plot_labels,
#                                      list_of_yaxis_labels=diff_plot_yaxis,
#                                      list_of_palette_dicts=diff_plot_pal_dict,
#                                      hrms_plot_dict=VS.cfgc_publication_hrms_plot_dict,
#                                      difference_mode='absolute',
#                                      output_name='19242_predicted_vs_actual.png',
#                                      output_file_type='png')

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
    
# MSP.multiple_ppe_violin_plot_from_pickle(list_of_pickle_dict=VS.cfgc_publication_list_of_pickle_dict,
#                                           violin_plot_dict=VS.cfgc_publication_violin_plot_dict,
#                                           output_name='cfgc_overall_results_violin.png',
#                                           output_file_type='png')

# dataset_param_dict = {'sm_file_directory': 'ExpCSV/S1_SM_folder',
#                       'ext_file_directory': 'ExpCSV/S1_ext_folder',
#                       'label_keys': committee_label_keys,
#                       'test_list': ['20062_nar.csv', '20063_nar.csv'],
#                       'locked_formula': locked_formula,
#                       'condition_dict': committee_condition_dict,
#                       'pickle_file': False,
#                       'output_name': 'placeholder'}

# MSP.heatmap_from_csv(csv_folder='Heatmap CSV',
#                      csv_file_name='Combined Ext for Heatmap.csv',
#                      output_name='Combined Ext Heatmap.png',
#                      output_file_type='png',
#                      heatmap_dict=VS.cfgc_heatmap_settings)
