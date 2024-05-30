This repository contains the code that was used to generate the results presented in our manuscript entitled
"Machine Learning in Complex Organic Mixtures: Applying Domain Knowledge Allows for Meaningful Performance with Small Datasets"
A working draft of the paper can be found at DOI: doi.org/10.26434/chemrxiv-2024-pz45l 

A brief description of the contents of each folder found here is as follows:

'AllCSV' contains the raw HRMS data for solvent extractions (19208_nar.csv to 19243_nar.csv), starting materials (L1_sm_nar.csv, S2_sm_nar.csv),
and identified compound classes ('Solvent_extractions_compound_classes_July2021.xlsx') used in the manuscript.

'Clustering' contains the compiled results of 'Formula Information Gain (FIG)' analysis and 'Compound Class (CC)' analysis used as inputs for UMAP clustering.

'D11 Pickles' contains the ML predictions that resulted during Leave-One-Out Cross-Validation (LOO-CV), when training with 11 additional pieces of context
from the starting material HRMS ('Depth 11'). For each extraction, two files were generated. 'XXXX_predact.pkl' is a pickled python list, where the list
has the structure [(actual, predicted), (actual, predicted), ...]. This data can be used to create violin plots showing the distribution of error per predicted point.
'XXXX_ppe.pkl' contains 'per-point error'. This file is a pickled python dictionary, where the dictionary keys correspond to a formula tuple (#C, #H, #N, #O, #S), which
points to a single float entry that is the absolute error for the predicted intensity of that ion: {(24, 36, 0, 0, 0): 0.0672843, (24, 38, 1, 0, 0): 0.0325824, ....}.
This data is used to constructed stacked plots that show actual vs. predicted vs. error amounts with the x-axis as the actual molecular mass. 'Split D11 Pickles' is identical,
with the key difference being that these ML models were trained with only examples from one of the two starting materials, rather than both, as described in the manuscript.

'Data Tools' contains key functions that transform the raw data output from HRMS analysis into a python dictionary useful for many functions ('BitumenCSVtoDict.py'), and then
further prepares this dictionary into a PyTorch dataset ('BitumenCreateUseDataset.py').

'ExtCSV' is simply duplicates of 'AllCSV', but without starting material HRMS included.

'ExtTIS LOO Trained Models' contains PyTorch state dictionaries for many of the models trained during LOO-CV in this work. For 3 extractions, it contains all trained models for
all 'Depths' (0-15), and it contains the highest performing models ('Depth 11') for all extractions. The code necessary for loading these state models into functioning networks
is found in 'BitumenFCNets.py', described below.

'FCNets' contains the main code required to create, train, test, and save the ML models developed in this work. 'BitumenWorkflows.py' contains the code necessary to generate
trained models (for either extraction prediction or formula information gain), and 'FinalNetworkTest.py' was used to generate the key results presented in the manuscript.

'Full Average Pickles', 'Persistence Pickels', and 'Split Average Pickles' are identical in structure to 'D11 Pickles', and contain the benchmark results that were compared
to the machine learning models.

'Heatmap CSV', 'PNG Files for Figures', and 'PNG Files for Supporting Information' contain the raw data/processed figures as indicated by the titles.

'SMCSV' is the complement for 'ExtCSV': it is a duplicate of 'AllCSV', but contains only the starting material HRMS.

'VisualizationTools' contains matplotlib code to generate the figures presented in the manuscript, including loading or freshly generating plots from saved/raw data.
