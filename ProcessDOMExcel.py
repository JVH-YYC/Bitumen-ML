"""
A short script that will process the openly available Murderkill river dataset
into a form that is directly processable with the existing BitumenML code
"""

import pandas as pd
import sys
import os

def create_formula(row):
    formula = ""
    if int(row['C']) > 0:
        if int(row['C']) > 1:
            formula += f"C{int(row['C'])}"
        else:
            formula += "C"
    if int(row['H']) > 0:
        if int(row['H']) > 1:
            formula += f"H{int(row['H'])}"
        else:
            formula += "H"
    if int(row['N']) > 0:
        if int(row['N']) > 1:
            formula += f"N{int(row['N'])}"
        else:
            formula += "N"
    if int(row['O']) > 0:
        if int(row['O']) > 1:
            formula += f"O{int(row['O'])}"
        else:
            formula += "O"
    if int(row['S']) > 0:
        if int(row['S']) > 1:
            formula += f"S{int(row['S'])}"
        else:
            formula += "S"

    return formula

def process_excel(file_path):
    # Read the Excel file
    xls = pd.ExcelFile(file_path)
    
    # Prepare the summary list to collect information for the .txt file
    summary = []
    
    for sheet_name in xls.sheet_names:
        # Read each sheet into a dataframe
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Total number of rows before filtering
        total_rows = len(df)
        
        # Number of rows excluded due to 'Na' != 0
        rows_na_excluded = len(df[df['Na'] != 0])
        
        # Number of rows excluded due to 'Cl' != 0
        rows_cl_excluded = len(df[df['Cl'] != 0])
        
        # Number of rows excluded due to '13C' != 0
        rows_13c_excluded = len(df[df['13C'] != 0])
        
        # Filter rows where 'Na', 'Cl', and '13C' are exactly zero
        df_filtered = df[(df['Na'] == 0) & (df['Cl'] == 0) & (df['13C'] == 0)]
        
        # Create the new dataframe with 'Formula' and 'Mono Inty'
        new_df = pd.DataFrame()
        new_df['Formula'] = df_filtered.apply(create_formula, axis=1)
        new_df['Mono Inty'] = df_filtered['Rel. Abundance']
        
        # Save the new dataframe as a CSV file
        csv_filename = f"{sheet_name}.csv"
        new_df.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")
        
        # Append detailed summary information
        summary.append(f"{csv_filename}: Total number of rows: {total_rows}, "
                       f"number of rows excluded for 'Na' != 0: {rows_na_excluded}, "
                       f"number of rows excluded for 'Cl' != 0: {rows_cl_excluded}, "
                       f"number of rows excluded for '13C' != 0: {rows_13c_excluded}")
    
    # Write the summary information to a text file
    summary_filename = "summary.txt"
    with open(summary_filename, 'w') as f:
        for line in summary:
            f.write(line + "\n")
    print(f"Summary saved to: {summary_filename}")

    return

