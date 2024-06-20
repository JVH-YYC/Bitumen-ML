"""
Additional dictionary processing tools.
Added during revisions to ChemRxiv submission found at https://doi.org/10.26434/chemrxiv-2024-pz45l
Additional simple functions for creating van Krevelen diagrams, and other associated things
Some simple function code generated with the help of ChatGPT/CoPilot.

"""

import DataTools.BitumenCreateUseDataset as BUD
import DataTools.BitumenCSVtoDict as BCD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def expand_with_elemental_ratios(bitumen_dictionary):
    """
    A function that takes a bitumen dictionary, as created in BCD,
    and appends to each formula entry the elemental ratios of the formula.
    This is made simple by the fact that the dictionary keys are the formulas,
    in the usual tuple order of (C, H, N, O, S).
    Create a new dictionary, that has the intensity at the first entry in a tuple,
    followed by a dictionary of elemental ratios and double bond equivalents.
    """

    expanded_dict = {}

    for key in bitumen_dictionary.keys():
        formula = key
        h_c_ratio = (formula[1] / formula[0])
        o_c_ratio = (formula[3] / formula[0])
        n_c_ratio = (formula[2] / formula[0])
        s_c_ratio = (formula[4] / formula[0])
        dbe = formula[0] - (formula[1] / 2) + (formula[2] / 2) + 1
        expanded_dict[key] = {'intensity': bitumen_dictionary[key],
                              'H/C': h_c_ratio,
                              'O/C': o_c_ratio,
                              'N/C': n_c_ratio,
                              'S/C': s_c_ratio,
                              'CNo': formula[0],
                              'DBE': dbe}
        #Now, add Kendrick mass defect
        iupac_mass = BCD.formula_to_mass(key)
        kendrick_mass = iupac_mass * (14.00000 / 14.01565)
        nominal_mass = BCD.nominal_kendrick_mass(key)
        kmd = nominal_mass - kendrick_mass
        expanded_dict[key]['KMD'] = kmd
        expanded_dict[key]['Nominal Mass'] = nominal_mass

    return expanded_dict

def create_expanded_single_dict(csv_name_incl_dir):
    """
    A function that takes a single .csv file, and creates a dictionary
    that includes all of the ratios necessary for creating more
    advanced diagrams.
    """

    column_names = ['Formula',
                    'Mono Inty']
    
    starting_frame = pd.read_csv(csv_name_incl_dir, names=column_names)
    formula_frame = starting_frame[starting_frame['Formula'].notna()]
    formula_frame = formula_frame[[x[0] == 'C' for x in formula_frame['Formula']]]
    
    formula_frame = formula_frame.reset_index(drop=True)

    simple_dict = BCD.single_sum_dict(formula_frame)
    final_dict = expand_with_elemental_ratios(simple_dict)

    return final_dict

def create_error_frame(csv_name_incl_dir):
    """
    A function that takes a single .csv file, which contains all the necessary
    error information for each formula in the bitumen dataset.
    Converts to a pandas dataframe, and returns the frame.
    """

    starting_frame = pd.read_csv(csv_name_incl_dir, header=0)
    starting_frame['ppm Error'] = pd.to_numeric(starting_frame['ppm Error'], errors='coerce')
    starting_frame['mDa Error'] = pd.to_numeric(starting_frame['mDa Error'], errors='coerce')
    
    formula_frame = starting_frame[starting_frame['Formula'].notna()]
    formula_frame = formula_frame[[x[0] == 'C' for x in formula_frame['Formula']]]

    formula_frame = formula_frame.reset_index(drop=True)

    return formula_frame

def plot_bitumen_data(bitumen_dict,
                      x_axis,
                      y_axis,
                      scale,
                      opacity,
                      log_scale=False):
    
    plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # Set the font family
    'axes.titlesize': 20,         # Font size for axis titles
    'axes.labelsize': 20,         # Font size for axis labels
    'xtick.labelsize': 16,        # Font size for x-axis tick labels
    'ytick.labelsize': 16,        # Font size for y-axis tick labels
    'lines.linewidth': 2,         # Line width for plot lines
    'axes.linewidth': 2,          # Line width for axes
    'xtick.major.width': 2,       # Line width for major x ticks
    'ytick.major.width': 2        # Line width for major y ticks
    })    
   
    # Extract data
    x_data = []
    y_data = []
    scale_data = []

    for key, sub_dict in bitumen_dict.items():
        x_data.append(float(sub_dict[x_axis]))
        y_data.append(float(sub_dict[y_axis]))
        if log_scale == False:
            scale_data.append(float(sub_dict[scale]))
        else:
            scale_data.append(np.log(float(sub_dict[scale])))

    # Convert lists to numpy arrays for better handling
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    scale_data = np.array(scale_data)

    # Determine min and max for color scaling
    scale_min = np.min(scale_data)
    scale_max = np.max(scale_data)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_data,
                          y_data,
                          c=scale_data,
                          cmap='inferno',
                          s=10)

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('log(ppt)')

    # Add labels and title
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    # Show plot
    plt.show()

    return

def plot_bitumen_hexbin(bitumen_dict,
                        x_axis,
                        y_axis,
                        scale,
                        opacity,
                        log_scale=False,
                        gridsize=50):
    # Extract data
    x_data = []
    y_data = []
    scale_data = []

    for key, sub_dict in bitumen_dict.items():
        x_data.append(float(sub_dict[x_axis]))
        y_data.append(float(sub_dict[y_axis]))
        scale_data.append(float(sub_dict[scale]))

    # Convert lists to numpy arrays for better handling
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    scale_data = np.array(scale_data)

    # Create hexbin plot
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(x_data, y_data, C=scale_data, gridsize=gridsize, cmap='viridis', reduce_C_function=np.mean)

    # Add color bar
    cbar = plt.colorbar(hb)
    cbar.set_label(scale)

    # Add labels and title
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'Hexbin plot of {y_axis} vs {x_axis} with {scale} as color scale')

    # Show plot
    plt.show()

    return

def plot_bitumen_hexbin_with_size(bitumen_dict,
                                  x_axis,
                                  y_axis,
                                  scale,
                                  opacity,
                                  log_scale=False,
                                  gridsize=50):
    plt.rcParams.update({
    'font.family': 'DejaVu Sans',  # Set the font family
    'axes.titlesize': 20,         # Font size for axis titles
    'axes.labelsize': 20,         # Font size for axis labels
    'xtick.labelsize': 16,        # Font size for x-axis tick labels
    'ytick.labelsize': 16,        # Font size for y-axis tick labels
    'lines.linewidth': 2,         # Line width for plot lines
    'axes.linewidth': 2,          # Line width for axes
    'xtick.major.width': 2,       # Line width for major x ticks
    'ytick.major.width': 2        # Line width for major y ticks
    })    
    # Extract data
    data = []
    for key, sub_dict in bitumen_dict.items():
        data.append((float(sub_dict[x_axis]), float(sub_dict[y_axis]), float(sub_dict[scale])))

    if log_scale == True:
        data = [(x, y, np.log(s)) for x, y, s in data]
        
    # Sort data by the size value to plot larger values on top
    data.sort(key=lambda x: x[2])

    # Unpack sorted data
    x_data, y_data, scale_data = zip(*data)

    # Normalize size data for plotting
    size_data = np.array(scale_data)
    size_data = (scale_data - np.min(scale_data)) / (np.max(scale_data) - np.min(scale_data)) * 250

    # Determine the extent of the data
    xmin, xmax = min(x_data), max(x_data)
    ymin, ymax = min(y_data), max(y_data)

    # Create hexbin plot
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(x_data, y_data, C=scale_data, gridsize=gridsize, cmap='inferno', reduce_C_function=np.mean, extent=(xmin, xmax, ymin, ymax))

    # Add color bar
    cbar = plt.colorbar(hb)
    cbar.set_label('log(ppt)')

    # Add labels and title
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)

    # Show plot
    plt.show()

    return

def plot_bitumen_scatter_with_size(bitumen_dict,
                                   x_axis,
                                   y_axis,
                                   size,
                                   opacity=0.5,
                                   log_scale=False,
                                   gridsize=50):
    # Extract data
    data = []
    for key, sub_dict in bitumen_dict.items():
        data.append((float(sub_dict[x_axis]), float(sub_dict[y_axis]), float(sub_dict[size])))

    if log_scale == True:
        data = [(x, y, np.log(s)) for x, y, s in data]
        
    # Sort data by the size value to plot larger values on top
    data.sort(key=lambda x: x[2])

    # Unpack sorted data
    x_data, y_data, size_data = zip(*data)

    # Normalize size data for plotting
    size_data = np.array(size_data)
    size_data = (size_data - np.min(size_data)) / (np.max(size_data) - np.min(size_data)) * 250

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x_data, y_data, s=size_data, c=size_data, cmap='inferno', alpha=opacity, edgecolors='none')

    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label(size)

    # Add labels and title
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'Scatter plot of {y_axis} vs {x_axis} with {size} as size and color')

    # Show plot
    plt.show()

    return

def plot_bitumen_scatter_with_size_3colors(dict1,
                                           dict2,
                                           x_axis,
                                           y_axis,
                                           scale,
                                           opacity=0.5,
                                           log_scale=False):
    # Extract data
    x_data = []
    y_data = []
    size_data = []
    colors = []

    # Define color maps
    color_map = {
        'dict1': 'Blues',
        'dict2': 'Oranges',
        'both': 'Greens'
    }

    # Process dict1
    for key, sub_dict in dict1.items():
        x_data.append(float(sub_dict[x_axis]))
        y_data.append(float(sub_dict[y_axis]))
        size_data.append(float(sub_dict[scale]))
        colors.append(color_map['dict1'])

    # Process dict2
    for key, sub_dict in dict2.items():
        if key in dict1:
            continue  # Skip keys already processed in dict1
        x_data.append(float(sub_dict[x_axis]))
        y_data.append(float(sub_dict[y_axis]))
        size_data.append(float(sub_dict[scale]))
        colors.append(color_map['dict2'])

    # Process common keys
    common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    for key in common_keys:
        sub_dict = dict1[key]  # Assuming dict1 and dict2 have the same values for common keys
        x_data.append(float(sub_dict[x_axis]))
        y_data.append(float(sub_dict[y_axis]))
        size_data.append(float(sub_dict[scale]))
        colors.append(color_map['both'])

    # Convert lists to numpy arrays for better handling
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    size_data = np.array(size_data)
    colors = np.array(colors)

    if log_scale == True:
        size_data = np.log(size_data)

    # Normalize size data for plotting
    size_data = (size_data - np.min(size_data)) / (np.max(size_data) - np.min(size_data)) * 1000

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    for color in np.unique(colors):
        indices = colors == color
        plt.scatter(x_data[indices], y_data[indices], s=size_data[indices], c=size_data[indices], cmap=color, alpha=opacity, edgecolors='none')

    # Add labels and title
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'Scatter plot of {y_axis} vs {x_axis} with {scale} as size and color')

    # Add legend
    plt.legend()
    
    # Show plot
    plt.show()

    return
    
def plot_bitumen_hexbin_3colors(dict1,
                                dict2,
                                x_axis,
                                y_axis,
                                scale,
                                log_scale=False,
                                opacity=0.5,
                                gridsize=50):
    # Extract data
    x_data_dict1 = []
    y_data_dict1 = []
    scale_data_dict1 = []

    x_data_dict2 = []
    y_data_dict2 = []
    scale_data_dict2 = []

    x_data_common = []
    y_data_common = []
    scale_data_common = []

    # Process dict1
    for key, sub_dict in dict1.items():
        x_data_dict1.append(float(sub_dict[x_axis]))
        y_data_dict1.append(float(sub_dict[y_axis]))
        scale_data_dict1.append(float(sub_dict[scale]))

    # Process dict2
    for key, sub_dict in dict2.items():
        if key in dict1:
            continue  # Skip keys already processed in dict1
        x_data_dict2.append(float(sub_dict[x_axis]))
        y_data_dict2.append(float(sub_dict[y_axis]))
        scale_data_dict2.append(float(sub_dict[scale]))

    # Process common keys
    common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    for key in common_keys:
        sub_dict = dict1[key]  # Assuming dict1 and dict2 have the same values for common keys
        x_data_common.append(float(sub_dict[x_axis]))
        y_data_common.append(float(sub_dict[y_axis]))
        scale_data_common.append(float(sub_dict[scale]))

    # Convert lists to numpy arrays for better handling
    x_data_dict1 = np.array(x_data_dict1)
    y_data_dict1 = np.array(y_data_dict1)
    scale_data_dict1 = np.array(scale_data_dict1)

    x_data_dict2 = np.array(x_data_dict2)
    y_data_dict2 = np.array(y_data_dict2)
    scale_data_dict2 = np.array(scale_data_dict2)

    x_data_common = np.array(x_data_common)
    y_data_common = np.array(y_data_common)
    scale_data_common = np.array(scale_data_common)

    if log_scale == True:
        scale_data_dict1 = np.log(scale_data_dict1)
        scale_data_dict2 = np.log(scale_data_dict2)
        scale_data_common = np.log(scale_data_common)

    # Determine the extent of the data
    xmin = min(np.min(x_data_dict1), np.min(x_data_dict2), np.min(x_data_common))
    xmax = max(np.max(x_data_dict1), np.max(x_data_dict2), np.max(x_data_common))
    ymin = min(np.min(y_data_dict1), np.min(y_data_dict2), np.min(y_data_common))
    ymax = max(np.max(y_data_dict1), np.max(y_data_dict2), np.max(y_data_common))

    # Create hexbin plot
    plt.figure(figsize=(10, 6))
    if len(x_data_dict1) > 0:
        hb1 = plt.hexbin(x_data_dict1, y_data_dict1, C=scale_data_dict1, gridsize=gridsize, cmap='Blues', alpha=opacity, reduce_C_function=np.mean, extent=(xmin, xmax, ymin, ymax))
        cbar1 = plt.colorbar(hb1)
        cbar1.set_label(f'{scale} (dict1)')

    if len(x_data_dict2) > 0:
        hb2 = plt.hexbin(x_data_dict2, y_data_dict2, C=scale_data_dict2, gridsize=gridsize, cmap='Oranges', alpha=opacity, reduce_C_function=np.mean, extent=(xmin, xmax, ymin, ymax))
        cbar2 = plt.colorbar(hb2)
        cbar2.set_label(f'{scale} (dict2)')

    if len(x_data_common) > 0:
        hb3 = plt.hexbin(x_data_common, y_data_common, C=scale_data_common, gridsize=gridsize, cmap='Greens', alpha=opacity, reduce_C_function=np.mean, extent=(xmin, xmax, ymin, ymax))
        cbar3 = plt.colorbar(hb3)
        cbar3.set_label(f'{scale} (common)')

    # Add labels and title
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(f'Hexbin plot of {y_axis} vs {x_axis} with {scale} as color scale')

    # Show plot
    plt.show()

    return

def plot_bitumen_hexbin_3plots(dict1,
                               dict2,
                               x_axis,
                               y_axis,
                               scale,
                               log_scale=False,
                               opacity=0.5,
                               savefig=False,
                               figname=None,
                               gridsize=50):
  
  # Extract data
    x_data_dict1 = []
    y_data_dict1 = []
    scale_data_dict1 = []

    x_data_dict2 = []
    y_data_dict2 = []
    scale_data_dict2 = []

    x_data_common = []
    y_data_common = []
    scale_data_common = []

    # Process dict1
    for key, sub_dict in dict1.items():
        x_data_dict1.append(float(sub_dict[x_axis]))
        y_data_dict1.append(float(sub_dict[y_axis]))
        scale_data_dict1.append(float(sub_dict[scale]))

    # Process dict2
    for key, sub_dict in dict2.items():
        if key in dict1:
            continue  # Skip keys already processed in dict1
        x_data_dict2.append(float(sub_dict[x_axis]))
        y_data_dict2.append(float(sub_dict[y_axis]))
        scale_data_dict2.append(float(sub_dict[scale]))

    # Process common keys
    common_keys = set(dict1.keys()).intersection(set(dict2.keys()))
    for key in common_keys:
        value = float((dict1[key][scale] + dict2[key][scale]) / 2) # Use average of common ions
        x_data_common.append(float(dict1[key][x_axis]))
        y_data_common.append(float(dict1[key][y_axis]))
        scale_data_common.append(value)

    # Convert lists to numpy arrays for better handling
    x_data_dict1 = np.array(x_data_dict1)
    y_data_dict1 = np.array(y_data_dict1)
    scale_data_dict1 = np.array(scale_data_dict1)

    x_data_dict2 = np.array(x_data_dict2)
    y_data_dict2 = np.array(y_data_dict2)
    scale_data_dict2 = np.array(scale_data_dict2)

    x_data_common = np.array(x_data_common)
    y_data_common = np.array(y_data_common)
    scale_data_common = np.array(scale_data_common)

    if log_scale == True:
        scale_data_dict1 = np.log(scale_data_dict1)
        scale_data_dict2 = np.log(scale_data_dict2)
        scale_data_common = np.log(scale_data_common)

    # Determine the extent of the data
    xmin = min(np.min(x_data_dict1), np.min(x_data_dict2), np.min(x_data_common))
    xmax = max(np.max(x_data_dict1), np.max(x_data_dict2), np.max(x_data_common))
    ymin = min(np.min(y_data_dict1), np.min(y_data_dict2), np.min(y_data_common))
    ymax = max(np.max(y_data_dict1), np.max(y_data_dict2), np.max(y_data_common))

    # Determine the uniform scale range
    scale_min = min(np.min(scale_data_dict1), np.min(scale_data_dict2), np.min(scale_data_common))
    scale_max = max(np.max(scale_data_dict1), np.max(scale_data_dict2), np.max(scale_data_common))

    # Create side-by-side hexbin plots using GridSpec
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.0)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharey=ax1)

    # Font dictionaries for Supporting Information
    title_font = {'fontsize': 20, 'fontweight': 'bold', 'fontname': 'DejaVu Sans'}
    axis_font = {'fontsize': 18, 'fontname': 'DejaVu Sans'}

    # Plot dict1
    hb1 = ax1.hexbin(x_data_dict1, y_data_dict1, C=scale_data_dict1, gridsize=gridsize, cmap='Blues', reduce_C_function=np.mean, extent=(xmin, xmax, ymin, ymax), vmin=scale_min, vmax=scale_max)
    ax1.set_title('A1 Only', fontdict=title_font)
    ax1.set_xlabel(x_axis, fontdict=axis_font)
    ax1.set_ylabel(y_axis, fontdict=axis_font)

    # Plot dict2
    hb2 = ax2.hexbin(x_data_dict2, y_data_dict2, C=scale_data_dict2, gridsize=gridsize, cmap='Oranges', reduce_C_function=np.mean, extent=(xmin, xmax, ymin, ymax), vmin=scale_min, vmax=scale_max)
    ax2.set_title('A2 Only', fontdict=title_font)
    ax2.set_xlabel(x_axis, fontdict=axis_font)
    ax2.tick_params(axis='y', which='both', left=False, labelleft=False, direction='in', pad=-15)

    # Plot common
    hb3 = ax3.hexbin(x_data_common, y_data_common, C=scale_data_common, gridsize=gridsize, cmap='Greens', reduce_C_function=np.mean, extent=(xmin, xmax, ymin, ymax), vmin=scale_min, vmax=scale_max)
    ax3.set_title('Shared', fontdict=title_font)
    ax3.set_xlabel(x_axis, fontdict=axis_font)
    ax3.tick_params(axis='y', which='both', left=False, labelleft=False, direction='in', pad=-15)

    # Create colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar1 = fig.colorbar(hb1, cax=cbar_ax1)
    cbar1.set_ticks([])

    cbar_ax2 = fig.add_axes([0.94, 0.1, 0.02, 0.8])
    cbar2 = fig.colorbar(hb2, cax=cbar_ax2)
    cbar2.set_ticks([])

    cbar_ax3 = fig.add_axes([0.96, 0.1, 0.02, 0.8])
    cbar3 = fig.colorbar(hb3, cax=cbar_ax3)
    cbar3.set_label('log(ppt)', fontdict=axis_font)

    plt.subplots_adjust(wspace=0)  # Remove horizontal space between plots
    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if savefig:
        plt.savefig(figname, format='png', dpi=300)
    plt.show()

    return

def plot_side_by_side_histograms(df, col1, col2, bins=50, alpha=1, color1='blue', color2='orange'):
    """
    Plot two side-by-side histograms for the specified columns in the dataframe.

    Parameters:
    - df: pandas DataFrame
    - col1: Name of the first column to plot
    - col2: Name of the second column to plot
    - bins: Number of bins for the histograms (default is 10)
    - alpha: Transparency level of the histograms (default is 0.7)
    - color1: Color of the first histogram (default is 'blue')
    - color2: Color of the second histogram (default is 'green')
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

    # Plot the first histogram
    df[col1].plot.hist(ax=axs[0], bins=bins, alpha=alpha, color=color1)
    axs[0].set_xlabel(col1)
    axs[0].set_ylabel('Frequency')

    # Plot the second histogram
    df[col2].plot.hist(ax=axs[1], bins=bins, alpha=alpha, color=color2)
    axs[1].set_xlabel(col2)
    axs[1].set_ylabel('Frequency')

    # Ensure x-axis is symmetric around zero
    max_limit = max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
    min_limit = min(axs[0].get_xlim()[0], axs[1].get_xlim()[0])
    max_abs_limit = max(abs(min_limit), abs(max_limit))
    axs[0].set_xlim(-max_abs_limit, max_abs_limit)
    axs[1].set_xlim(-max_abs_limit, max_abs_limit)

    plt.subplots_adjust(wspace=0)
    plt.tight_layout()
    plt.show()

    return