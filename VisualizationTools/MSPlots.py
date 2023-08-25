#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:27:29 2023

Data visualization tools for bitumen collaboration:
Plotting and stacking of actual mass spectra

@author: jvh
"""

import DataTools.BitumenCreateUseDataset as CUD
import DataTools.BitumenCSVtoDict as CTD
import FCNets.BitumenWorkflows as BWF
import FCNets.BitumenFCNets as BFN
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import pickle
import VisualizationTools.UMAPPlots as UMP
import umap

def load_single_ms(csv_file_directory,
                   csv_file_name):
    """
    A function that loads a single HRMS spectra in the standard format
    for future processing

    Parameters
    ----------
    csv_file_directory : string
        Name of folder that contains .csv HRMS file of interest
    csv_file_name : string
        Name of specific .csv file that will be plotted

    Returns
    -------
    The HRMS opened into a standard MW 200-1000 dataset as in BitumenCSVtoDict

    """

    filepath = Path('.', csv_file_directory, csv_file_name)
    filename = str(filepath)
    hrms_frame = CTD.load_from_list_nar(filename)
    hrms_dict = CTD.single_sum_dict(hrms_frame)
    
    return hrms_dict

def create_single_hrms_xy_pairs(hrms_dict):
    """
    A function that takes a HRMS dictionary of the standard format from BitumenCSVtoDict,
    and converts it into a list of (x, y) tuples for conversion into a visual plot of the spectra

    Parameters
    ----------
    hrms_dict : dictionary
        A dictionary that has keys that are a tuple of format (#C, #H, #N, #O, #S)
        with each key pointing to a single value that is the intensity of that ion

    Returns
    -------
    A list of tuples where each tuple is now a (molecular weight, intensity) pair

    """
    hrms_ion_list = []

    for specific_ion in hrms_dict:
        curr_fw = CTD.formula_to_mass(specific_ion)
        hrms_ion_list.append((curr_fw, hrms_dict[specific_ion]))
    
    return hrms_ion_list

def create_hrms_xy_list(list_of_csv_file_directories,
                        list_of_csv_file_names):
    """
    A folder that opens and processes a list of file folders and .csv file names,
    to allow for plotting/stacking of multiple spectra

    Parameters
    ----------
    list_of_csv_file_directories : TYPE
        DESCRIPTION.
    list_of_csv_file_names : TYPE
        DESCRIPTION.

    Returns
    -------
    An ordered list of xy points for plotting in the same order as the files entered

    """

    list_of_xy_data = []
    
    for file_position in range(len(list_of_csv_file_directories)):
        curr_dict = load_single_ms(list_of_csv_file_directories[file_position],
                                   list_of_csv_file_names[file_position])
        curr_xy_pairs = create_single_hrms_xy_pairs(curr_dict)
        list_of_xy_data.append(curr_xy_pairs)
    
    return list_of_xy_data

def create_difference_data(list_of_csv_file_directories,
                           list_of_csv_file_names,
                           difference_mode):
    """
    A function that takes 2 HRMS .csv files, and creates a third set of data representing the 'difference'
    between the spectra. 3 difference difference modes, 'raw', 'absolute', and 'squared'.
    Returns the third xy data list for plotting as a bar stack.

    Parameters
    ----------
    list_of_csv_file_directories : TYPE
        DESCRIPTION.
    list_of_csv_file_names : TYPE
        DESCRIPTION.
    difference_mode : TYPE
        DESCRIPTION.

    Returns
    -------
    An ordered list of xy data for plotting, where the third entry is the difference between spectra

    """
    if len(list_of_csv_file_directories) != 2 or len(list_of_csv_file_names) != 2:
        raise ValueError('Incorret number of HRMS files specified to perform difference calculation')
    
    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')
        
    list_of_xy_data = []
    starting_spec_1 = load_single_ms(list_of_csv_file_directories[0],
                                     list_of_csv_file_names[0])
    starting_spec_2 = load_single_ms(list_of_csv_file_directories[1],
                                     list_of_csv_file_names[1])

    #Need to loop through both spectra dictionaries to account for disappearing/new HRMS peaks    
    difference_dict = {}
    for first_spectra_key in starting_spec_1:
        if first_spectra_key in starting_spec_2:
            difference_dict[first_spectra_key] = compute_function(starting_spec_1[first_spectra_key],
                                                                  starting_spec_2[first_spectra_key])
        else:
            difference_dict[first_spectra_key] = compute_function(starting_spec_1[first_spectra_key],
                                                                  0.0)
    
    for second_spectra_key in starting_spec_2:
        if second_spectra_key not in difference_dict:
            if second_spectra_key in starting_spec_1:
                difference_dict[second_spectra_key] = compute_function(starting_spec_1[second_spectra_key],
                                                                       starting_spec_2[second_spectra_key])
            else:
                difference_dict[second_spectra_key] = compute_function(0.0,
                                                                       starting_spec_2[second_spectra_key])
    
    #Now, process and return xy points for plotting in the desired (typical) stack order
    list_of_xy_data = [create_single_hrms_xy_pairs(starting_spec_1),
                       create_single_hrms_xy_pairs(starting_spec_2),
                       create_single_hrms_xy_pairs(difference_dict)]
    
    return list_of_xy_data

def return_raw(spectra_1_val,
               spectra_2_val):
    return (spectra_2_val - spectra_1_val)

def return_absolute(spectra_1_val,
                    spectra_2_val):
    return abs(spectra_2_val - spectra_1_val)

def return_square(spectra_1_val,
                  spectra_2_val):
    return (spectra_2_val - spectra_1_val)**2

def plot_single_ms(hrms_ion_list,
                   hrms_plot_dict):
    """
    A function that creates a single plot of a bitumen HRMS
    The function is as variable as possible, with a handful of standards used
    throughout the JVH group: 300 dpi resolution, white background, and with 0.05 in
    margins

    Parameters
    ----------
    hrms_ion_list : list of tuples
        A list of tuples where each tuple is a (molecular weight, intensity) pair
    hrms_plot_dict : dictionary
        A dictionary that contains all of the necessary information
        for creating the plot of interest, which at minimum includes:
            'plot_width' - horizontal size of plot
            'plot_height' - vertical size of plot
            'title' - self-explanatory (s/e)
            'title_size' - font size for title
            'xaxis_label' - s/e
            'yaxis_label' - s/e
            'axis_size' - font size for label
            'axis_scale_size' - font size for axis label scale
            'font_type' - s/e
            'palette' - s/e
            'num_shades' - how many shades to generate for a given palette
            'shade_choice' - which # shade from the palette to use for the bars
            'xmajor_ticks' - s/e
            'xminor_ticks' - s/e
            'ymajor_ticks' - s/e
            'yminor_ticks' - s/e
            'bar_width' - s/e, standard value is 0.8
            'opacity' - opacity of the plot bars, 1 = opaque 0 = transparent
            'save_output' - Boolean s/e
            'output_name' - s/e, must explicitly include .png

    Returns
    -------
    None, but saves the plot if desired

    """
    #Set the font family
    plt.rcParams['font.family'] = hrms_plot_dict['font_type']
    #Create palette and select color
    palette = sns.color_palette(hrms_plot_dict['palette'], hrms_plot_dict['num_shades'])
    bar_color = palette[hrms_plot_dict['shade_choice']]
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(hrms_plot_dict['plot_width'],
                                    hrms_plot_dict['plot_height']))
    # Create the bar plot
    x = [x[0] for x in hrms_ion_list]
    y = [x[1] for x in hrms_ion_list]
    ax.bar(x, y,
           color=bar_color,
           width=hrms_plot_dict['bar_width'],
           alpha=hrms_plot_dict['opacity'])
    # Set the title
    ax.set_title(hrms_plot_dict['title'],
                    fontsize=hrms_plot_dict['title_size'])
    # Set the x-axis label and font sizes
    ax.set_xlabel(hrms_plot_dict['xaxis_label'],
                    fontsize=hrms_plot_dict['axis_size'])
    ax.xaxis.label.set_fontsize(hrms_plot_dict['axis_scale_size'])
    # Set the y-axis label
    ax.set_ylabel(hrms_plot_dict['yaxis_label'],
                    fontsize=hrms_plot_dict['axis_size'])
    ax.yaxis.label.set_fontsize(hrms_plot_dict['axis_scale_size'])
    # Set the x-axis ticks
    xmajor_locator = ticker.MultipleLocator(hrms_plot_dict['xmajor_ticks'])
    xminor_locator = ticker.MultipleLocator(hrms_plot_dict['xminor_ticks'])
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    # Set the y-axis ticks
    ymajor_locator = ticker.MultipleLocator(hrms_plot_dict['ymajor_ticks'])
    yminor_locator = ticker.MultipleLocator(hrms_plot_dict['yminor_ticks'])
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)
    # Save the figure
    if hrms_plot_dict['save_output'] == True:
            plt.savefig(hrms_plot_dict['output_name'],
                        facecolor="white",
                        bbox_inches='tight',
                        pad_inches=0.05,
                        dpi=300)
    plt.close()
    
    return

def plot_multiple_hrms_stack(list_of_hrms_spectra,
                             list_of_plot_labels,
                             list_of_yaxis_labels,
                             list_of_palette_dicts,
                             hrms_plot_dict):
    """
    A function, similar to above, that takes multiple HRMS spectra and
    stacks them on top of one another with a shared x-axis at the bottom
    The function is as variable as possible, with a handful of standards used
    throughout the JVH group: 300 dpi resolution, white background, and with 0.05 in
    margins. By default the x-axis is shared and all y-axes have the same scale.
    This function, therefore, can also be used to plot predicted/actual/error stacks

    Parameters
    ----------
    list_of_hrms_spectra : list of lists
        Each entry in the list it itself a list, where that list contains
        tuples of (molecular weight, ion intensity)
    list_of_plot_labels : list of strings
        In the same order as the list_of_hrms_spectra, this list contains
        the figure labels that will be added to the plots.
    list_of_yaxis_labels : list of string
        In the same order as the list_of_hrms_spectra, this list contains
        all of the y-axis labels that will be added to the plots.
    list_of_palette_dicts : list of dictionaries
        In the same order as the list_of_hrms_spectra, this list contains
        the color information that will be used to generate the plots
    hrms_plot_dict : dictionary
        A dictionary that contains all of the necessary information
        for creating the plot of interest, which at minimum includes:
            'plot_width' - horizontal size of plot
            'plot_height' - vertical size of plot
            'title' - self-explanatory (s/e)
            'title_size' - font size for title
            'xaxis_label' - s/e
            'yaxis_label' - s/e
            'axis_size' - font size for label
            'axis_scale_size' - font size for axis label scale
            'sublabel_size' - the size of the title for each individual stacked plot
            'font_type' - s/e
            'palette' - s/e
            'num_shades' - how many shades to generate for a given palette
            'shade_choice' - which # shade from the palette to use for the bars
            'xmajor_ticks' - s/e
            'xminor_ticks' - s/e
            'ymajor_ticks' - s/e
            'yminor_ticks' - s/e
            'bar_width' - s/e, standard value is 0.8
            'opacity' - opacity of the plot bars, 1 = opaque 0 = transparent
            'save_output' - Boolean s/e
            'output_name' - s/e, must explicitly include .png
            'label_x_pos_fraction' - how far along the x axis the individual plot axis
                                        its label should appear
            'label_y_pos_fraction' - how for along the y axis the individual plot axis
                                        its label should appear
            'label_headspace' - what amount extra y-axis should be added to each plot to
                                        allow for a label to be added. Default units
                                        is inches.
            'stack_headspace' - what will be the spacing between the stacked plots?
            'uniform_height' - Boolean. If true, before plotting the stacked spectra,
                                        will check to see largest y-axis value, and
                                        rescales all plots to this max scale
            'uniform_scaling' - Boolean. If true, this will re-size the plots so that
                                        the spacing between tick marks is unchanged. (IE
                                        if one plot has a max y-value of 4, and one has
                                        a value of 3, then the first chart should be 33% taller)

    Returns
    -------
    None, but saves the plot if desired.

    """
    #Set the font family
    plt.rcParams['font.family'] = hrms_plot_dict['font_type']

    # Set the x-axis ticks
    xmajor_locator = ticker.MultipleLocator(hrms_plot_dict['xmajor_ticks'])
    xminor_locator = ticker.MultipleLocator(hrms_plot_dict['xminor_ticks'])

    # Set the y-axis ticks. Don't know why, but doesn't work properly with MultipleLocator
    ymajor_size = hrms_plot_dict['ymajor_ticks']
    yminor_size = hrms_plot_dict['yminor_ticks']
    
    # Create the figure
    num_stacks = len(list_of_hrms_spectra)
    
    if hrms_plot_dict['uniform_scaling'] == True:
        scaling_list = define_scaling_ratio(list_of_hrms_spectra,
                                            hrms_plot_dict['label_headspace'])
        fig, ax_list = plt.subplots(num_stacks,
                                    1,
                                figsize=(hrms_plot_dict['plot_width'],
                                    hrms_plot_dict['plot_height']),
                                sharex=True,
                                gridspec_kw={'height_ratios': scaling_list})
    else:
        fig, ax_list = plt.subplots(num_stacks,
                                    1,
                                    figsize=(hrms_plot_dict['plot_width'],
                                        hrms_plot_dict['plot_height']),
                                    sharex=True)
        
    # Create the (multiple) bar plots
    for position, single_hrms_plot in enumerate(list_of_hrms_spectra):
        #Select this plot colors
        palette = sns.color_palette(list_of_palette_dicts[position]['palette'],
                                    list_of_palette_dicts[position]['num_shades'])
        bar_color = palette[hrms_plot_dict['shade_choice']]
        
        #Add this plot data
        x = [x[0] for x in single_hrms_plot]
        y = [x[1] for x in single_hrms_plot]
        ax_list[position].bar(x, y,
                              color=bar_color,
                              width=hrms_plot_dict['bar_width'],
                              alpha=hrms_plot_dict['opacity'])
        #Round up to nearest minor tick
        desired_max = (max(y) + hrms_plot_dict['label_headspace'])
        
        round_ylim = round_to_nearest_tick(desired_max,
                                           hrms_plot_dict['ymajor_ticks'],
                                           hrms_plot_dict['yminor_ticks'])
        
        #Add y-axis headroom so that a label can be appended
        ax_list[position].set_ylim(0, round_ylim)
        #Set this plot y-axis label
        ax_list[position].set_ylabel(list_of_yaxis_labels[position],
                                     fontsize=hrms_plot_dict['axis_size'])
        #Set this plot x-axis ticks
        ax_list[position].xaxis.set_major_locator(xmajor_locator)
        ax_list[position].xaxis.set_minor_locator(xminor_locator)
        ax_list[position].tick_params(axis='x', labelsize=hrms_plot_dict['axis_scale_size'])

        #Set this plot y-axis ticks
        ax_list[position].yaxis.set_major_locator(ticker.FixedLocator(np.arange(ymajor_size,
                                                                                (desired_max + ymajor_size),
                                                                                ymajor_size)))
        ax_list[position].yaxis.set_minor_locator(ticker.FixedLocator(np.arange(yminor_size,
                                                                                round_ylim + yminor_size,
                                                                                yminor_size)))
        ax_list[position].tick_params(axis='y', labelsize=hrms_plot_dict['axis_scale_size'])
        
        #Add label to this plot
        ax_list[position].annotate(list_of_plot_labels[position],
                                   xy=(hrms_plot_dict['label_x_pos_fraction'],
                                       hrms_plot_dict['label_y_pos_fraction']),
                                   xycoords='axes fraction',
                                   ha='right',
                                   va='top',
                                   fontsize=hrms_plot_dict['sublabel_size'])

    # Set the overall plot title
    plt.suptitle(hrms_plot_dict['title'],
                    fontsize=hrms_plot_dict['title_size'])
    # Set the x-axis label
    ax_list[-1].set_xlabel(hrms_plot_dict['xaxis_label'],
                    fontsize=hrms_plot_dict['axis_size'])
    # Set the stack spacing
    fig.subplots_adjust(hspace=hrms_plot_dict['stack_headspace'])
    
    #Check for uniform height adjustment - only if uniform scaling was False
    if hrms_plot_dict['uniform_height'] == True and hrms_plot_dict['uniform_scaling'] == False:
        ax_list = scale_to_max_y(ax_list,
                                 hrms_plot_dict['ymajor_ticks'],
                                 hrms_plot_dict['yminor_ticks'])            
        
    # Save the figure
    if hrms_plot_dict['save_output'] == True:
            plt.savefig(hrms_plot_dict['output_name'],
                        facecolor="white",
                        bbox_inches='tight',
                        pad_inches=0.05,
                        dpi=300)
    plt.close()

def round_to_nearest_tick(curr_axis_max,
                          major_tick_size,
                          minor_tick_size):
    """
    A simple helper function that creates plots that can be perfectly stacks
    with no irregular tick marks. Currently assumes that y-axis values start at zero

    Parameters
    ----------
    curr_axis_max : float
        The largest y-axis value in the given dataset
    major_tick_size : float
        The value for the major tick spacing on the y-axis
    minor_tick_size : float
        The value for the major tick spacing on the y-axis

    Returns
    -------
    A float value that is the smallest y-axis value for a given plot that will land
    perfectly on a minor tick-mark when stacked with zero headspace
    """

    return_counter = 0.0
    while return_counter < curr_axis_max:
        return_counter = return_counter + major_tick_size
    #Now counter has overrun by one major tick
    return_counter = return_counter - major_tick_size
    
    while return_counter < curr_axis_max:
        return_counter = return_counter + minor_tick_size
    
    return return_counter

def scale_to_max_y(list_of_matplot_axes,
                   ymajor_tick_size,
                   yminor_tick_size):
    """
    A function that takes a series of stacked plots, and re-scales all of them such that
    they share the same y-axis total scale.    

    Parameters
    ----------
    list_of_matplot_axes : A list of matplotlib axes objects
        As above, the list of axes objects that was generated in plot_multiple_hrms_stack()
    ymajor_tick_size : float
        Self-explanatory
    yminor_tick_size : float
        Self-explanatory

    Returns
    -------
    The list of axes with new ymax values

    """    
    
    max_list = [x.get_ylim()[1] for x in list_of_matplot_axes]
    max_yval = max(max_list)
    
    for single_axis in list_of_matplot_axes:
        single_axis.set_ylim(0, max_yval)
        ymajor_locator = ticker.MultipleLocator(ymajor_tick_size)
        single_axis.yaxis.set_major_locator(ymajor_locator)
        single_axis.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(yminor_tick_size,
                                                                          max_yval + yminor_tick_size,
                                                                          yminor_tick_size)))
    
    return list_of_matplot_axes
    
def define_scaling_ratio(list_of_hrms_spectra,
                         subplot_headspace):
    """
    A function that takes a list of HRMS spectra, and finds the max intensity
    in each spectra. Then, it divides each max by the global max to provide
    the necessary ratios to properly scale all plots

    Parameters
    ----------
    list_of_hrms_spectra : list of dictionaries
        A list, where each entry is a dictionary of the usual form, with the key being
        a formula tuple that points to an ion intensity
    subplot_headspace : float
        The amount of additional y-axis space that each plot receives to incorporate its
        subtitle label

    Returns
    -------
    An ordered list of scaling values to be used in creating variable y-axis height

    """

    max_val_list = [max(single_spectrum, key=lambda x: x[1])[1] for single_spectrum in list_of_hrms_spectra]
    w_title_list = [x + subplot_headspace for x in max_val_list]
    global_max = max(w_title_list)
    ratio_list = [x / global_max for x in w_title_list]
    
    return ratio_list
 
def create_single_ms_bar_from_file(csv_file_directory,
                                   csv_file_name,
                                   hrms_plot_dict):
    """
    A function that takes a single .csv file, and with all of the parameters specified
    in hrms_plot_dict, creates a bar chart of molecular weight vs. ion intensity

    Parameters
    ----------
    csv_file_directory : string
        Name of folder that contains .csv HRMS file of interest
    csv_file_name : string
        Name of specific .csv file that will be plotted
    hrms_plot_dict : dictionary
        A dictionary that contains all of the necessary information
        for creating the plot of interest, which at minimum includes:
            'plot_width' - horizontal size of plot
            'plot_height' - vertical size of plot
            'title' - self-explanatory (s/e)
            'title_size' - font size for title
            'xaxis_label' - s/e
            'yaxis_label' - s/e
            'axis_size' - font size for label
            'axis_scale_size' - font size for axis label scale
            'sublabel_size' - the size of the title for each individual stacked plot
            'font_type' - s/e
            'palette' - s/e
            'num_shades' - how many shades to generate for a given palette
            'shade_choice' - which # shade from the palette to use for the bars
            'xmajor_ticks' - s/e
            'xminor_ticks' - s/e
            'ymajor_ticks' - s/e
            'yminor_ticks' - s/e
            'bar_width' - s/e, standard value is 0.8
            'opacity' - opacity of the plot bars, 1 = opaque 0 = transparent
            'save_output' - Boolean s/e
            'output_name' - s/e, must explicitly include .png
            'label_x_pos_fraction' - how far along the x axis the individual plot axis
                                        its label should appear
            'label_y_pos_fraction' - how for along the y axis the individual plot axis
                                        its label should appear
            'label_headspace' - what amount extra y-axis should be added to each plot to
                                        allow for a label to be added. Default units
                                        is inches.
            'stack_headspace' - what will be the spacing between the stacked plots?
            'uniform_height' - Boolean. If true, before plotting the stacked spectra,
                                        will check to see largest y-axis value, and
                                        rescales all plots to this max scale
            'uniform_scaling' - Boolean. If true, this will re-size the plots so that
                                        the spacing between tick marks is unchanged. (IE
                                        if one plot has a max y-value of 4, and one has
                                        a value of 3, then the first chart should be 33% taller)

    Returns
    -------
    None, but saves plot if desired

    """

    hrms_dict = load_single_ms(csv_file_directory,
                               csv_file_name)
    
    plot_xy_pairs = create_single_hrms_xy_pairs(hrms_dict)
    
    plot_single_ms(plot_xy_pairs,
                   hrms_plot_dict)
    
    return

def create_ms_bar_stack_from_file_list(list_of_csv_file_directories,
                                       list_of_csv_file_names,
                                       list_of_plot_labels,
                                       list_of_yaxis_labels,
                                       list_of_palette_dicts,
                                       hrms_plot_dict):
    """
    A function that takes an unlimited number of MS .csv files and plots them
    in an ordered stack (first file = top of plot). It requires each entry
    to provide, in an ordered list, the file directory, file name, label
    for the sub-plot title, label for the y-axis, and a dictionary describing the 
    palette/shade choice for the plot. There is a single hrms_plot_dict that
    has the settings for the overall construction.

    Parameters
    ----------
    list_of_csv_file_directories : list
        An ordered list that described the file directory containing each .csv MS file (strings)
    list_of_csv_file_names : list
        An ordered list that decsribed the specific file from each directory to
        be processed (strings)
    list_of_plot_labels : list
        An ordered list of strings that will be included as sub-titles for each layer of the plot stack
    list_of_yaxis_labels : list
        An ordered list of strings that will be included as the y-axis label for each plot in the stack
    list_of_palette_dicts : list
        An ordered list of dictionaries, where each dictionary has the information for the color palette and shade
        to use for each plot in the stack.
    hrms_plot_dict : dictionary
        A dictionary that contains all of the necessary information
        for creating the plot of interest, which at minimum includes:
            'plot_width' - horizontal size of plot
            'plot_height' - vertical size of plot
            'title' - self-explanatory (s/e)
            'title_size' - font size for title
            'xaxis_label' - s/e
            'yaxis_label' - s/e
            'axis_size' - font size for label
            'axis_scale_size' - font size for axis label scale
            'sublabel_size' - the size of the title for each individual stacked plot
            'font_type' - s/e
            'palette' - s/e
            'num_shades' - how many shades to generate for a given palette
            'shade_choice' - which # shade from the palette to use for the bars
            'xmajor_ticks' - s/e
            'xminor_ticks' - s/e
            'ymajor_ticks' - s/e
            'yminor_ticks' - s/e
            'bar_width' - s/e, standard value is 0.8
            'opacity' - opacity of the plot bars, 1 = opaque 0 = transparent
            'save_output' - Boolean s/e
            'output_name' - s/e, must explicitly include .png
            'label_x_pos_fraction' - how far along the x axis the individual plot axis
                                        its label should appear
            'label_y_pos_fraction' - how for along the y axis the individual plot axis
                                        its label should appear
            'label_headspace' - what amount extra y-axis should be added to each plot to
                                        allow for a label to be added. Default units
                                        is inches.
            'stack_headspace' - what will be the spacing between the stacked plots?
            'uniform_height' - Boolean. If true, before plotting the stacked spectra,
                                        will check to see largest y-axis value, and
                                        rescales all plots to this max scale
            'uniform_scaling' - Boolean. If true, this will re-size the plots so that
                                        the spacing between tick marks is unchanged. (IE
                                        if one plot has a max y-value of 4, and one has
                                        a value of 3, then the first chart should be 33% taller)

    Returns
    -------
    None, but saves the stacked plot if desired

    """

    list_of_hrms_spectra = create_hrms_xy_list(list_of_csv_file_directories,
                                               list_of_csv_file_names)
    
    plot_multiple_hrms_stack(list_of_hrms_spectra,
                             list_of_plot_labels,
                             list_of_yaxis_labels,
                             list_of_palette_dicts,
                             hrms_plot_dict)
    
    return
    
def create_ms_difference_stack(list_of_csv_file_directories,
                               list_of_csv_file_names,
                               list_of_plot_labels,
                               new_plot_label,
                               list_of_yaxis_labels,
                               new_yaxis_label,
                               list_of_palette_dicts,
                               new_palette_dict,
                               hrms_plot_dict,
                               difference_mode):
    """
    A function that takes two .csv files, and substracts the *first* from the *second*. So, if the second 
    (typically==predicted) spectrum overestimates a peak, the direction of the error will be positive.
    Three possible plotting modes: 'raw', 'absolute' and 'squared'

    Parameters
    ----------
    list_of_csv_file_directories : list
        An ordered list that described the file directory containing each .csv MS file (strings)
    list_of_csv_file_names : list
        An ordered list that decsribed the specific file from each directory to
        be processed (strings)
    list_of_plot_labels : list
        An ordered list of strings that will be included as sub-titles for each layer of the plot stack
    new_plot_label : string
        Name for the 3rd 'difference' plot to add as a sub-title
    list_of_yaxis_labels : list
        An ordered list of strings that will be included as the y-axis label for each plot in the stack
    new_yaxis_label : string
        Title for the y-axis on the 'difference' plot that is created here
    list_of_palette_dicts : list
        An ordered list of dictionaries, where each dictionary has the information for the color palette and shade
        to use for each plot in the stack.
    new_palette_dict : dictionary
        palette/shade information for the 'difference' plot created here
    hrms_plot_dict : dictionary
        A dictionary that contains all of the necessary information
        for creating the plot of interest, which at minimum includes:
            'plot_width' - horizontal size of plot
            'plot_height' - vertical size of plot
            'title' - self-explanatory (s/e)
            'title_size' - font size for title
            'xaxis_label' - s/e
            'yaxis_label' - s/e
            'axis_size' - font size for label
            'axis_scale_size' - font size for axis label scale
            'sublabel_size' - the size of the title for each individual stacked plot
            'font_type' - s/e
            'palette' - s/e
            'num_shades' - how many shades to generate for a given palette
            'shade_choice' - which # shade from the palette to use for the bars
            'xmajor_ticks' - s/e
            'xminor_ticks' - s/e
            'ymajor_ticks' - s/e
            'yminor_ticks' - s/e
            'bar_width' - s/e, standard value is 0.8
            'opacity' - opacity of the plot bars, 1 = opaque 0 = transparent
            'save_output' - Boolean s/e
            'output_name' - s/e, must explicitly include .png
            'label_x_pos_fraction' - how far along the x axis the individual plot axis
                                        its label should appear
            'label_y_pos_fraction' - how for along the y axis the individual plot axis
                                        its label should appear
            'label_headspace' - what amount extra y-axis should be added to each plot to
                                        allow for a label to be added. Default units
                                        is inches.
            'stack_headspace' - what will be the spacing between the stacked plots?
            'uniform_height' - Boolean. If true, before plotting the stacked spectra,
                                        will check to see largest y-axis value, and
                                        rescales all plots to this max scale
            'uniform_scaling' - Boolean. If true, this will re-size the plots so that
                                        the spacing between tick marks is unchanged. (IE
                                        if one plot has a max y-value of 4, and one has
                                        a value of 3, then the first chart should be 33% taller)
    difference_mode : string
        If 'raw', use the true error between spectra (second minus first)
        If 'absolute', turn the raw error into absolute values
        If 'squared', take the square of the raw error

    Returns
    -------
    None, but saves the plot if desired.

    """

    list_of_hrms_spectra = create_difference_data(list_of_csv_file_directories,
                                                  list_of_csv_file_names,
                                                  difference_mode)

    list_of_plot_labels.append(new_plot_label)
    list_of_yaxis_labels.append(new_yaxis_label)
    list_of_palette_dicts.append(new_palette_dict)

    plot_multiple_hrms_stack(list_of_hrms_spectra,
                             list_of_plot_labels,
                             list_of_yaxis_labels,
                             list_of_palette_dicts,
                             hrms_plot_dict)  
    
    return
    
def create_predicted_vs_actual_stack(dataset_param_dict,
                                     network_load_dict,
                                     network_param_dict,
                                     file_to_test,
                                     list_of_plot_labels,
                                     list_of_yaxis_labels,
                                     list_of_palette_dicts,
                                     hrms_plot_dict,
                                     difference_mode):
    """
    A function that takes an extraction HRMS file, as well as a trained network (using the total ion set approach),
    calculates the predicted HRMS, and also the difference between the two (in any of 3 kinds of error)

    Parameters
    ----------
    dataset_param_dict : dictionary
        A dictionary that contains the information necessary to create the pytorch Dataset object, which
        must include:
                sm_file_directory - the folder that contains the starting material file(s)
                ext_file_directory - the folder that contains the extraction HRMS file(s)
                label_keys - a dictionary of keys that describes which starting material was used for each extraction
                test_list - a list of HRMS files that are excluded from the training set (and therefore the total ion set)
                locked_formula - a list of formula relationships that will be included in the network
                condition_dict - a dictionary of extraction conditions
                pickle_file - a Boolean - if True, information about the dataset is pickled and saved
                output_name - used for saving the above pickled information)
    network_load_dict : dictionary
        A short dictionary that contains the folder/file information necessary to load a trained network:
            'trained_net_directory' and 'trained_net_name' being self-explanatory
    network_param_dict : dictionary
        A dictionary that contains the information necessary to load the pre-trained network model,
        which must include:
                layer_size_list - a list that described the width of each intermediate layer in a fully connected network
                starting_width - a parameter that described the width of the starting layer, based on the number of
                formula context points provided
                batch_norm - Boolean, whether to include batch_norm in layers or not
                activation - a pytorch activation function, which will be used in all nodes
                dropout_amt - a float between zero and 1, amout of dropout in all but final layers)
    file_to_test : string
        The name of the specific file which will be plotted in the stack
    list_of_plot_labels : list
        An orderd list that contains the plot labels for the stacked plot sub-titles
    list_of_yaxis_labels : list
        An ordered list that contains the y-axis labels for each plot in the stack
    list_of_palette_dicts : list
        A list of dictionaries describing the palette and shade to use in each sub-plot
    hrms_plot_dict : dictionary
        A dictionary that contains all of the necessary information
        for creating the plot of interest, which at minimum includes:
            'plot_width' - horizontal size of plot
            'plot_height' - vertical size of plot
            'title' - self-explanatory (s/e)
            'title_size' - font size for title
            'xaxis_label' - s/e
            'yaxis_label' - s/e
            'axis_size' - font size for label
            'axis_scale_size' - font size for axis label scale
            'sublabel_size' - the size of the title for each individual stacked plot
            'font_type' - s/e
            'palette' - s/e
            'num_shades' - how many shades to generate for a given palette
            'shade_choice' - which # shade from the palette to use for the bars
            'xmajor_ticks' - s/e
            'xminor_ticks' - s/e
            'ymajor_ticks' - s/e
            'yminor_ticks' - s/e
            'bar_width' - s/e, standard value is 0.8
            'opacity' - opacity of the plot bars, 1 = opaque 0 = transparent
            'save_output' - Boolean s/e
            'output_name' - s/e, must explicitly include .png
            'label_x_pos_fraction' - how far along the x axis the individual plot axis
                                        its label should appear
            'label_y_pos_fraction' - how for along the y axis the individual plot axis
                                        its label should appear
            'label_headspace' - what amount extra y-axis should be added to each plot to
                                        allow for a label to be added. Default units
                                        is inches.
            'stack_headspace' - what will be the spacing between the stacked plots?
            'uniform_height' - Boolean. If true, before plotting the stacked spectra,
                                        will check to see largest y-axis value, and
                                        rescales all plots to this max scale
            'uniform_scaling' - Boolean. If true, this will re-size the plots so that
                                        the spacing between tick marks is unchanged. (IE
                                        if one plot has a max y-value of 4, and one has
                                        a value of 3, then the first chart should be 33% taller)
    difference_mode : string
        If 'raw', use the true error between spectra (second minus first)
        If 'absolute', turn the raw error into absolute values
        If 'squared', take the square of the raw error

    Returns
    -------
    None, but saves the stacked plots if desired.

    """
 
    extraction_dataset = BFN.BitumenExtTISDataset(sm_file_directory=dataset_param_dict['sm_file_directory'],
                                                  ext_file_directory=dataset_param_dict['ext_file_directory'],
                                                  label_keys=dataset_param_dict['label_keys'],
                                                  test_list=dataset_param_dict['test_list'],
                                                  locked_formula=dataset_param_dict['locked_formula'],
                                                  condition_dict=dataset_param_dict['condition_dict'],
                                                  pickle_file=dataset_param_dict['pickle_file'],
                                                  output_name=dataset_param_dict['output_name'])
    
    
    trained_network = BFN.load_extraction_tis_network(network_load_dict['trained_net_directory'],
                                                      network_load_dict['trained_net_name'],
                                                      network_param_dict,
                                                      dataset_param_dict['locked_formula'])
    
    #In extraction dataset, the getitem_list holds the target value at position[5] for creating 'actual' plot
    #The tensor necessary to feed to the trained network is held at position[4].
    #The formula tuple is held at position[2], filename at [0]. From these, a standard 'dictionary' formation can be created
    #That is then processed using standard functions.
    
    actual_plot_dict = {}
    predict_plot_dict = {}
    
    for single_ion in extraction_dataset.test_getitem_list:
        if file_to_test in single_ion[0]:
            actual_plot_dict[single_ion[2]] = single_ion[5].item()
            predict_plot_dict[single_ion[2]] = trained_network(single_ion[4]).item()
    
    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')

    difference_dict = {}
    for formula_tuple in actual_plot_dict:
        difference_dict[formula_tuple] = compute_function(actual_plot_dict[formula_tuple],
                                                          predict_plot_dict[formula_tuple])
    
    list_of_xy_data = [create_single_hrms_xy_pairs(actual_plot_dict),
                       create_single_hrms_xy_pairs(predict_plot_dict),
                       create_single_hrms_xy_pairs(difference_dict)]
    
    plot_multiple_hrms_stack(list_of_xy_data,
                             list_of_plot_labels,
                             list_of_yaxis_labels,
                             list_of_palette_dicts,
                             hrms_plot_dict)  
    
    return

def calc_and_save_ml_predictions(dataset_param_dict,
                                 network_load_dict,
                                 network_param_dict,
                                 file_to_test,
                                 difference_mode,
                                 output_name):
    """
    A function that measures the accuracy of a machine-leanred model on extraction prediction,
    and saves a dictionary of {(mol weight, error), (mol weight, error)} as a pickle file
    with the given name + _ppe, and a list of [(predicted, actual), (predicted, actual)] with
    the given name + _predact

    Parameters
    ----------
    dataset_param_dict : dictionary
        A dictionary that contains the information necessary to create the pytorch Dataset object, which
        must include:
                sm_file_directory - the folder that contains the starting material file(s)
                ext_file_directory - the folder that contains the extraction HRMS file(s)
                label_keys - a dictionary of keys that describes which starting material was used for each extraction
                test_list - a list of HRMS files that are excluded from the training set (and therefore the total ion set)
                locked_formula - a list of formula relationships that will be included in the network
                condition_dict - a dictionary of extraction conditions
                pickle_file - a Boolean - if True, information about the dataset is pickled and saved
                output_name - used for saving the above pickled information)
    network_load_dict : dictionary
        A short dictionary that contains the folder/file information necessary to load a trained network:
            'trained_net_directory' and 'trained_net_name' being self-explanatory
    network_param_dict : dictionary
        A dictionary that contains the information necessary to load the pre-trained network model,
        which must include:
                layer_size_list - a list that described the width of each intermediate layer in a fully connected network
                starting_width - a parameter that described the width of the starting layer, based on the number of
                formula context points provided
                batch_norm - Boolean, whether to include batch_norm in layers or not
                activation - a pytorch activation function, which will be used in all nodes
                dropout_amt - a float between zero and 1, amout of dropout in all but final layers)
    file_to_test : string
        The name of the specific file which will be plotted in the stack
    difference_mode : string
        If 'raw', use the true error between spectra (second minus first)
        If 'absolute', turn the raw error into absolute values
        If 'squared', take the square of the raw error
    output_name : string
        Name to save the assembled dictionaries, as .pkl files
        _predact, and _ppe are appended to the name.
        Predact for making predicted v actual plots, and ppe for making violin plots

    Returns
    -------
    None, but saves the dictionary of interest as a pickle file

    """
    
    extraction_dataset = BFN.BitumenExtTISDataset(sm_file_directory=dataset_param_dict['sm_file_directory'],
                                                  ext_file_directory=dataset_param_dict['ext_file_directory'],
                                                  label_keys=dataset_param_dict['label_keys'],
                                                  test_list=dataset_param_dict['test_list'],
                                                  locked_formula=dataset_param_dict['locked_formula'],
                                                  condition_dict=dataset_param_dict['condition_dict'],
                                                  pickle_file=dataset_param_dict['pickle_file'],
                                                  output_name=dataset_param_dict['output_name'])
    
    
    trained_network = BFN.load_extraction_tis_network(network_load_dict['trained_net_directory'],
                                                      network_load_dict['trained_net_name'],
                                                      network_param_dict,
                                                      dataset_param_dict['locked_formula'])

    #In extraction dataset, the getitem_list holds the target value at position[5] for creating 'actual' plot
    #The tensor necessary to feed to the trained network is held at position[4].
    #The formula tuple is held at position[2], filename at [0]. From these, a standard 'dictionary' formation can be created
    #That is then processed using standard functions.
    
    predact_plot_list = []
    ppe_plot_dict = {}

    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')

    for single_ion in extraction_dataset.test_getitem_list:
        if file_to_test in single_ion[0]:
            actual_val = single_ion[5].item()
            predict_val = trained_network(single_ion[4]).item()
            curr_error = compute_function(actual_val,
                                          predict_val)
            predact_plot_list.append((actual_val, predict_val))
            ppe_plot_dict[single_ion[2]] = curr_error

    predact_pickle_string = output_name + '_predact.pkl'
    predact_pickle_file = open(predact_pickle_string, 'wb')
    pickle.dump(predact_plot_list, predact_pickle_file)
    predact_pickle_file.close()
    
    ppe_pickle_string = output_name + '_ppe.pkl'
    ppe_pickle_file = open(ppe_pickle_string, 'wb')
    pickle.dump(ppe_plot_dict, ppe_pickle_file)
    ppe_pickle_file.close()
    
    return

def create_model_dictionary(trained_net_directory,
                            trained_net_name_list,
                            trained_net_param_dict,
                            dataset_param_dict):
    """
    A function that takes a list of trained Pytorch networks, and creates a network for
    evaluation for each stored in a dictionary. For use in query by committee testing

    Parameters
    ----------
    trained_net_directory : string
        Name of the directory containing trained networks
    trained_net_name_list : list of string
        Name of each file that will be used for evaluation
    trained_net_parameters : dictionary
        A dictionary that contains all of the network information necessary to
        create the trained networks for evaluation
    dataset_param_dict : dictionary
        A dictionary that contains all of the information necessary to make the
        training/testing dataset for evaluation. Even in cases where no training is done,
        the training set (even if empty) is used to create the list of ions to predict

    Returns
    -------
    A dictionary of trained extraction models

    """
    trained_model_dictionary = {}
    
    for specific_net in trained_net_name_list:
        trained_network=BFN.load_extraction_tis_network(trained_net_directory,
                                                        specific_net,
                                                        trained_net_param_dict,
                                                        dataset_param_dict['locked_formula'])
        trained_model_dictionary[specific_net] = trained_network
    
    return trained_model_dictionary

def calc_and_save_ml_by_committee(trained_net_directory,
                                  trained_net_name_list,
                                  trained_net_param_dict,
                                  dataset_param_dict,
                                  file_to_test,
                                  difference_mode,
                                  output_name,
                                  dataset_pass):
    """
    Function for evaluating the performance of a trained network from its saved .state_dict()

    Parameters
    ----------
    trained_net_diretory : TYPE
        DESCRIPTION.
    trained_net_name : TYPE
        DESCRIPTION.
    trained_net_param_dict : TYPE
        DESCRIPTION.
    dataset_param_dict : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.
    csv_output : TYPE
        DESCRIPTION.

    Returns
    -------
    The top level mse and ppe dictionaries, and optionally saves these as pickle files with output_name

    """    
    
    if dataset_pass == None:
        extraction_dataset=BFN.BitumenExtTISDataset(dataset_param_dict['sm_file_directory'],
                                                    dataset_param_dict['ext_file_directory'],
                                                    dataset_param_dict['label_keys'],
                                                    dataset_param_dict['test_list'],
                                                    dataset_param_dict['locked_formula'],
                                                    dataset_param_dict['condition_dict'],
                                                    dataset_param_dict['pickle_file'],
                                                    output_name)
    else:
        extraction_dataset = dataset_pass
    
    trained_net_dictionary = create_model_dictionary(trained_net_directory,
                                                     trained_net_name_list,
                                                     trained_net_param_dict,
                                                     dataset_param_dict)    
    
    predact_plot_list = []
    ppe_plot_dict = {}

    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')

    for single_ion in extraction_dataset.test_getitem_list:
        if file_to_test in single_ion[0]:
            actual_val = single_ion[5].item()
            #Calculate committee prediction
            curr_ion_pred_list = []
            for committee_network in trained_net_dictionary:
                curr_ion_pred_list.append(trained_net_dictionary[committee_network](single_ion[4]).item())
            predict_val = sum(curr_ion_pred_list) / len(curr_ion_pred_list)    

            curr_error = compute_function(actual_val,
                                          predict_val)
            predact_plot_list.append((actual_val, predict_val))
            ppe_plot_dict[single_ion[2]] = curr_error

    predact_pickle_string = output_name + '_predact.pkl'
    predact_pickle_file = open(predact_pickle_string, 'wb')
    pickle.dump(predact_plot_list, predact_pickle_file)
    predact_pickle_file.close()
    
    ppe_pickle_string = output_name + '_ppe.pkl'
    ppe_pickle_file = open(ppe_pickle_string, 'wb')
    pickle.dump(ppe_plot_dict, ppe_pickle_file)
    ppe_pickle_file.close()

    return    
    
def calc_and_save_persistance_prediction(dataset_param_dict,
                                         difference_mode,
                                         file_to_test,
                                         output_name):

    extraction_dataset = BFN.BitumenExtTISDataset(sm_file_directory=dataset_param_dict['sm_file_directory'],
                                                  ext_file_directory=dataset_param_dict['ext_file_directory'],
                                                  label_keys=dataset_param_dict['label_keys'],
                                                  test_list=dataset_param_dict['test_list'],
                                                  locked_formula=dataset_param_dict['locked_formula'],
                                                  condition_dict=dataset_param_dict['condition_dict'],
                                                  pickle_file=dataset_param_dict['pickle_file'],
                                                  output_name=dataset_param_dict['output_name'])

    #In extraction dataset, the getitem_list holds the target value at position[5] for creating 'actual' plot
    #The tensor necessary to feed to the trained network is held at position[4]. In that tensor, the first
    #5 positions are the C,H,N,O,S values - the 6th position holds the starting material value for 'persistance' plots.
    #The formula tuple is held at position[2], filename at [0]. From these, a standard 'dictionary' formation can be created
    #That is then processed using standard functions.
    
    predact_plot_list = []
    ppe_plot_dict = {}

    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')

    for single_ion in extraction_dataset.test_getitem_list:
        if file_to_test in single_ion[0]:
            actual_val = single_ion[5].item()
            persistance_val = single_ion[4][5].item()
            curr_error = compute_function(actual_val,
                                          persistance_val)
            predact_plot_list.append((actual_val, persistance_val))
            ppe_plot_dict[single_ion[2]] = curr_error

    predact_pickle_string = output_name + '_predact.pkl'
    predact_pickle_file = open(predact_pickle_string, 'wb')
    pickle.dump(predact_plot_list, predact_pickle_file)
    predact_pickle_file.close()
    
    ppe_pickle_string = output_name + '_ppe.pkl'
    ppe_pickle_file = open(ppe_pickle_string, 'wb')
    pickle.dump(ppe_plot_dict, ppe_pickle_file)
    ppe_pickle_file.close()

    return

def calc_and_save_average_prediction(dataset_param_dict,
                                     difference_mode,
                                     file_to_test,
                                     output_name):
    
    extraction_dataset = BFN.BitumenExtTISDataset(sm_file_directory=dataset_param_dict['sm_file_directory'],
                                                  ext_file_directory=dataset_param_dict['ext_file_directory'],
                                                  label_keys=dataset_param_dict['label_keys'],
                                                  test_list=dataset_param_dict['test_list'],
                                                  locked_formula=dataset_param_dict['locked_formula'],
                                                  condition_dict=dataset_param_dict['condition_dict'],
                                                  pickle_file=dataset_param_dict['pickle_file'],
                                                  output_name=dataset_param_dict['output_name'])

    #In extraction dataset, the getitem_list holds the target value at position[5] for creating 'actual' plot
    #The tensor necessary to feed to the trained network is held at position[4]. In that tensor, the first
    #5 positions are the C,H,N,O,S values - the 6th position holds the starting material value for 'persistance' plots.
    #The formula tuple is held at position[2], filename at [0]. From these, a standard 'dictionary' formation can be created
    #That is then processed using standard functions.
    
    predact_plot_list = []
    ppe_plot_dict = {}

    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')
    
    for single_ion in extraction_dataset.test_getitem_list:
        if file_to_test in single_ion[0]:
            ion_running_sum_list = []
            actual_val = single_ion[5].item()
            for training_example in extraction_dataset.training_dict:
                if 'SM' not in training_example:
                    try:
                        #If ion was observed, add to running sum list, except add 0.0
                        ion_running_sum_list.append(extraction_dataset.training_dict[training_example][1][single_ion[2]])
                    except:
                        ion_running_sum_list.append(0.0)
            
            average_val = sum(ion_running_sum_list) / len(ion_running_sum_list)
            curr_error = compute_function(actual_val,
                                          average_val)
            predact_plot_list.append((actual_val, average_val))
            ppe_plot_dict[single_ion[2]] = curr_error
    
    predact_pickle_string = output_name + '_predact.pkl'
    predact_pickle_file = open(predact_pickle_string, 'wb')
    pickle.dump(predact_plot_list, predact_pickle_file)
    predact_pickle_file.close()
    
    ppe_pickle_string = output_name + '_ppe.pkl'
    ppe_pickle_file = open(ppe_pickle_string, 'wb')
    pickle.dump(ppe_plot_dict, ppe_pickle_file)
    ppe_pickle_file.close()

    return

def calc_cfgc_phase_ML_difference(trained_net_directory,
                                  trained_net_name_list,
                                  trained_net_param_dict,
                                  dataset_param_dict,
                                  paired_files_to_test,
                                  paired_file_names,
                                  paired_sm_labels,
                                  difference_mode,
                                  output_name,
                                  dataset_pass):
    """
    A function that takes 2 HRMS extraction files - identical solvent conditions in both,
    with a different phase starting material for each - and compares the 'true' difference
    between the extracted batches with the predicted difference between extracted batches

    Parameters
    ----------
    trained_net_directory : TYPE
        DESCRIPTION.
    trained_net_name_list : TYPE
        DESCRIPTION.
    trained_net_param_dict : TYPE
        DESCRIPTION.
    dataset_param_dict : TYPE
        DESCRIPTION.
    paired_files_to_test : TYPE
        DESCRIPTION.
    paired_sm_labels : TYPE
        DESCRIPTION.
    difference_mode : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.
    dataset_pass : TYPE
        DESCRIPTION.

    Returns
    -------
    None, but saves the output as a pickle file of the usual form

    """

    if dataset_pass == None:
        extraction_dataset=BFN.BitumenExtTISDataset(dataset_param_dict['sm_file_directory'],
                                                    dataset_param_dict['ext_file_directory'],
                                                    dataset_param_dict['label_keys'],
                                                    dataset_param_dict['test_list'],
                                                    dataset_param_dict['locked_formula'],
                                                    dataset_param_dict['condition_dict'],
                                                    dataset_param_dict['pickle_file'],
                                                    output_name)
    else:
        extraction_dataset = dataset_pass
    
    trained_net_dictionary = create_model_dictionary(trained_net_directory,
                                                     trained_net_name_list,
                                                     trained_net_param_dict,
                                                     dataset_param_dict)    
    
    predact_plot_list = []
    
    first_ext_file = dataset_param_dict['ext_file_directory'] + '/' + paired_files_to_test[0]
    second_ext_file = dataset_param_dict['ext_file_directory'] + '/' + paired_files_to_test[1]
    
    first_sm_label = dataset_param_dict['label_keys'][paired_file_names[0]]
    second_sm_label = dataset_param_dict['label_keys'][paired_file_names[1]]
    
    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')

    for single_ion in extraction_dataset.total_ion_set:
        pred_1_tensor = CUD.create_tis_extraction_tensor(entry_filename=first_ext_file,
                                                         entry_sm=first_sm_label,
                                                         entry_formula_tuple=single_ion,
                                                         rectified_condition_dict=extraction_dataset.rectified_condition_dict,
                                                         additional_formula_list=dataset_param_dict['locked_formula'],
                                                         normalization_tuple=extraction_dataset.normalization_tuple,
                                                         dataset_dict=extraction_dataset.test_dict)
        pred_2_tensor = CUD.create_tis_extraction_tensor(entry_filename=second_ext_file,
                                                         entry_sm=second_sm_label,
                                                         entry_formula_tuple=single_ion,
                                                         rectified_condition_dict=extraction_dataset.rectified_condition_dict,
                                                         additional_formula_list=dataset_param_dict['locked_formula'],
                                                         normalization_tuple=extraction_dataset.normalization_tuple,
                                                         dataset_dict=extraction_dataset.test_dict)
    
        pred_1_running_list = []
        pred_2_running_list = []
        for trained_model in trained_net_dictionary:
            pred_1_running_list.append(trained_net_dictionary[trained_model](pred_1_tensor).item()) 
            pred_2_running_list.append(trained_net_dictionary[trained_model](pred_2_tensor).item()) 
        
        pred_1_avg = sum(pred_1_running_list) / len(pred_1_running_list)
        pred_2_avg = sum(pred_2_running_list) / len(pred_2_running_list)
        pred_difference = pred_1_avg - pred_2_avg
        
        #Need to do try-except for ions not appearing in extraction spectra
        try:
            true_1_val = extraction_dataset.test_dict[first_ext_file][1][single_ion]
        except:
            true_1_val = 0.0
            
        try:
            true_2_val = extraction_dataset.test_dict[second_ext_file][1][single_ion]
        except:
            true_2_val = 0.0
            
        true_difference = true_1_val - true_2_val
        
        predact_plot_list.append((true_difference, pred_difference))
        
    predact_pickle_string = output_name + '_predact.pkl'
    predact_pickle_file = open(predact_pickle_string, 'wb')
    pickle.dump(predact_plot_list, predact_pickle_file)
    predact_pickle_file.close()

    return    

def calc_cfgc_SM_only_difference(dataset_param_dict,
                                 paired_files_to_test,
                                 paired_file_names,
                                 paired_sm_labels,
                                 difference_mode,
                                 output_name,
                                 dataset_pass):
    """
    A function that compares the difference between Phase 1 SM ion intensities to
    the intensities of those same ions in the extracted fractions, when 2 paired
    extractions were done under identical conditions

    Parameters
    ----------
    dataset_param_dict : TYPE
        DESCRIPTION.
    paired_files_to_test : TYPE
        DESCRIPTION.
    paired_file_names : TYPE
        DESCRIPTION.
    paired_sm_labels : TYPE
        DESCRIPTION.
    difference_mode : TYPE
        DESCRIPTION.
    output_name : TYPE
        DESCRIPTION.
    dataset_pass : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if dataset_pass == None:
        extraction_dataset=BFN.BitumenExtTISDataset(dataset_param_dict['sm_file_directory'],
                                                    dataset_param_dict['ext_file_directory'],
                                                    dataset_param_dict['label_keys'],
                                                    dataset_param_dict['test_list'],
                                                    dataset_param_dict['locked_formula'],
                                                    dataset_param_dict['condition_dict'],
                                                    dataset_param_dict['pickle_file'],
                                                    output_name)
    else:
        extraction_dataset = dataset_pass

    #Set true SM labels for each component
    for poss_sm_1 in extraction_dataset.test_dict.keys():
        sm_string = paired_sm_labels[0] + '_SM'
        if sm_string in poss_sm_1:
            true_sm_1_file = poss_sm_1
            break

    for poss_sm_2 in extraction_dataset.test_dict.keys():
        sm_string = paired_sm_labels[1] + '_SM'
        if sm_string in poss_sm_2:
            true_sm_2_file = poss_sm_2
            break

    predact_plot_list = []
    
    first_ext_file = dataset_param_dict['ext_file_directory'] + '/' + paired_files_to_test[0]
    second_ext_file = dataset_param_dict['ext_file_directory'] + '/' + paired_files_to_test[1]
    
    first_sm_label = dataset_param_dict['label_keys'][paired_file_names[0]]
    second_sm_label = dataset_param_dict['label_keys'][paired_file_names[1]]
    
    if difference_mode == 'raw':
        compute_function = return_raw
    elif difference_mode == 'absolute':
        compute_function = return_absolute
    elif difference_mode == 'square':
        compute_function = return_square
    else:
        raise ValueError('Incorrect compute function chosen for creating difference calculation')

    for single_ion in extraction_dataset.total_ion_set:
        #Need to do try-except for ions not appearing in extraction spectra
        try:        
            sm_1_val = extraction_dataset.test_dict[true_sm_1_file][1][single_ion]
        except:
            sm_1_val = 0.0
        
        try:
            sm_2_val = extraction_dataset.test_dict[true_sm_2_file][1][single_ion]
        except:
            sm_2_val = 0.0
        
        try:
            true_1_val = extraction_dataset.test_dict[first_ext_file][1][single_ion]
        except:
            true_1_val = 0.0
            
        try:
            true_2_val = extraction_dataset.test_dict[second_ext_file][1][single_ion]
        except:
            true_2_val = 0.0
        
        sm_pred_difference = sm_1_val - sm_2_val
        true_difference = true_1_val - true_2_val
        
        predact_plot_list.append((true_difference, sm_pred_difference))
        
    predact_pickle_string = output_name + '_predact.pkl'
    predact_pickle_file = open(predact_pickle_string, 'wb')
    pickle.dump(predact_plot_list, predact_pickle_file)
    predact_pickle_file.close()

    return        

def single_predact_plot_from_pickle(pickle_file_directory_list,
                                    pickle_file_name_list,
                                    predact_plot_dict):
        """Create a scatter plot that displays the predicted vs
        actual data from a pickle file. The pickle file is found
         in pickle_file_directory with the name pickle_file_name.
          The pickle file is a list of the form:
        [(actual, predicted), (actual, predicted), ...]
        The parameters for the plot are found in predact_plot_dict,
            which has the form:
            {'plot_width': the desired width of the plot,
            'plot_height': the desired height of the plot,
            'xaxis_label': the label for the x-axis,
            'yaxis_label': the label for the y-axis,
            'font_type': the font to use for the labels,
            'font_size': the size of the font for the labels,
            'save_output': True or False,
            'output_name': the name of the output file}
            'palette': the color palette to use for the plot
            'num_shades': the number of shades in the color palette
            'shade_choice': the shade of the color palette to use
            'opacity': the opacity of the points in the plot
            'point_size': the size of the points in the plot}
            """
        # Load the pickle files
        # Always assume a list is being loaded. A single file is just len(list) == 1
        predact_list = combine_pickled_data(pickle_file_directory_list,
                                            pickle_file_name_list)
        # Create the figure
        fig, ax = plt.subplots(figsize=(predact_plot_dict['plot_width'],
                                        predact_plot_dict['plot_height']))
        # Create the scatter plot
        x = [x[0] for x in predact_list]
        y = [x[1] for x in predact_list]
        
        palette = sns.color_palette(predact_plot_dict['palette'],
                                    predact_plot_dict['num_shades'])
        point_color = palette[predact_plot_dict['shade_choice']]
        
        ax.scatter(x, y, color=point_color,
                     alpha=predact_plot_dict['opacity'],
                        s=predact_plot_dict['point_size'])
        
        #Round up to nearest minor tick
        #Scale both axes to the max of x/y to create square plots
        max_scale = max([max(x), max(y)])
        min_scale = min([min(x), min(y)])
                
        round_max = round_to_nearest_tick(max_scale,
                                          predact_plot_dict['major_ticks'],
                                          predact_plot_dict['minor_ticks'])
        
        round_min = -1 * (round_to_nearest_tick(-1 * min_scale,
                                                predact_plot_dict['major_ticks'],
                                                predact_plot_dict['minor_ticks']))
        # Set the x-axis label
        ax.set_xlabel(predact_plot_dict['xaxis_label'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])
        # Set the y-axis label
        ax.set_ylabel(predact_plot_dict['yaxis_label'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])

        #Add ticks
        ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(round_min,
                                                                 round_max + predact_plot_dict['major_ticks'],
                                                                 predact_plot_dict['major_ticks'])))
        ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(round_min,
                                                                 round_max + predact_plot_dict['minor_ticks'],
                                                                 predact_plot_dict['minor_ticks'])))
        ax.tick_params(axis='y', labelsize=predact_plot_dict['axis_scale_size'])
        
        ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(round_min,
                                                                 round_max + predact_plot_dict['major_ticks'],
                                                                 predact_plot_dict['major_ticks'])))
        ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(round_min,
                                                                 round_max + predact_plot_dict['minor_ticks'],
                                                                 predact_plot_dict['minor_ticks'])))
        ax.tick_params(axis='x', labelsize=predact_plot_dict['axis_scale_size'])


        # Set the font for the x-axis tick labels
        for label in ax.get_xticklabels():
            label.set_fontname(predact_plot_dict['font_type'])
            label.set_fontsize(predact_plot_dict['font_size'])
        # Set the font for the y-axis tick labels
        for label in ax.get_yticklabels():
            label.set_fontname(predact_plot_dict['font_type'])
            label.set_fontsize(predact_plot_dict['font_size'])
        
        # Set the x-axis limits
        ax.set_xlim(round_min, round_max)
        # Set the y-axis limits
        ax.set_ylim(round_min, round_max)
        # Add x = y trend line
        ax.plot([round_min, round_max],
                [round_min, round_max],
                color='black',
                linewidth=0.25)
        # Set the plot title
        ax.set_title(predact_plot_dict['plot_title'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])
        # Save the plot
        if predact_plot_dict['save_output']:
            plt.savefig(predact_plot_dict['output_name'],
                        facecolor="white",
                        bbox_inches='tight',
                        pad_inches=0.05,
                        dpi=300)
        # Show the plot
        plt.show()

        return

def single_ppe_violin_plot_from_pickle(pickle_file_directory_list,
                                       pickle_file_name_list,
                                       violin_plot_dict):
        """Create a violin plot from a pickled dictionary. The
        dictionary keys have no effect on the plot. Every key
        points to a value that should be included in the violin.
        That is, the form is: {key: value, key: value, ...}
        The parameters for the plot are found in predact_plot_dict,
            which has the form:
            {'plot_width': the desired width of the plot,
            'plot_height': the desired height of the plot,
            'xaxis_label': the label for the x-axis,
            'yaxis_label': the label for the y-axis,
            'font_type': the font to use for the labels,
            'font_size': the size of the font for the labels,
            'save_output': True or False,
            'output_name': the name of the output file}
            'palette': the color palette to use for the plot
            'num_shades': the number of shades in the color palette
            'shade_choice': the shade of the color palette to use
            'opacity': the opacity of the points in the plot
            'point_size': the size of the points in the plot}
        A violin plot may need more parameters than this, please include
        whatever is needed.
            """
        # Always assume a list of files. If only 1 file desired, only 1 entry in list.
        # Load the pickled dictionary
        compiled_data = combine_pickled_data(pickle_file_directory_list,
                                             pickle_file_name_list)
        
        violin_data = pd.DataFrame({'data': list(compiled_data.values())})
        epsilon = 1e-6
        
        violin_data['data'] = violin_data['data'].apply(lambda x: np.log10(x + epsilon))
        
        max_error = violin_data.max()[0]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(violin_plot_dict['plot_width'],
                                        violin_plot_dict['plot_height']))
        
        palette = sns.color_palette(violin_plot_dict['palette'],
                                    violin_plot_dict['num_shades'])
        inner_color_rgb = palette[violin_plot_dict['shade_choice']]
        # Create the violin plot
        sns.violinplot(data=violin_data,
                       y='data',
                       inner=None,
                       color=inner_color_rgb,
                       bw=violin_plot_dict['bandwidth'],
                       cut=violin_plot_dict['cut'],
                       ax=ax)
        
        #Add y-axis ticks
        # ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0,
        #                                                          round_error + violin_plot_dict['major_ticks'],
        #                                                          violin_plot_dict['major_ticks'])))
        # ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,
        #                                                          round_error + violin_plot_dict['minor_ticks'],
        #                                                          violin_plot_dict['minor_ticks'])))
        # ax.tick_params(axis='y', labelsize=violin_plot_dict['axis_scale_size'])
        
        # Set the x-axis label
        ax.set_xlabel(violin_plot_dict['xaxis_label'],
                        fontname=violin_plot_dict['font_type'],
                        fontsize=violin_plot_dict['font_size'])
        # Set the y-axis label
        ax.set_ylabel(violin_plot_dict['yaxis_label'],
                        fontname=violin_plot_dict['font_type'],
                        fontsize=violin_plot_dict['font_size'])
        # Set the font for the x-axis tick labels
        for label in ax.get_xticklabels():
            label.set_fontname(violin_plot_dict['font_type'])
            label.set_fontsize(violin_plot_dict['font_size'])
        # Set the font for the y-axis tick labels
        for label in ax.get_yticklabels():
            label.set_fontname(violin_plot_dict['font_type'])
            label.set_fontsize(violin_plot_dict['font_size'])
        # Save the figure
        if violin_plot_dict['save_output'] == True:
                plt.savefig(violin_plot_dict['output_name'],
                            facecolor="white",
                            bbox_inches='tight',
                            pad_inches=0.05,
                            dpi=300)
        plt.close()
        return

def combine_pickled_data(list_of_file_directories,
                         list_of_file_names):
    """
    Helper function that takes a list of file directories & names (ordered),
    opens and combines the data.
    Current pickled file types are list and dict, and so start by checking
    type so that it can be combined appropriately

    Parameters
    ----------
    list_of_file_directories : list
        An ordered list of string describing the file directory where pickled data lives
    list_of_file_names : list
        An ordered list of strings describing the specific pickle file to open.

    Returns
    -------
    All of the data combined into a single list or dictionary, as appropriate

    """

    if len(list_of_file_directories) != len(list_of_file_names):
        raise ValueError('Mismatch between number of directories:', len(list_of_file_directories),
                         'and number of file names:', len(list_of_file_names))

    with open(list_of_file_directories[0] + list_of_file_names[0], 'rb') as handle:
        initial_data_charge = pickle.load(handle)
    if type(initial_data_charge) == dict:
        initial_data_charge = list(initial_data_charge.values())
        for next_data_entry in range(len(list_of_file_directories) - 1):
            with open(list_of_file_directories[next_data_entry + 1] + list_of_file_names[next_data_entry + 1], 'rb') as handle:
                next_data_charge = pickle.load(handle)
                next_data_charge = list(next_data_charge.values())
                initial_data_charge = initial_data_charge + next_data_charge
    
    elif type(initial_data_charge) == list:
        for next_data_entry in range(len(list_of_file_directories) - 1):
            with open(list_of_file_directories[next_data_entry + 1] + list_of_file_names[next_data_entry + 1], 'rb') as handle:
                next_data_charge = pickle.load(handle)
                initial_data_charge = initial_data_charge + next_data_charge
                
    else:
        raise TypeError('Loaded pickle file is neither a dictionary nor a list')
    return initial_data_charge

def multiple_ppe_violin_plot_from_pickle(list_of_pickle_dict,
                                         violin_plot_dict):
    """
    A function that takes a list of dictionaries, where each dictionary contains the information
    necessary to combine and number of pickled data files (or a single file), and plots a single
    violin plot with all of the data. The overall parameters are held in violing_plot_dict,
    but each dictionary also contains specific information about the shade to be used for a given
    violin as well as its x-axis label.

    Parameters
    ----------
    list_of_pickle_dict : list of dictionaries
        An ordered list (horizontal across the x-axis), which contains all the information necessary
        for each violin of the plot held in a dictionary
    violin_plot_dict : dictionary
        A dictionary, which holds at a minimum:
            'plot_width': 
            'plot_height': 
            'palette': 
            'num_shades': 
            'shade_choice': 
            'opacity': 
            'point_size': 
            'xaxis_label': 
            'yaxis_label': 
            'font_type': 
            'font_size': 
            'axis_scale_size': 
            'plot_title': 
            'major_ticks': 
            'minor_ticks': 
            'bandwidth': 
            'cut': 
            'save_output': 
            'output_name':
            'multi-plot scale'
            'xaxis_rotation': 
    Returns
    -------
    None, but saves the plot output if desired

    """

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(violin_plot_dict['plot_width'], violin_plot_dict['plot_height']))
    
    #Combine data
    combined_dataframe = pd.DataFrame()
    total_violins = len(list_of_pickle_dict)
    label_list = []
    color_list = []
    
    for entry in list_of_pickle_dict:
        label_list.append(entry['title'])
        compiled_data = combine_pickled_data(entry['directory_list'], entry['file_list'])
        print('For entry:', entry['title'], 'length of data is:', len(compiled_data))
        violin_data = pd.DataFrame({entry['title']: compiled_data})
        
        if violin_plot_dict['multi-plot scale'] == 'squared':
            violin_data[entry['title']] = violin_data[entry['title']].apply(lambda x: x**2)
        epsilon = 1e-6            
        violin_data[entry['title']] = violin_data[entry['title']].apply(lambda x: np.log10(x + epsilon))
        
        combined_dataframe[entry['title']] = violin_data[entry['title']]
        palette = sns.color_palette(entry['palette'],
                                    entry['num_shades'])
        inner_color_rgb = palette[entry['shade_choice']]
        color_list.append(inner_color_rgb)
        
    long_combined_dataframe = combined_dataframe.melt(var_name='label', value_name='data')
    # Set x-axis ticks and titles
    x_tick_positions = np.arange(total_violins)
    x_tick_labels = label_list
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
        
    # Plot
    sns.violinplot(data=long_combined_dataframe, x=long_combined_dataframe['label'], y=long_combined_dataframe['data'],
                   hue='label', palette=color_list, bw=violin_plot_dict['bandwidth'],
                   cut=violin_plot_dict['cut'], inner=violin_plot_dict['inner'],
                   dodge=False)

    # Set the font type and size
    plt.rcParams['font.family'] = violin_plot_dict['font_type']
    plt.rcParams['font.size'] = violin_plot_dict['font_size']
    # Set the y-axis label
    ax.set_ylabel(violin_plot_dict['yaxis_label'])
    # Set the x-axis label
    ax.set_xlabel(violin_plot_dict['xaxis_label'])
    # Set the x-axis tick label rotation
    plt.xticks(rotation=violin_plot_dict['xaxis_rotation'])
    # Set the y-axis tick label rotation
    plt.yticks(rotation=violin_plot_dict['yaxis_rotation'])
    # Set the y-axis tick label font size
    ax.tick_params(axis='y', labelsize=violin_plot_dict['axis_scale_size'])
    # Set the x-axis tick label font size
    ax.tick_params(axis='x', labelsize=violin_plot_dict['axis_scale_size'])
    
    # Set the y-axis labels and ticks
    # Round up to nearest minor tick
    desired_max = (max(long_combined_dataframe['data']) + violin_plot_dict['yaxis_headspace'])
    
    round_ylim = round_to_nearest_tick(desired_max,
                                       violin_plot_dict['major_ticks'],
                                       violin_plot_dict['minor_ticks'])
    
    #Add y-axis headroom so that a label can be appended
    ax.set_ylim(min(long_combined_dataframe['data']), round_ylim)

    # Remove legend if desired
    if violin_plot_dict['legend'] == False:
        ax.legend().remove()
        
    # If save_output is true, save the figure as a .png file
    if violin_plot_dict['save_output']:
        plt.savefig(violin_plot_dict['output_name'], dpi=300, bbox_inches='tight')

    return long_combined_dataframe
    
def multiple_predact_scatter_from_pickel(list_of_pickle_dict,
                                         scatter_plot_dict):
    """
    A function that takes multiple sets of existing data and plots them as a
    scatter plot, optionally with different colors, shapes, etc.

    Parameters
    ----------
    list_of_pickle_dict : list of dictionaries
        An ordered list (descending in plot legend order) of data to scatter plot
        Each dictionary minimally contains:
            'title': the name for the title that will appear in the legend
            'directory_list': the name of the file directory to find each file that will
                                be combined into this scatter plot
            'file_list': the name of the actual .pkl files that will be combined in each scatter
    scatter_plot_dict : dictionary
        A dictionary containing all of the necessary parameters for plotting a predicted
        vs. actual scatter plot, which minimally includes:

    Returns
    -------
    None, but saves the plot if desired
    
    """    
    
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(scatter_plot_dict['plot_width'], scatter_plot_dict['plot_height']))

    list_of_markers = []
    list_of_titles = []
    running_max_val = None
    running_min_val = None
    
    for entry in list_of_pickle_dict:
        list_of_titles.append(entry['title'])
        compiled_data = combine_pickled_data(entry['directory_list'], entry['file_list'])
        palette = sns.color_palette(entry['palette'],
                                    entry['num_shades'])
        marker_color = palette[entry['shade_choice']]
        x_values = [point[0] for point in compiled_data]
        y_values = [point[1] for point in compiled_data]
        
        marker = ax.scatter(x_values,y_values,color=marker_color,
                            alpha=scatter_plot_dict['opacity'],
                               s=scatter_plot_dict['point_size'])
        list_of_markers.append(marker)

        curr_max = max(max(x_values), max(y_values))
        curr_min = min(min(x_values), min(y_values))
    
        if running_max_val == None:
            running_max_val = curr_max
        else:
            if curr_max > running_max_val:
                running_max_val = curr_max
        
        if running_min_val == None:
            running_min_val = curr_min
        else:
            if curr_min < running_min_val:
                running_min_val = curr_min
    
    round_max = round_to_nearest_tick(running_max_val,
                                      scatter_plot_dict['major_ticks'],
                                      scatter_plot_dict['minor_ticks'])
    
    round_min = -1 * (round_to_nearest_tick(-1 * running_min_val,
                                            scatter_plot_dict['major_ticks'],
                                            scatter_plot_dict['minor_ticks']))
    
    ax.legend(list_of_markers, list_of_titles)

    # Set the font type and size
    plt.rcParams['font.family'] = scatter_plot_dict['font_type']
    plt.rcParams['font.size'] = scatter_plot_dict['font_size']
    # Set the y-axis label
    ax.set_ylabel(scatter_plot_dict['yaxis_label'])
    # Set the x-axis label
    ax.set_xlabel(scatter_plot_dict['xaxis_label'])
    # Set the y-axis tick label font size
    ax.tick_params(axis='y', labelsize=scatter_plot_dict['axis_scale_size'])
    # Set the x-axis tick label font size
    ax.tick_params(axis='x', labelsize=scatter_plot_dict['axis_scale_size'])
    
    #Add ticks
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(round_min,
                                                             round_max + scatter_plot_dict['major_ticks'],
                                                             scatter_plot_dict['major_ticks'])))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(round_min,
                                                             round_max + scatter_plot_dict['minor_ticks'],
                                                             scatter_plot_dict['minor_ticks'])))
    ax.tick_params(axis='y', labelsize=scatter_plot_dict['axis_scale_size'])
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(round_min,
                                                             round_max + scatter_plot_dict['major_ticks'],
                                                             scatter_plot_dict['major_ticks'])))
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(round_min,
                                                             round_max + scatter_plot_dict['minor_ticks'],
                                                             scatter_plot_dict['minor_ticks'])))
    ax.tick_params(axis='x', labelsize=scatter_plot_dict['axis_scale_size'])
    
    # Set the x-axis limits
    if scatter_plot_dict['limit_override'] == True:
        ax.set_xlim(scatter_plot_dict['xmin'], scatter_plot_dict['xmax'])
        ax.set_ylim(scatter_plot_dict['ymin'], scatter_plot_dict['ymax'])
        ax.plot([scatter_plot_dict['xmin'], scatter_plot_dict['xmax']],
                [scatter_plot_dict['ymin'], scatter_plot_dict['ymax']],
                color='black',
                linewidth=0.25)
    else:
        ax.set_xlim(round_min, round_max)
        # Set the y-axis limits
        ax.set_ylim(round_min, round_max)
        # Add x = y trend line
        ax.plot([round_min, round_max],
                [round_min, round_max],
                color='black',
                linewidth=0.25)
    # Set the plot title
    ax.set_title(scatter_plot_dict['plot_title'],
                 fontname=scatter_plot_dict['font_type'],
                 fontsize=scatter_plot_dict['font_size'])
    # Save the plot
    if scatter_plot_dict['save_output']:
        plt.savefig(scatter_plot_dict['output_name'],
                    facecolor="white",
                    bbox_inches='tight',
                    pad_inches=0.05,
                    dpi=300)
    # Show the plot
    plt.show()

    return
                
def compare_mse_for_scatters(list_of_pickle_dict):
    """
    A function that takes a list of data sources (can reuse same form as from
    multiple_scatter_from_pickle) and computes and prints the mean squared
    error for each data source

    Parameters
    ----------
    list_of_pickle_dict : list of dictionaries
        An ordered list (descending in plot legend order) of data to scatter plot
        Each dictionary minimally contains:
            'title': the name for the title that will appear in the legend
            'directory_list': the name of the file directory to find each file that will
                                be combined into this scatter plot
            'file_list': the name of the actual .pkl files that will be combined in each scatter

    Returns
    -------
    None, but prints the metrics

    """
    for entry in list_of_pickle_dict:
        compiled_data = combine_pickled_data(entry['directory_list'], entry['file_list'])
        running_mse_list = []
        for scatter_point in compiled_data:
            curr_error = (scatter_point[0] - scatter_point[1])**2
            running_mse_list.append(curr_error)
        actual_mse = sum(running_mse_list) / len(running_mse_list)
        print('MSE for', entry['title'], 'is:', str(actual_mse))
    
    return

def create_umap_cluster_frame(umap_cluster_dict,
                              umap_constant_settings,
                              n_neighbors,
                              min_dist):
    """
    

    Parameters
    ----------
    csv_file_directory : TYPE
        DESCRIPTION.
    csv_file_name : TYPE
        DESCRIPTION.
    umap_cluster_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    cluster_data_frame = UMP.import_csv_and_craft(umap_cluster_dict['csv_file_directory'],
                                                  umap_cluster_dict['csv_file_name'],
                                                  umap_cluster_dict['target_column_list'],
                                                  umap_cluster_dict['data_column_list'],
                                                  umap_cluster_dict['target_mode'],
                                                  umap_cluster_dict['data_mode'])
    
    reducer = umap.UMAP(**umap_constant_settings,
                        n_neighbors=n_neighbors,
                        min_dist=min_dist)
    
    #Remove labels from dataframe
    data_only_frame = cluster_data_frame.drop(columns=[col for col in cluster_data_frame.columns if col in umap_cluster_dict['target_column_list']])

    #Create x and y plot values from UMAP
    scatter_data = reducer.fit_transform(data_only_frame)

    cluster_data_frame['X'] = scatter_data[:, 0]
    cluster_data_frame['Y'] = scatter_data[:, 1]
                         
    return cluster_data_frame

def create_umap_scatter(umap_cluster_dataframe,
                        scatter_plot_dict):
    """
    

    Parameters
    ----------
    umap_cluster_dataframe : TYPE
        DESCRIPTION.
    scatter_plot_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """                         
    #Extract scatter points                             
    x_val = umap_cluster_dataframe['X']
    y_val = umap_cluster_dataframe['Y']
    
    #Set up scatter plot
    fig, ax = plt.subplots(figsize=(scatter_plot_dict['plot_width'],
                                    scatter_plot_dict['plot_height']))

    #Set up marker color. 'palette', 'num_shades', and 'shade_choice' have the structure of a dictionary of dictionaries
    #The first key in the dictionary is the column name to search. The second key maps a value found in that column to
    #the values of interest
    
    color_column = list(scatter_plot_dict['marker_colors'])[0]
    marker_color_list = [sns.color_palette(scatter_plot_dict['marker_colors'][color_column][marker_value][0],
                          scatter_plot_dict['marker_colors'][color_column][marker_value][1])[scatter_plot_dict['marker_colors'][color_column][marker_value][2]] for marker_value in umap_cluster_dataframe[color_column]]

    opacity_column = list(scatter_plot_dict['marker_opacity'])[0]
    opacity_list = [scatter_plot_dict['marker_opacity'][opacity_column][marker_value] for marker_value in umap_cluster_dataframe[opacity_column]]

    marker_type_column = list(scatter_plot_dict['marker_type'])[0]
    marker_type_list = [scatter_plot_dict['marker_type'][marker_type_column][marker_value] for marker_value in umap_cluster_dataframe[marker_type_column]]
   
    #Iterate over points to make the plot
    for x, y, color, opacity, marker_type in zip(x_val, y_val, marker_color_list, opacity_list, marker_type_list):
        plt.scatter(x, y, color=color, alpha=opacity, marker=marker_type)

    return marker_color_list                             

def update_scatter_ax(ax,
                      umap_cluster_dataframe,
                      umap_scatter_dict):
    """
    The umap_scatter_dict needs to have the following structure:
        The key 'marker_colors' should map to a dictionary, whose keys correspond to the column names
        that you're using to define the colors. Each of these inner keys should then map to another
        dictionary that assigns each unique value in the column to a list of 3 elements:
        The Seaborn palette name (e.g., 'husl')
        The number of shades in the palette
        The index for which shade to choose in the palette

    The key 'marker_opacity' should similarly map to a dictionary.
    The inner keys should map to the unique values in the corresponding DataFrame column
    and set an opacity value (between 0 and 1) for each unique value.

    The key 'marker_type' follows the same pattern, mapping to a dictionary
    whose inner keys correspond to the unique values in a DataFrame column.
    Each inner key should map to a marker type (e.g., 'o', 's', 'x', etc.)
    """
    # Extract scatter points
    x_val = umap_cluster_dataframe['X']
    y_val = umap_cluster_dataframe['Y']
    
    # Set up marker color
    color_column = list(umap_scatter_dict['marker_colors'])[0]
    marker_color_list = [
        sns.color_palette(
            umap_scatter_dict['marker_colors'][color_column][marker_value][0],
            umap_scatter_dict['marker_colors'][color_column][marker_value][1]
        )[umap_scatter_dict['marker_colors'][color_column][marker_value][2]]
        for marker_value in umap_cluster_dataframe[color_column]
    ]
    
    # Set up marker opacity
    opacity_column = list(umap_scatter_dict['marker_opacity'])[0]
    opacity_list = [
        umap_scatter_dict['marker_opacity'][opacity_column][marker_value]
        for marker_value in umap_cluster_dataframe[opacity_column]
    ]
    
    # Set up marker type
    marker_type_column = list(umap_scatter_dict['marker_type'])[0]
    marker_type_list = [
        umap_scatter_dict['marker_type'][marker_type_column][marker_value]
        for marker_value in umap_cluster_dataframe[marker_type_column]
    ]
    
    # Iterate over points to make the plot
    for x, y, color, opacity, marker_type in zip(x_val, y_val, marker_color_list, opacity_list, marker_type_list):
        ax.scatter(x, y, color=color, alpha=opacity, marker=marker_type)
    
    return ax

def create_multiple_UMAP_scatter(multi_scatter_plot_dict,
                                 umap_scatter_dict,
                                 constant_umap_settings,
                                 list_of_cluster_targets,
                                 output_file_name):
    """
    Docstring
    """
    # Create the figure and axes
    fig, multi_ax = plt.subplots(nrows=multi_scatter_plot_dict['num_rows'],
                                 ncols=multi_scatter_plot_dict['num_columns'],
                                 figsize=(multi_scatter_plot_dict['plot_width'],
                                          multi_scatter_plot_dict['plot_height']),
                                          gridspec_kw={'wspace': 0, 'hspace': 0})
    
    for i, ax_row in enumerate(multi_ax):
        for j, ax in enumerate(ax_row):
            #Create individual scatter plot by updating ax
            cluster_frame = create_umap_cluster_frame(list_of_cluster_targets[j],
                                                      constant_umap_settings,
                                                      multi_scatter_plot_dict['list_of_UMAP_settings'][i]['n_neighbors'],
                                                      multi_scatter_plot_dict['list_of_UMAP_settings'][i]['min_dist'])
            
            ax = update_scatter_ax(ax,
                                   cluster_frame,
                                   umap_scatter_dict)
            
            #Add row title at first column
            if j == 0:
                ax.annotate(multi_scatter_plot_dict['list_of_row_labels'][i],
                            xy=(0, 0.5),
                            xycoords=ax.yaxis.label,
                            textcoords='offset points',
                            ha='right', va='center')
            
            #Add column title at top row
            if i == 0:
                ax.set_title(multi_scatter_plot_dict['list_of_column_labels'][j], loc='center')
            
            #Remove all axis tick-marks and label
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    #Set full figure background as white
    fig.patch.set_facecolor('white')        
    #If output_file_name is not none, save the plot as a .png file with output_file_name.png
    if output_file_name != None:
        plt.savefig(output_file_name + '.png', dpi=300, bbox_inches='tight')
    
    return

def create_consistent_multiUMAP_scatter(multi_scatter_plot_dict,
                                        umap_scatter_dict,
                                        constant_umap_settings,
                                        nested_multiscatter_dict,
                                        list_of_cluster_targets):
    """
    Docstring
    """
    # Do a dummy loop to create the original UMAP x/y coordinates.
    # The looping order should be consistent, so using a list to store order
    # Should be sufficient
    fig, multi_ax = plt.subplots(nrows=multi_scatter_plot_dict['num_rows'],
                                 ncols=multi_scatter_plot_dict['num_columns'],
                                 figsize=(multi_scatter_plot_dict['plot_width'],
                                          multi_scatter_plot_dict['plot_height']),
                                          gridspec_kw={'wspace': 0, 'hspace': 0})
    
    list_of_umap_frames = []

    for i, ax_row in enumerate(multi_ax):
        for j, ax in enumerate(ax_row):
            cluster_frame = create_umap_cluster_frame(list_of_cluster_targets[j],
                                                      constant_umap_settings,
                                                      multi_scatter_plot_dict['list_of_UMAP_settings'][i]['n_neighbors'],
                                                      multi_scatter_plot_dict['list_of_UMAP_settings'][i]['min_dist'])

            list_of_umap_frames.append(cluster_frame)

    # Now, for every set of settings in nested_multiscatter_dict, create and save
    # The actual desired plots
    for setting in nested_multiscatter_dict:
        # Create the figure and axes
        fig, multi_ax = plt.subplots(nrows=multi_scatter_plot_dict['num_rows'],
                                        ncols=multi_scatter_plot_dict['num_columns'],
                                        figsize=(multi_scatter_plot_dict['plot_width'],
                                                multi_scatter_plot_dict['plot_height']),
                                                gridspec_kw={'wspace': 0, 'hspace': 0})
        for i, ax_row in enumerate(multi_ax):
            for j, ax in enumerate(ax_row):
                #Create individual scatter plot by updating ax
                cluster_frame = list_of_umap_frames[i * multi_scatter_plot_dict['num_columns'] + j]
                ax = update_scatter_ax(ax,
                                       cluster_frame,
                                       nested_multiscatter_dict[setting]['umap_settings'])
                
                #Add row title at first column
                if j == 0:
                    ax.annotate(multi_scatter_plot_dict['list_of_row_labels'][i],
                                xy=(0, 0.5),
                                xycoords=ax.yaxis.label,
                                textcoords='offset points',
                                ha='right', va='center')
                
                #Add column title at top row
                if i == 0:
                    ax.set_title(multi_scatter_plot_dict['list_of_column_labels'][j], loc='center')
                
                #Remove all axis tick-marks and label
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        
        #Set full figure background as white
        fig.patch.set_facecolor('white')        
    
        #If output_file_name is not none, save the plot as a .png file with output_file_name.png
        if nested_multiscatter_dict[setting]['output_name'] != None:
            plt.savefig(nested_multiscatter_dict[setting]['output_name'] + '.png', dpi=300, bbox_inches='tight')
    
    return
    


