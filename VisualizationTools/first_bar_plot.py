import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns
import sys

def create_single_hrms_plot(hrms_ion_list,
                            hrms_plot_dict):
        """Create a plot of the HRMS spectrum of a single sample.
        The input data is held in hrms_ion_list which has the form:
        [(m/z, intensity), (m/z, intensity), ...]
        The parameters for the plot are found in hrms_plot_dict, which has the form:
        {'title': the title for the plot,
        'plot_widht': the desired width of the plot,
        'plot_height': the desired height of the plot,
        'xaxis_label': the label for the x-axis,
        'yaxis_label': the label for the y-axis,
        'font_type': the font to use for the labels,
        'font_size': the size of the font for the labels,
        'bar_color': the color to use for the bars,
        'major_ticks': the major tick marks,
        'minor_ticks': the minor tick marks,
        'save_output': True or False,
        'output_name': the name of the output file}
        """
        # Create the figure
        fig, ax = plt.subplots(figsize=(hrms_plot_dict['plot_width'],
                                        hrms_plot_dict['plot_height']))
        # Create the bar plot
        x = [x[0] for x in hrms_ion_list]
        y = [x[1] for x in hrms_ion_list]
        ax.bar(x, y, color=hrms_plot_dict['bar_color'])
        # Set the title
        ax.set_title(hrms_plot_dict['title'],
                        fontname=hrms_plot_dict['font_type'],
                        fontsize=hrms_plot_dict['font_size'])
        # Set the x-axis label
        ax.set_xlabel(hrms_plot_dict['xaxis_label'],
                        fontname=hrms_plot_dict['font_type'],
                        fontsize=hrms_plot_dict['font_size'])
        # Set the y-axis label
        ax.set_ylabel(hrms_plot_dict['yaxis_label'],
                        fontname=hrms_plot_dict['font_type'],
                        fontsize=hrms_plot_dict['font_size'])
        # Set the x-axis major ticks
        ax.set_xticks(hrms_plot_dict['major_ticks'])
        # Set the x-axis minor ticks
        ax.set_xticks(hrms_plot_dict['minor_ticks'], minor=True)
        # Save the figure
        if hrms_plot_dict['save_output'] == True:
                plt.savefig(hrms_plot_dict['output_name'])
        plt.close()
        return

