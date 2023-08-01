import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns

def single_predact_plot_from_pickle(pickle_file_directory,
                                    pickle_file_name,
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
        # Load the pickle file
        predact_list = pd.read_pickle(pickle_file_directory + pickle_file_name)
        # Create the figure
        fig, ax = plt.subplots(figsize=(predact_plot_dict['plot_width'],
                                        predact_plot_dict['plot_height']))
        # Create the scatter plot
        x = [x[0] for x in predact_list]
        y = [x[1] for x in predact_list]
        ax.scatter(x, y, color=predact_plot_dict['palette'][predact_plot_dict['shade_choice']],
                     alpha=predact_plot_dict['opacity'],
                        s=predact_plot_dict['point_size'])
        # Set the x-axis label
        ax.set_xlabel(predact_plot_dict['xaxis_label'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])
        # Set the y-axis label
        ax.set_ylabel(predact_plot_dict['yaxis_label'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])
        # Set the x-axis major ticks
        ax.set_xticks(predact_plot_dict['major_ticks'])
        # Set the x-axis minor ticks
        ax.set_xticks(predact_plot_dict['minor_ticks'], minor=True)
        # Set the y-axis major ticks
        ax.set_yticks(predact_plot_dict['major_ticks'])
        # Set the y-axis minor ticks
        ax.set_yticks(predact_plot_dict['minor_ticks'], minor=True)
        # Set the font for the x-axis tick labels
        for label in ax.get_xticklabels():
            label.set_fontname(predact_plot_dict['font_type'])
            label.set_fontsize(predact_plot_dict['font_size'])
        # Set the font for the y-axis tick labels
        for label in ax.get_yticklabels():
            label.set_fontname(predact_plot_dict['font_type'])
            label.set_fontsize(predact_plot_dict['font_size'])
        # Set the x-axis limits
        ax.set_xlim(predact_plot_dict['xaxis_min'], predact_plot_dict['xaxis_max'])
        # Set the y-axis limits
        ax.set_ylim(predact_plot_dict['yaxis_min'], predact_plot_dict['yaxis_max'])
        # Set the plot title
        ax.set_title(predact_plot_dict['plot_title'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])
        #Set output dpi to 300
        plt.rcParams['savefig.dpi'] = 300
        # Save the plot
        if predact_plot_dict['save_output']:
            plt.savefig(predact_plot_dict['output_name'])
        # Show the plot
        plt.show()

        return



