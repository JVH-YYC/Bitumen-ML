import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import seaborn as sns

def single_predact_plot_from_pickle(pickle_file_directory,
                                    pickle_file_name,
                                    predact_plot_dict):
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
        # Load the pickled dictionary
        with open(pickle_file_directory + pickle_file_name, 'rb') as handle:
                predact_dict = pickle.load(handle)
        # Create the figure
        fig, ax = plt.subplots(figsize=(predact_plot_dict['plot_width'],
                                        predact_plot_dict['plot_height']))
        # Create the violin plot
        sns.violinplot(data=predact_dict,
                        palette=sns.color_palette(predact_plot_dict['palette'],
                                                    predact_plot_dict['num_shades']),
                        inner=predact_plot_dict['shade_choice'],
                        ax=ax)
        # Set the x-axis label
        ax.set_xlabel(predact_plot_dict['xaxis_label'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])
        # Set the y-axis label
        ax.set_ylabel(predact_plot_dict['yaxis_label'],
                        fontname=predact_plot_dict['font_type'],
                        fontsize=predact_plot_dict['font_size'])
        # Set the x-axis tick labels
        ax.set_xticklabels(ax.get_xticklabels(),
                            fontname=predact_plot_dict['font_type'],
                            fontsize=predact_plot_dict['font_size'])
        # Set the y-axis tick labels
        ax.set_yticklabels(ax.get_yticklabels(),
                            fontname=predact_plot_dict['font_type'],
                            fontsize=predact_plot_dict['font_size'])
        # Save the figure
        if predact_plot_dict['save_output'] == True:
                plt.savefig(predact_plot_dict['output_name'])
        plt.close()
        return

