import numpy as np
import pandas as pd

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

class DifferenceVisualizer:
    """
    A class for visualizing data with latitude-longitude scatter plots and distance histograms.
    """
    @staticmethod
    def calculate_rms_distance(df, distance_col):
        """
        Calculate the RMS for a given distance column in the DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            distance_col (str): Column name for the distance values.

        Returns:
            float: The RMS distance.
        """
        return np.sqrt(np.mean(df[distance_col] ** 2))

    @staticmethod
    def plot_lat_lon_differences(df, loc_1, loc_2, name):
        """
        Plots the differences in latitude and longitude between two sets of coordinates using Seaborn,
        with KDE contours and a colorbar for the KDE.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            loc_1 (list): List of two column names for the first set of lat and lon. Format: [latitude, longitude].
            loc_2 (list): List of two column names for the second set of lat and lon. Format: [latitude, longitude].
            name (str): Name to be used in the title of the plot.

        Returns:
            None: Displays the scatter plot with KDE contours and colorbar.
        """
        
        # Calculate the differences between the two sets of coordinates using .loc
        df.loc[:, 'lat_diff'] = df.loc[:, loc_1[0]] - df.loc[:, loc_2[0]]
        df.loc[:, 'lon_diff'] = df.loc[:, loc_1[1]] - df.loc[:, loc_2[1]]
        # Set up the figure
        plt.figure(figsize=(10, 6))

        # Create the KDE plot with contours using Seaborn
        kde = sns.kdeplot(
            data=df, x='lat_diff', y='lon_diff', cmap='viridis', levels=10, thresh=0.05
        )

        # Add a scatter plot for the individual points
        sns.scatterplot(
            data=df, x='lat_diff', y='lon_diff', color='blue', s=50, alpha=0.6, edgecolor='black'
        )

        # Create the colorbar for the KDE plot
        cbar = plt.colorbar(kde.collections[-1], label='Density')

        # Labeling and display
        plt.xlabel('Latitude Difference (degrees)')
        plt.ylabel('Longitude Difference (degrees)')
        
        # Update the title to include the number of matches
        plt.title(f'Scatter Plot of Latitude and Longitude Differences, Match: {name} ({len(df)} matches)')
        plt.grid(True)
        plt.show()

    
    @staticmethod
    def plot_distance_histogram_with_pdf(df, distance_col, name, bins=None):
        """
        Create a histogram with a PDF (kde) overlay of the distance between targets and annotate with RMS.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            distance_col (str): Column name for the distance values to be plotted on the x-axis.
            name (str): Name to be used in the title of the plot.
            bins (int, optional): Number of bins for the histogram. If not provided, uses Freedman-Diaconis rule.

        Returns:
            None: Displays the histogram plot with PDF and RMS annotation.
        """
        # Calculate RMS for the distance
        rms_distance = DifferenceVisualizer.calculate_rms_distance(df, distance_col)

        # Calculate the number of bins using the Freedman-Diaconis rule if bins is None
        distance_data = df[distance_col]
        if bins is None:
            q25, q75 = np.percentile(distance_data, [25, 75])
            iqr = q75 - q25
            bin_width = 2 * iqr / np.cbrt(len(distance_data))
            bins = int(np.ceil((distance_data.max() - distance_data.min()) / bin_width))

        # Create a histogram with count on y-axis and PDF overlay
        plt.figure(figsize=(10, 6))
        sns.histplot(distance_data, kde=True, color='skyblue', bins=bins, stat="count", label=f'Distance with PDF (n = {len(df)})')
        
        # Annotate the RMS on the plot
        plt.axvline(rms_distance, color='red', linestyle='--', label=f'RMS = {rms_distance:.4f}')
        plt.text(rms_distance, plt.ylim()[1] * 0.8, f'RMS: {rms_distance:.4f}', color='red', fontsize=12, ha='center')

        # Titles and labels
        plt.title(f'Histogram of Distances with PDF, Match: {name} ({len(df)} matches)')
        plt.xlabel('Distance (km)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def dataframe_comparisons(dataframes_dict, id1, id2):
        """
        Compares id1 and id2 columns between all pairs of DataFrames in the dictionary and prints the differences.

        Parameters:
        - dataframes_dict (dict): Dictionary of DataFrames with the same structure.
        - id1 (str): Name of the first ID column to evaluate.
        - id2 (str): Name of the second ID column to evaluate.
        """

        # Iterate through all pairs of DataFrames
        for (df_name1, df_name2) in combinations(dataframes_dict.keys(), 2):
            df1 = dataframes_dict[df_name1]
            df2 = dataframes_dict[df_name2]

            # Merge on id1 and id2 to find common and differing pairs
            merged = pd.merge(df1, df2, on=[id1, id2], how='outer', indicator=True)

            # Separate different types of matches
            only_in_df1 = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
            only_in_df2 = merged[merged['_merge'] == 'right_only'].drop('_merge', axis=1)
            in_both = merged[merged['_merge'] == 'both'].drop('_merge', axis=1)

            # Print comparison results
            print(f"\nComparison between '{df_name1}' and '{df_name2}':")
            print(f"Number of common ID pairs: {len(in_both)}")
            print(f"Number of ID pairs only in '{df_name1}': {len(only_in_df1)}")
            print(f"Number of ID pairs only in '{df_name2}': {len(only_in_df2)}")