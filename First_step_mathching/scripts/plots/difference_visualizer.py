import numpy as np
import pandas as pd

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

# VENN DIAGRAMS FUNCTIONS#
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt


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
    def plot_lat_lon_differences(dfs, loc_1, loc_2, name):
        """
        Plots the differences in latitude and longitude between two sets of coordinates using Seaborn,
        with KDE contours and a colorbar for the KDE.

        Parameters:
            dfs (pd.DataFrame): DataFrame containing the data.
            loc_1 (list): List of two column names for the first set of lat and lon. Format: [latitude, longitude].
            loc_2 (list): List of two column names for the second set of lat and lon. Format: [latitude, longitude].
            name (str): Name to be used in the title of the plot.

        Returns:
            None: Displays the scatter plot with KDE contours and colorbar.
        """
        df = dfs.copy()
        # Calculate the differences between the two sets of coordinates using .loc
        df['lat_diff'] = df[loc_1[0]] - df[loc_2[0]]
        df['lon_diff'] = df[loc_1[1]] - df[loc_2[1]]
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

    @staticmethod
    def match_venn2(df1, df2, id_column, df1_label='DF1', df2_label='DF2', model='MISSING'):
        """
        Matches two dataframes based on a specific id column and outputs a Venn diagram.
        
        Parameters:
        - df1: First dataframe
        - df2: Second dataframe
        - id_column: The name of the ID column to match on
        - df1_label: Label for the first dataframe in the Venn diagram
        - df2_label: Label for the second dataframe in the Venn diagram
        
        Returns:
        A Venn diagram showing the overlap of the two dataframes based on the id_column.
        """
        # Convert ID columns to sets for comparison
        set1 = set(df1[id_column])
        set2 = set(df2[id_column])
        
        # Create the Venn diagram
        plt.figure(figsize=(8, 8))
        venn2([set1, set2], set_labels=(df1_label, df2_label))
        
        # Show plot
        plt.title(f'Venn Diagram of {df1_label} and {df2_label} by {id_column} using {model.upper()}')
        plt.show()

    @staticmethod
    def match_venn3_matching(dfs: list[pd.DataFrame], model: str, id_columns: list[str], labels: list[str]):
        """
        Generalized function to create a Venn diagram from multiple dataframes based on specified ID columns.
        
        Parameters:
        - dfs: List of dataframes (assumed to be AIS-SAR, AIS-NORSAT, SAR-NORSAT in that order).
        - model: The key to access the relevant part of each dataframe (e.g., 'cham').
        - id_columns: List containing the IDs to use for merging (in the same order as the dataframes).
        Example: ['mmsi', 'sar_id', 'norsat_id']
        - labels: Labels for the datasets in the Venn diagram (e.g., ['AIS-SAR', 'AIS-NORSAT', 'SAR-NORSAT']).
        
        Returns:
        - Venn diagram showing the overlaps between the datasets based on the specified columns.
        """
        
        # Extract dataframes based on model
        ais_sar = dfs[0][model].copy()  # First dataframe (AIS-SAR)
        ais_norsat = dfs[1][model].copy()  # Second dataframe (AIS-NORSAT)
        sar_norsat = dfs[2][model].copy()  # Third dataframe (SAR-NORSAT)

        # Merge AIS-SAR with SAR-NORSAT based on the second ID (sar_id)
        ais_sar_sar_norsat_merge = ais_sar.merge(sar_norsat, on=[id_columns[1]])
        ais_sar_sar_norsat_count = len(ais_sar_sar_norsat_merge)

        # Merge AIS-SAR-SAR-NORSAT with AIS-NORSAT based on the first (mmsi) and third IDs (norsat_id)
        full_match_merge = ais_sar_sar_norsat_merge.merge(ais_norsat, on=[id_columns[0], id_columns[2]])
        full_match_count = len(full_match_merge)

        # Merge AIS-NORSAT with SAR-NORSAT based on the third ID (norsat_id)
        ais_norsat_sar_norsat_merge = ais_norsat.merge(sar_norsat, on=[id_columns[2]])
        ais_norsat_sar_norsat_count = len(ais_norsat_sar_norsat_merge)

        # Find unique counts for each set
        ais_count = len(ais_sar)  # AIS-SAR matches
        norsat_count = len(ais_norsat)  # AIS-NORSAT matches
        sar_count = len(sar_norsat)  # SAR-NORSAT matches

        # Calculate overlaps
        only_ais = ais_count - ais_sar_sar_norsat_count  # AIS-SAR only
        only_norsat = norsat_count - ais_norsat_sar_norsat_count  # AIS-NORSAT only
        only_sar = sar_count - ais_sar_sar_norsat_count  # SAR-NORSAT only
        ais_norsat_overlap = ais_norsat_sar_norsat_count - full_match_count  # AIS-NORSAT overlap
        ais_sar_overlap = ais_sar_sar_norsat_count - full_match_count  # AIS-SAR overlap
        sar_norsat_overlap = ais_norsat_sar_norsat_count - full_match_count  # SAR-NORSAT overlap
        full_overlap = full_match_count  # Full overlap (all three)

        # Create the Venn diagram
        plt.figure(figsize=(8, 8))
        venn3(subsets=(only_ais, only_norsat, ais_norsat_overlap,
                    only_sar, ais_sar_overlap, sar_norsat_overlap, full_overlap),
            set_labels=(labels[0], labels[1], labels[2]))

        # Show the Venn diagram
        plt.title(f'Venn Diagram of {labels[0]}, {labels[1]}, and {labels[2]} matches using {model.upper()}')
        plt.show()

    @staticmethod
    def match_venn3(outerLayer, dfs, model, id_columns, labels):
        """
        Creates a Venn diagram for AIS, SAR, and NORSAT data, showing intersections for pairwise and triple matches,
        ensuring that all circles are the same size while displaying the correct numbers.
        
        Parameters:
        - outerLayer (dict): Contains 'ais', 'sar', 'norsat' keys with 'df' and 'length' values.
        - dfs (list): List of DataFrames containing pairwise matches [AIS-SAR, AIS-NORSAT, SAR-NORSAT].
        - model (str): Column name representing the model in dfs.
        - id_columns (list): Column names to use for merging; expected as [mmsi_id, sar_id, norsat_id].
        - labels (list): List of names to display in the Venn diagram, e.g., ['AIS', 'SAR', 'NORSAT'].
        """
        # Calculate innerlayer from dfs and model
        innerlayer = {
            'ais_sar': {'df': dfs[0][model], 'length': len(dfs[0][model])},
            'ais_norsat': {'df': dfs[1][model], 'length': len(dfs[1][model])},
            'sar_norsat': {'df': dfs[2][model], 'length': len(dfs[2][model])}
        }
        
        # Extract dataframes for intersections
        ais_sar = innerlayer['ais_sar']['df']
        sar_norsat = innerlayer['sar_norsat']['df']
        ais_norsat = innerlayer['ais_norsat']['df']
        
        # Step 1: Merge AIS-SAR with SAR-NORSAT on SAR ID
        ais_sar_sar_norsat_merge = ais_sar.merge(sar_norsat, on=[id_columns[1]])
        ais_sar_sar_norsat_count = len(ais_sar_sar_norsat_merge)
        
        # Step 2: Merge AIS-SAR-SAR-NORSAT with AIS-NORSAT on MMSI and NORSAT IDs for triple matches
        full_match_merge = ais_sar_sar_norsat_merge.merge(ais_norsat, on=[id_columns[0], id_columns[2]])
        full_overlap = len(full_match_merge)  # Full overlap (all three)
        
        # Calculate counts for each area of the Venn diagram
        ais_count = outerLayer['ais']['length']
        sar_count = outerLayer['sar']['length']
        norsat_count = outerLayer['norsat']['length']
        ais_norsat_sar_norsat_count = len(ais_norsat)  # All AIS-NORSAT pairs

        # Calculate overlaps
        only_ais = ais_count - ais_sar_sar_norsat_count  # AIS only
        only_sar = sar_count - ais_sar_sar_norsat_count  # SAR only
        only_norsat = norsat_count - ais_norsat_sar_norsat_count  # NORSAT only

        ais_norsat_overlap = ais_norsat_sar_norsat_count - full_overlap  # AIS-NORSAT overlap
        ais_sar_overlap = ais_sar_sar_norsat_count - full_overlap        # AIS-SAR overlap
        sar_norsat_overlap = ais_norsat_sar_norsat_count - full_overlap  # SAR-NORSAT overlap

        # Set all non-overlap areas to an equal size for equal-sized circles
        equal_size = 200

        # Create the Venn diagram with equal-sized circles
        plt.figure(figsize=(8, 8))
        venn = venn3(
            subsets=(equal_size, equal_size, ais_sar_overlap,
                    equal_size, ais_norsat_overlap, sar_norsat_overlap, full_overlap),
            set_labels=(labels[0], labels[1], labels[2])
        )
        font_size = 12
        # Manually set labels to reflect actual counts
        venn.get_label_by_id('100').set_text(only_ais)
        venn.get_label_by_id('100').set_fontsize(font_size)

        venn.get_label_by_id('010').set_text(only_sar)
        venn.get_label_by_id('010').set_fontsize(font_size)

        venn.get_label_by_id('110').set_text(ais_sar_overlap)
        venn.get_label_by_id('110').set_fontsize(font_size)

        venn.get_label_by_id('001').set_text(only_norsat)
        venn.get_label_by_id('001').set_fontsize(font_size)

        venn.get_label_by_id('101').set_text(ais_norsat_overlap)
        venn.get_label_by_id('101').set_fontsize(font_size)

        venn.get_label_by_id('011').set_text(sar_norsat_overlap)
        venn.get_label_by_id('011').set_fontsize(font_size)

        venn.get_label_by_id('111').set_text(full_overlap)
        venn.get_label_by_id('111').set_fontsize(font_size)
        # Set title
        plt.title(f'Venn Diagram of {labels[0]}, {labels[1]}, and {labels[2]} matches using {model.upper()}')

        # Show the plot
        plt.show()
