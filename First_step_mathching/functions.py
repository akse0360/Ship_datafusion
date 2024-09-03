# Python Lib
import os

# Extended Libs
import numpy as np
import pandas as pd
import json

# Plotting Libs
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Choose file:
import tkinter as tk


# Importing required library
from mpl_interactions import ioff, panhandler, zoom_factory
import matplotlib.pyplot as plt


class norsat_data_processing():
    def norsat_formatting(norsat_df : pd.DataFrame) -> pd.DataFrame: 
        # Format NRDEmitterPosition
        norsat_df['latitude'] = norsat_df['NRDEmitterPosition'].apply(lambda x: x['Latitude'])
        norsat_df['longitude'] = norsat_df['NRDEmitterPosition'].apply(lambda x: x['Longitude'])

        # Drop the NRDEmitterPosition column, if no longer need it:
        #norsat_df = norsat_df.drop(columns=['NRDEmitterPosition'])
        
        return norsat_df
    
    def plot_lat_lon_map(dfs_dict : dict) -> None:
        """
        Plots latitude and longitude from multiple DataFrames on the same map.

        Parameters:
        dfs_dict (dict): A dictionary where keys are labels (e.g., dates) and values are DataFrames containing 'Latitude' and 'Longitude'.

        """
        # Initialize the map
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Determine the overall extent based on all dataframes
        all_lats = []
        all_lons = []
        for df in dfs_dict.values():
            all_lats.extend(df['Latitude'])
            all_lons.extend(df['Longitude'])

        # Set the extent (slightly extended) for the map
        ax.set_extent([min(all_lons)-5, max(all_lons)+5, min(all_lats)-5, max(all_lats)+5])

        # Add features like coastlines and borders
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot each DataFrame's points with different markers or colors
        for label, df in dfs_dict.items():
            ax.scatter(df['Longitude'], df['Latitude'], label=label, s=10, transform=ccrs.PlateCarree())

        # Add gridlines and labels
        gl = ax.gridlines(draw_labels = True, crs = ccrs.PlateCarree(), linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12, 'color': 'black'}
        gl.ylabel_style = {'size': 12, 'color': 'black'}

        # Add a legend to differentiate between different DataFrames
        plt.legend(title = "DataFrames", loc = 'upper right')

        # Add a title
        plt.title('Mapping of Norsat data')
        plt.ion()
        # Show the plot
        plt.show()
    
class sar_data_processing():
    def expand_objects_for_date(dfs_sar : dict, date_key : str) -> pd.DataFrame:
        """
        Expands the objects from a specific date in the dfs_sar dictionary into a DataFrame.

        Parameters:
        - dfs_sar: Dictionary where keys are dates and values are DataFrames containing objects.
        - date_key: The specific date (key) in the format 'DD-MM-YYYY' for which to expand the objects.

        Returns:
        - A DataFrame with the expanded objects for the specified date.
        """
        if date_key not in dfs_sar:
            raise ValueError(f"Date {date_key} not found in the dictionary.")
        
        df = dfs_sar[date_key]
        expanded_data = []

        for _, row in df.iterrows():
            start_time = row['Start']
            start_time = row['End']
            objects = row['Objects']
            
            # Convert the JSON objects to a dict
            if isinstance(objects, str):  # In case JSON is stored as a string
                objects = json.loads(objects)
            
            # Expand each object in the 'Objects' field
            for obj_id, obj_data in objects.items():
                expanded_row = {
                    'Start': start_time,
                    'Object_ID': obj_id,
                    'x': obj_data['x'],
                    'y': obj_data['y'],
                    'width': obj_data['width'],
                    'height': obj_data['height'],
                    'class': obj_data['class'],
                    'latitude': obj_data['latitude'],
                    'longitude': obj_data['longitude'],
                    'probabilities': obj_data['probabilities'],
                    'encoded_image': obj_data['encoded_image']
                }
                expanded_data.append(expanded_row)

        # Convert the expanded data into a DataFrame
        expanded_df = pd.DataFrame(expanded_data)
        
        return expanded_df

class ais_data_processing():
    def ais_sns_plots(df_ais : pd.DataFrame) -> None: 
        # Assuming your DataFrame is named df and 'bs_ts' is already converted to datetime
        df_ais['hour_minute'] = df_ais['bs_ts_time'].dt.floor('15T')  # Rounding down to the nearest 15 minutes

        ## Plotting the histogram with bins for each 15-minute interval ##
        plt.figure(figsize = (10, 6))
        sns.histplot(df_ais['hour_minute'], bins = 16, kde=True)
        plt.title('Distribution of 15-Minute Intervals Between 14:00 and 18:00')
        plt.xlabel('Time Interval')
        plt.ylabel('Frequency')
        plt.xticks(rotation = 45)
        plt.show()

        ## Pair plot ##
        sns.pairplot(df_ais[['lat', 'lon']], diag_kind = 'kde', kind = 'hist')
        sns.set_theme(style = "ticks")
        plt.show()