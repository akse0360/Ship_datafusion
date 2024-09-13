# Python Lib
import os

# Extended Libs
import numpy as np
import pandas as pd
import json

# Plotting Libs
import seaborn as sns
import matplotlib.pyplot as plt

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

