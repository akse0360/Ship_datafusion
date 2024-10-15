import pandas as pd
import numpy as np


# Matched data Filters class
class MatchedFilters:
    
    @staticmethod
    def filter_by_distance(dict_dfs : dict, distance_threshold : float =15, printer : bool = False) -> dict:
        """
        Filters DataFrames in the given dictionary by a specified distance threshold.

        Parameters:
            dict_dfs (dict): Dictionary where keys are identifiers and values are DataFrames.
                        Each DataFrame may contain a 'distance_km' column.
            distance_threshold (float): The distance threshold (in km) to filter the DataFrames. Default is 15 km.
            printer (bool): Whether to print the number of matches before and after filtering. Default is False.
            
        Returns:
            thresholded_dict (dict): A dictionary containing DataFrames filtered by the given distance threshold.
        """
        thresholded_dict = {}
        
        for key, df in dict_dfs.items():
            if 'distance_km' in df.columns:
                
                thresholded_dict[key] = df[df['distance_km'] <= distance_threshold]
                if printer:
                    print(f'Number of matches {key} : {len(df)}')
                    print(f'Number of matches between {key} within {distance_threshold} km: {len(thresholded_dict[key])}')
            else:
                print(f'Warning: DataFrame for {key} does not have a "distance_km" column.')

        return thresholded_dict