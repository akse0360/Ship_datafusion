# libaries for data manipulation
import pandas as pd

# libaries for interpolation
import numpy as np
from scipy.interpolate import interp1d, CubicSpline


# Script contains two classes: AISFilters and AISInterpolation


# AIS Filters class
class AISFilters:
    @staticmethod
    def ais_find_matching_vessels(AIS_data, comparison_data, delta_time):
        """
        Find vessels in AIS_data whose timestamps are close to the timestamps in comparison_data within a given delta_time.

        Parameters:
        - AIS_data: DataFrame containing AIS data with columns 'ais_id', 'mmsi', 'TimeStamp', 'latitude_x', 'longitude_x',
                    'length', 'width_x', 'sog', 'cog', 'source_x'.
        - comparison_data: DataFrame containing comparison data with 'TimeStamp'.
        - delta_time: The pd.Timedelta object representing the time threshold for comparison.

        Returns:
        - grouped_df: DataFrame grouped by 'mmsi' with only the groups where the threshold is met.
                    The columns are ['ais_id', 'mmsi', 'TimeStamp', 'latitude_x', 'longitude_x', 'length',
                    'width_x', 'sog', 'cog', 'source_x', 'comparison_timestamp'].
        """

        # Ensure both 'TimeStamp' columns are datetime type
        AIS_data['TimeStamp'] = pd.to_datetime(AIS_data['TimeStamp'])
        comparison_data['TimeStamp'] = pd.to_datetime(comparison_data['TimeStamp'])

        # Sort both dataframes by 'TimeStamp' to prepare for merge_asof
        AIS_data = AIS_data.sort_values(by='TimeStamp')
        comparison_data = comparison_data.sort_values(by='TimeStamp')

        # Rename the 'TimeStamp' column in comparison_data to avoid conflicts during merge
        comparison_data = comparison_data.rename(columns={'TimeStamp': 'comparison_timestamp'})

        # Use merge_asof to find the closest timestamps within the delta_time
        merged_df = pd.merge_asof(
            AIS_data, 
            comparison_data, 
            left_on='TimeStamp', 
            right_on='comparison_timestamp', 
            tolerance=delta_time, 
            direction='nearest'
        )

        # Drop rows where no match was found (merge_asof fills NaN for unmatched rows)
        merged_df = merged_df.dropna(subset=['comparison_timestamp'])

        # Calculate the time difference and add it as a new column
        merged_df['time_difference'] = abs(merged_df['TimeStamp'] - merged_df['comparison_timestamp'])

        # Select only the relevant columns for the final output
        selected_columns = ['ais_id', 'mmsi', 'TimeStamp', 'latitude_x', 'longitude_x', 'length',
                            'width_x', 'sog', 'cog', 'source_x', 'comparison_timestamp']
        
        # Ensure that all necessary columns exist in the merged DataFrame
        merged_df = merged_df[[col for col in selected_columns if col in merged_df.columns]]

        # Group by 'mmsi' and return the grouped dataframe
        return merged_df.groupby(by='mmsi')
    
    @staticmethod
    def filter_by_mmsi(grouped_df, target_df):
        """
        Filters the target dataframe to include only rows with MMSI numbers present in the grouped_df.

        Args:
            grouped_df (pd.DataFrameGroupBy): The grouped dataframe from ais_find_matching_vessels.
            target_df (pd.DataFrame): The dataframe to be filtered.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        # Extract the list of MMSI numbers from grouped_df
        mmsi_list = grouped_df.groups.keys()

        # Filter the target dataframe based on the MMSI numbers
        return target_df[target_df['mmsi'].isin(mmsi_list)]


# AIS Interpolation class
class AISInterpolation:
    # Define the interpolation function
    @staticmethod
    def ais_interpolate_mmsi_points(ais_data, interpolation_time_col):
        """
        Interpolates MMSI points from the given DataFrame in ais_data.

        Parameters:
        - ais_data: A DataFrame containing AIS data with the following columns:
            ['mmsi', 'TimeStamp', 'latitude', 'longitude', 'length', 'width', 'sog', 'cog']
        - interpolation_time_col: The time to interpolate to (should be a pandas Timestamp).

        Returns:
        - A DataFrame with the interpolated latitude and longitude along with data before and after the interpolation time.
        """
        # Group the AIS data by MMSI number
        ais_mmsi = ais_data.groupby(by=['mmsi'])
        mmsi_numbers = list(ais_mmsi.indices.keys())
        not_enough = []  # List to store MMSI numbers with insufficient data points
        interpolated_data = []  # List to store the interpolated results

        # Convert interpolation time to numeric for interpolation purposes
        interpolation_time_numeric = pd.to_numeric(interpolation_time_col).mean()

        for mmsi in mmsi_numbers:
            mmsi_group = ais_mmsi.get_group((mmsi,))

            # Check for NaN values and drop them
            if mmsi_group[['TimeStamp', 'longitude', 'latitude']].isnull().any().any():
                continue  # Skip this group if there are NaN values

            # Ensure there are at least two points for interpolation
            if len(mmsi_group) < 2:
                not_enough.append(mmsi)
                continue

            # Sort the group by timestamp
            mmsi_group = mmsi_group.sort_values(by='TimeStamp')

            # Convert TimeStamp to numeric values for interpolation
            time_numeric = pd.to_numeric(mmsi_group['TimeStamp'])

            # Ensure that interpolation_time is within the bounds of time_numeric to avoid extrapolation
            if interpolation_time_numeric < time_numeric.min() or interpolation_time_numeric > time_numeric.max():
                not_enough.append(mmsi)
                continue

            # Find the two closest points around the interpolation time
            before_idx = time_numeric[time_numeric <= interpolation_time_numeric].idxmax()
            after_idx = time_numeric[time_numeric >= interpolation_time_numeric].idxmin()

            # Extract before and after information
            before_point = mmsi_group.loc[before_idx]
            after_point = mmsi_group.loc[after_idx]

            # Prepare the time and coordinate data for interpolation
            longitude = mmsi_group['longitude'].values
            latitude = mmsi_group['latitude'].values

            # Choose linear interpolation
            if len(mmsi_group) == 2:
                interpol = interp1d(time_numeric, np.c_[longitude, latitude], axis=0, kind='linear')
            # Choose cubic spline interpolation
            elif len(mmsi_group) > 2:
                interpol = CubicSpline(time_numeric, np.c_[longitude, latitude], axis=0)
            else:
                continue
            
            # Perform interpolation for the specified interpolation time
            x_interp, y_interp = interpol(interpolation_time_numeric).T

            # Create a dictionary with the interpolated data and before/after information
            interpolated_entry = {
                'int_TimeStamp': pd.to_datetime(interpolation_time_numeric, unit='ns').tz_localize('UTC'),
                'mmsi': mmsi,
                'int_latitude': y_interp,
                'int_longitude': x_interp,
                'TimeStamp_before': before_point['TimeStamp'],
                'latitude_before': before_point['latitude'],
                'longitude_before': before_point['longitude'],
                'length_before': before_point['length'],
                'width_before': before_point['width'],
                'sog_before': before_point['sog'],
                'cog_before': before_point['cog'],
                'TimeStamp_after': after_point['TimeStamp'],
                'latitude_after': after_point['latitude'],
                'longitude_after': after_point['longitude'],
                'length_after': after_point['length'],
                'width_after': after_point['width'],
                'sog_after': after_point['sog'],
                'cog_after': after_point['cog']
            }

            interpolated_data.append(interpolated_entry)

        # Convert the interpolated data list to a DataFrame
        interpolated_df = pd.DataFrame(interpolated_data)

        # After processing, you can check the not_enough list
        print(f"MMSI numbers with insufficient data points: {len(not_enough)} out of {len(mmsi_numbers)}")

        return interpolated_df, not_enough
    
    @staticmethod
    def evaluate_sog_between_points(interpolated_df, max_threshold_sog):
        """
        Evaluates the speed over the ground between the interpolated point and the 'before' and 'after' positions.
        
        Parameters:
        - interpolated_df: DataFrame containing interpolated points and their 'before' and 'after' positions.
        - max_threshold_sog: Maximum allowable speed over the ground (in knots or km/h) as a threshold.
        
        Returns:
        - A DataFrame with the following columns:
        ['mmsi', 'int_latitude', 'int_longitude', 'TimeStamp_before', 'TimeStamp_after', 
        'latitude_before', 'longitude_before', 'latitude_after', 'longitude_after', 
        'sog_before_to_interp', 'sog_after_to_interp', 'exceeds_threshold_before', 'exceeds_threshold_after'].
        """
        # Haversine distance function as defined by you
        def haversine_distance(lat1, lon1, lat2, lon2):
            """
            Calculate the Haversine distance between two points in vectorized form using numpy.
            
            Args:
                lat1, lon1, lat2, lon2: Arrays or Series representing latitude and longitude.
                
            Returns:
                Series or array: Haversine distance in kilometers.
            """
            R = 6371.0  # Radius of the Earth in kilometers

            # Convert latitude and longitude from degrees to radians
            lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)

            # Compute differences
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            # Haversine formula
            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            # Distance in kilometers
            return R * c
        
        # Calculate the distance between interpolated point and before/after positions using Haversine distance
        interpolated_df['distance_before_interp'] = haversine_distance(
            interpolated_df['int_latitude'], interpolated_df['int_longitude'],
            interpolated_df['latitude_before'], interpolated_df['longitude_before']
        )
        
        interpolated_df['distance_after_interp'] = haversine_distance(
            interpolated_df['int_latitude'], interpolated_df['int_longitude'],
            interpolated_df['latitude_after'], interpolated_df['longitude_after']
        )

        # Calculate the time differences in hours
        interpolated_df['time_before_interp'] = (interpolated_df['TimeStamp_before'] - interpolated_df['int_TimeStamp']).dt.total_seconds() / 3600
        interpolated_df['time_after_interp'] = (interpolated_df['TimeStamp_after'] - interpolated_df['int_TimeStamp']).dt.total_seconds() / 3600

        # Calculate speed over ground (SOG) as distance/time
        interpolated_df['sog_before_to_interp'] = interpolated_df['distance_before_interp'] / interpolated_df['time_before_interp'].abs()
        interpolated_df['sog_after_to_interp'] = interpolated_df['distance_after_interp'] / interpolated_df['time_after_interp'].abs()

        # Replace any infinite values (which can occur if time_before_interp or time_after_interp is zero) with NaN
        interpolated_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check if the calculated SOG exceeds the threshold
        interpolated_df['exceeds_threshold_before'] = interpolated_df['sog_before_to_interp'] > max_threshold_sog
        interpolated_df['exceeds_threshold_after'] = interpolated_df['sog_after_to_interp'] > max_threshold_sog

        # Return the DataFrame with additional calculated columns
        return interpolated_df


    def der():
        print('Hello AIS Interpolation')