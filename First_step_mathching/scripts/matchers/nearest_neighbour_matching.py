# Nearest neighbour matching #
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def match_nearest_neighbour_unique(df1, df2, id1_col, id2_col):
    """
    Match each point in df1 to the nearest unique point in df2 based on latitude and longitude.
    
    Args:
        df1 (pd.DataFrame): DataFrame containing the first set of points with 'latitude' and 'longitude' columns.
        df2 (pd.DataFrame): DataFrame containing the second set of points with 'latitude' and 'longitude' columns.
        id1_col (str): Column name representing the ID of points in df1.
        id2_col (str): Column name representing the ID of points in df2.
        
    Returns:
        pd.DataFrame: DataFrame containing the matched points and their distances, ensuring unique matches.
    """
    # Create copies to avoid modifying the original DataFrames
    df1 = df1.copy()
    df2 = df2.copy()

    # Convert lat/lon to radians for computational purposes
    df1['lat_rad'] = np.radians(df1['latitude'])
    df1['lon_rad'] = np.radians(df1['longitude'])
    df2['lat_rad'] = np.radians(df2['latitude'])
    df2['lon_rad'] = np.radians(df2['longitude'])
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
    # Convert latitude and longitude into Cartesian coordinates for each point
    def lat_lon_to_cartesian(lat, lon):
        R = 6371.0  # Radius of Earth in kilometers
        x = R * np.cos(lat) * np.cos(lon)
        y = R * np.cos(lat) * np.sin(lon)
        z = R * np.sin(lat)
        return np.vstack([x, y, z]).T

    df1_cartesian = lat_lon_to_cartesian(df1['lat_rad'].values, df1['lon_rad'].values)
    df2_cartesian = lat_lon_to_cartesian(df2['lat_rad'].values, df2['lon_rad'].values)

    # Build KDTree for the second dataframe
    tree = cKDTree(df2_cartesian)

    # Find the nearest neighbors
    distances, indices = tree.query(df1_cartesian)

    # Create the initial result DataFrame
    matches_df = pd.DataFrame({
        'df1_id': df1[id1_col].values,
        'df1_lat': df1['latitude'].values,
        'df1_lon': df1['longitude'].values,
        'df2_id': df2.iloc[indices][id2_col].values,
        'df2_lat': df2.iloc[indices]['latitude'].values,
        'df2_lon': df2.iloc[indices]['longitude'].values,
        'haversine_distance': haversine_distance(
            df1['latitude'].values, df1['longitude'].values, 
            df2.iloc[indices]['latitude'].values, df2.iloc[indices]['longitude'].values
        ),
        'df2_index': indices  # Keep track of the indices of df2 used for matching
    })

    # Remove duplicate df2 matches to ensure unique matching
    matches_df = matches_df.sort_values('haversine_distance').drop_duplicates(subset=['df2_index'], keep='first')
    print(len(matches_df))
    # Remove rows where df1_id has multiple entries (df1 should have unique matches)
    matches_df = matches_df.drop_duplicates(subset=['df1_id'], keep='first')
    print(len(matches_df))
    # Drop the helper 'df2_index' column
    matches_df = matches_df.drop(columns='df2_index')
    print(len(matches_df))
    return matches_df