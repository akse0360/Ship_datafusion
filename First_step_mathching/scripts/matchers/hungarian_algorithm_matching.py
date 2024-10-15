import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianAlgorithmMatcher:
    @staticmethod
    def haversine_vec(lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points in kilometers.
        """
        # Convert latitude and longitude to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Calculate the difference between the two coordinates
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # Apply the haversine formula
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Radius of Earth in kilometers
        r = 6371

        # Return the distance in km
        return c * r

    @staticmethod
    def create_cost_matrix_vectorized(df1, df2):
        """
        Create a cost matrix using the Haversine distance between points in df1 and df2.
        """
        # Extract the coordinates of the ships and SAR detections from DataFrames
        ship_coords = df1[['latitude', 'longitude']].to_numpy()
        sar_coords = df2[['latitude', 'longitude']].to_numpy()

        # Vectorized haversine distance calculation for each pair of ships and SAR detections
        ship_lats, ship_lons = ship_coords[:, 0], ship_coords[:, 1]
        sar_lats, sar_lons = sar_coords[:, 0], sar_coords[:, 1]

        # Broadcasting the ship and SAR coordinates to calculate all distances at once
        ship_lats_b, sar_lats_b = np.broadcast_arrays(ship_lats[:, None], sar_lats[None, :])
        ship_lons_b, sar_lons_b = np.broadcast_arrays(ship_lons[:, None], sar_lons[None, :])

        return HungarianAlgorithmMatcher.haversine_vec(ship_lats_b, ship_lons_b, sar_lats_b, sar_lons_b)

    @staticmethod
    def hungarian_method_matching(df1: pd.DataFrame, df2: pd.DataFrame, id1_col: str, id2_col: str) -> pd.DataFrame:
        """
        Matches ships from df1 to df2 using the Hungarian algorithm.

        Parameters:
        - df1: DataFrame containing data with columns [id1_col, 'latitude', 'longitude'].
        - df2: DataFrame containing data with columns [id2_col, 'latitude', 'longitude'].

        Returns:
        - A DataFrame with matched results in the format:
          [id1_col, 'df1_lat', 'df1_lon', id2_col, 'df2_lat', 'df2_lon', 'distance_km']
        """
        # Create copies to avoid modifying the original DataFrames
        df1 = df1.copy()
        df2 = df2.copy()
        
        # Ensure the DataFrames contain 'latitude' and 'longitude' columns        
        if not {'latitude', 'longitude'}.issubset(df1.columns) or not {'latitude', 'longitude'}.issubset(df2.columns):
            if {'int_latitude', 'int_longitude'}.issubset(df1.columns):
                df1.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            elif {'int_latitude', 'int_longitude'}.issubset(df2.columns):
                df2.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            else:
                raise ValueError("Input DataFrames must contain 'latitude' and 'longitude' columns.")

        # Create the cost matrix using Haversine distances
        cost_matrix = HungarianAlgorithmMatcher.create_cost_matrix_vectorized(df1, df2)

        # Extract the ID columns and get the ship index values for df1
        df1_idx = df1[id1_col].values
        df2_idx = df2[id2_col].values

        # Apply the Hungarian algorithm to minimize the total cost (distance)
        ship_indices, sar_indices = linear_sum_assignment(cost_matrix)

        # Create a list of matched results
        matches = []
        for ship_idx, sar_idx in zip(ship_indices, sar_indices):
            # Get the original IDs and coordinates
            ship_id = df1_idx[ship_idx]
            sar_id = df2_idx[sar_idx]
            ship_lat, ship_lon = df1.iloc[ship_idx][['latitude', 'longitude']]
            sar_lat, sar_lon = df2.iloc[sar_idx][['latitude', 'longitude']]
            distance_km = cost_matrix[ship_idx, sar_idx]

            # Create a match record
            match = {
                id1_col: ship_id.astype(int),
                'df1_lat': ship_lat,
                'df1_lon': ship_lon,
                id2_col: sar_id.astype(int),
                'df2_lat': sar_lat,
                'df2_lon': sar_lon,
                'distance_km': distance_km
            }
            matches.append(match)

        return pd.DataFrame(matches), cost_matrix