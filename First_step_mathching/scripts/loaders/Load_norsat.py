import pandas as pd
import numpy as np
import os

class Load_norsat:
    """
    Loads and formats Norsat data files for further processing.

    This class is responsible for reading JSON files specified in a dictionary, renaming 
    the TimeStamp column to time, adding a source identifier, and applying additional 
    formatting to each DataFrame. It also provides a method to extract latitude and 
    longitude from the NRDEmitterPosition column in the DataFrames.

    Args:
        base_path (str): The base path where the Norsat files are located.
        norsat_files (dict): A dictionary mapping dates to corresponding Norsat file names.

    Attributes:
        dfs_norsat (dict): A dictionary containing loaded Norsat DataFrames with dates as keys.
    """

    def __init__(self, base_path: str, norsat_files: dict) -> None:
        """
        Initializes a new instance of the class by loading and formatting Norsat data files.

        This constructor reads JSON files specified in the norsat_files dictionary, renames the 
        TimeStamp column to time, adds a source identifier, and applies additional formatting 
        to each DataFrame. It also prints the keys and columns of the loaded Norsat data for 
        verification.

        Args:
            base_path (str): The base path where the Norsat files are located.
            norsat_files (dict): A dictionary mapping dates to corresponding Norsat file names.

        Returns:
            None
        """

        # Load the Norsat data into DataFrames
        self.dfs_norsat = {date: pd.read_json(os.path.join(base_path, file)) for date, file in norsat_files.items()}

        # Process and format each loaded DataFrame
        for date, df in self.dfs_norsat.items():
            # Convert TimeStamp column to datetime and add source column
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], utc=True)
            df['source'] = 'norsat'

            # Format the DataFrame to extract latitude and longitude
            self.dfs_norsat[date] = self.norsat_formatting(df)

            # Reset index and create 'norsat_id' column
            self.dfs_norsat[date].reset_index(drop=False, inplace=True)
            self.dfs_norsat[date].rename(columns={'index': 'norsat_id'}, inplace=True)

            # Add the uncertainty ellipse points
            self.dfs_norsat[date] = self.add_uncertainty_ellipse_points(self.dfs_norsat, date_key=date)

            # Reorder the columns to the desired order
            self.dfs_norsat[date] = self.dfs_norsat[date].reindex(columns=[
                'norsat_id', 'TimeStamp', 'latitude', 'longitude', 
                'CollectionInformation', 'NRDEmitterPosition', 'CandidateList', 
                'source', 'UncertaintyEllipsePoints'
            ], fill_value=None)

        print(f"Norsat Data Loaded:\n{self.dfs_norsat.keys()}")
        print(f"Columns for the first DataFrame: {self.dfs_norsat[list(self.dfs_norsat.keys())[0]].columns}")

    def norsat_formatting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Formats a DataFrame by extracting latitude and longitude from the NRDEmitterPosition column.

        This function processes the input DataFrame to create new columns for latitude and longitude 
        by extracting these values from the NRDEmitterPosition column, which is expected to contain 
        dictionaries. If the NRDEmitterPosition is not a dictionary, the corresponding latitude or 
        longitude will be set to None.

        Args:
            df (pd.DataFrame): The input DataFrame containing the NRDEmitterPosition column.

        Returns:
            pd.DataFrame: The modified DataFrame with added 'latitude' and 'longitude' columns.
        """
        df['latitude'] = df['NRDEmitterPosition'].apply(lambda x: x.get('Latitude') if isinstance(x, dict) else None)
        df['longitude'] = df['NRDEmitterPosition'].apply(lambda x: x.get('Longitude') if isinstance(x, dict) else None)
        return df

    def add_uncertainty_ellipse_points(self, df: dict, date_key: str) -> pd.DataFrame:
        """
        Adds a new column 'UncertaintyEllipsePoints' to the Norsat DataFrame for a specific date.

        This method computes the points representing the uncertainty ellipse for each row in the 
        DataFrame. The points are calculated based on the parameters found in the 
        'NRDEmitterPosition' column.

        Args:
            df (dict): Dictionary of Norsat DataFrames with dates as keys.
            date_key (str): The specific date key for the DataFrame to process.

        Returns:
            pd.DataFrame: The modified DataFrame with a new column 'UncertaintyEllipsePoints'.
        """
        # Function to generate the coordinates of an ellipse
        def generate_ellipse_points(center_lat, center_lon, major_axis, minor_axis, angle, num_points=100):
            def meters_to_degrees_lat(meters):
                return meters / 111320

            def meters_to_degrees_lon(meters, latitude):
                return meters / (111320 * np.cos(np.radians(latitude)))
            
            major_axis_lat = meters_to_degrees_lat(major_axis)
            minor_axis_lon = meters_to_degrees_lon(minor_axis, center_lat)
            
            theta = np.linspace(0, 2 * np.pi, num_points)
            ellipse_lat = major_axis_lat * np.cos(theta)
            ellipse_lon = minor_axis_lon * np.sin(theta)
            
            angle_rad = np.radians(angle)
            lat_rot = ellipse_lat * np.cos(angle_rad) - ellipse_lon * np.sin(angle_rad)
            lon_rot = ellipse_lat * np.sin(angle_rad) + ellipse_lon * np.cos(angle_rad)
            
            lat_points = center_lat + lat_rot
            lon_points = center_lon + lon_rot
            
            return list(zip(lat_points, lon_points))

        def compute_ellipse_points(row):
            emitter_position = row['NRDEmitterPosition']
            if isinstance(emitter_position, dict):
                latitude = emitter_position.get('Latitude', None)
                longitude = emitter_position.get('Longitude', None)
                ellipse = emitter_position.get('UncertaintyEllipse', {})

                if all(key in ellipse for key in ['MajorAxis', 'MinorAxis', 'AngleRelativeNorth']):
                    major_axis = ellipse['MajorAxis']
                    minor_axis = ellipse['MinorAxis']
                    angle = ellipse['AngleRelativeNorth']

                    ellipse_points = generate_ellipse_points(latitude, longitude, major_axis, minor_axis, angle)
                    return ellipse_points
            return None
        
        # Apply the function to each row in the DataFrame for the specified date_key
        df[date_key]['UncertaintyEllipsePoints'] = df[date_key].apply(compute_ellipse_points, axis=1)
        return df[date_key]