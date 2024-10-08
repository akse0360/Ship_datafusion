import pandas as pd
import os
import geopandas as gpd
from roaring_landmask import RoaringLandmask

from scripts.functions import functions as func

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import numpy as np

class Load_sar:
    """
    Loads and formats SAR data files for further analysis.

    This class is responsible for reading JSON files specified in a dictionary, renaming 
    columns for consistency, adding a source identifier, and converting the time columns 
    to datetime format. It also standardizes the DataFrame columns to the expected format.

    Args:
        base_path (str): The base path where the SAR files are located.
        sar_files (dict): A dictionary mapping dates to corresponding SAR file names.

    Attributes:
        dfs_sar (dict): A dictionary containing loaded SAR DataFrames with dates as keys.
        sar_object_dfs (dict): A dictionary containing DataFrames of individual object details extracted from the SAR DataFrames.
    """
    # Expected columns for the final DataFrame
    expected_columns = [
        'ProductType', 'Polarization', 'Swath', 'TimeStamp', 
        'TimeStamp_end', 'Name', 'Satellite', 'Shape', 'Objects', 'source'
    ]

    def __init__(self, base_path: str, sar_files: dict) -> None:
        """
        Initializes a new instance of the LoadSAR class by loading and formatting SAR data files,
        and extracts SAR objects into separate DataFrames.

        Args:
            base_path (str): The base path where the SAR files are located.
            sar_files (dict): A dictionary mapping dates to corresponding SAR file names.

        Returns:
            None
        """
        # Load and process the SAR files into a dictionary of DataFrames
        self.dfs_sar = {
            date: self._process_file(os.path.join(base_path, file)) for date, file in sar_files.items()
        }

        print(f"SAR Data Loaded:\n{self.dfs_sar.keys()}")
        print(f"Columns for the first DataFrame: {self.dfs_sar[list(self.dfs_sar.keys())[0]].columns}")

        # Extract SAR objects and store them as a dictionary of DataFrames
        self.sar_object_dfs = self.extract_sar_objects()
        
        self.add_on_sea_column()

        # NOTE: CHANGE SHAPEFILE E.G.: RESOLUTION 
        shoreline_gdf = gpd.read_file("shpfile/GSHHS_l_L1.shp")
        for date in self.sar_object_dfs.keys():
            self.compute_nearest_shoreline_haversine(df=self.sar_object_dfs[date], shoreline_gdf = shoreline_gdf, buffer=0.5)
                
        print(f"SAR object Loaded:\n{self.sar_object_dfs.keys()}")
        print(f"Columns for the first DataFrame: {self.sar_object_dfs[list(self.sar_object_dfs.keys())[0]].columns}")


    def _process_file(self, file_path: str) -> pd.DataFrame:
        """
        Helper function to load and process individual SAR files.

        Args:
            file_path (str): Path to the SAR file to load.

        Returns:
            pd.DataFrame: Processed SAR DataFrame with the expected format.
        """
        # Load JSON file into a DataFrame
        df = pd.read_json(file_path, orient='index')

        # Rename columns and add source
        df = df.rename(columns={'Start': 'TimeStamp'})
        df = df.assign(
            source='sar',
            TimeStamp=pd.to_datetime(df['TimeStamp'], utc=True) if 'TimeStamp' in df.columns else pd.NaT,
            TimeStamp_end=pd.to_datetime(df['End'], utc=True) if 'End' in df.columns else pd.NaT
        )

        # Ensure all expected columns are present with NaN values if missing
        missing_cols = set(self.expected_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = pd.NA

        # Reorder columns to match expected format
        df = df[self.expected_columns]

        return df

    def extract_sar_objects(self) -> dict:
        """
        Extracts individual object details from the 'Objects' column of each SAR DataFrame.

        This method creates a new dictionary, `sar_object_dfs`, where each key is a date, and each
        value is a DataFrame containing extracted object details with the following columns in order:
        ['sar_object_id', 'sar_image_id', 'TimeStamp', 'latitude', 'longitude', 'width', 'height', 
        'probabilities', 'source', 'encoded_image'].

        Returns:
            dict: Dictionary of DataFrames with extracted SAR object details.
        """
        # Dictionary to store the new DataFrames
        sar_object_dfs = {}

        # Define the desired column order
        desired_columns = ['sar_id', 'sar_image_id', 'TimeStamp', 'latitude', 'longitude', 'width', 
                        'height', 'probabilities', 'source', 'encoded_image']

        # Iterate through each date and corresponding DataFrame in dfs_sar
        for date, df in self.dfs_sar.items():
            # Check if 'Objects' column exists and is not empty
            if 'Objects' in df.columns and not df['Objects'].isnull().all():
                # Initialize a list to hold individual object data
                object_list = []

                # Iterate through each row in the DataFrame
                for sar_image_id, row in df.iterrows():  # Use `sar_image_id` as the index
                    # Get the TimeStamp of the current row
                    timestamp = row['TimeStamp']

                    # Extract the 'Objects' field, which should be a dictionary of object entries
                    objects = row['Objects']

                    # Iterate through each object in the 'Objects' column
                    if isinstance(objects, dict):
                        for object_key, object_value in objects.items():
                            if isinstance(object_value, dict):
                                # Create a new row for each object, including the TimeStamp from the SAR row
                                object_data = {
                                    'sar_image_id': sar_image_id,  # Assign sar_image_id based on the index of the original DataFrame
                                    'TimeStamp': timestamp,
                                    'latitude': object_value.get('latitude', None),
                                    'longitude': object_value.get('longitude', None),
                                    'width': object_value.get('width', None),
                                    'height': object_value.get('height', None),
                                    'probabilities': object_value.get('probabilities', None),
                                    'source': 'SAR',
                                    'encoded_image': object_value.get('encoded_image', None),
                                }

                                # Append to the object list
                                object_list.append(object_data)

                # Create a DataFrame from the object list for the current date
                df_objects = pd.DataFrame(object_list)

                # Add the 'sar_object_id' column by resetting the index
                df_objects.reset_index(drop=False, inplace=True)
                df_objects.rename(columns={'index': 'sar_id'}, inplace=True)

                # Reorder the columns according to the desired order
                sar_object_dfs[date] = df_objects.reindex(columns=desired_columns)

        return sar_object_dfs

    def add_on_sea_column(self) -> None:
        """
        Adds an 'on_sea' column to each SAR object DataFrame in the sar_object_dfs attribute,
        indicating whether each point is on water (True) or on land (False) using RoaringLandmask.
        """
        # Create an instance of RoaringLandmask
        landmask = RoaringLandmask.new()
        # Create an instance of RoaringLandmask
        for date, df in self.sar_object_dfs.items():
            # Get latitude and longitude as arrays
            latitudes = df['latitude'].values
            longitudes = df['longitude'].values
            
            # Use contains_many to check all points in one go
            on_land = landmask.contains_many(longitudes, latitudes)  # Note: longitudes first, then latitudes

            # Create 'on_sea' column based on whether the points are on water
            df['on_sea'] = ~on_land  # on_sea is True if not on land

            # Update the dictionary with the new DataFrame containing the 'on_sea' column
            self.sar_object_dfs[date] = df

    # Function to compute nearest shoreline using vectorized Haversine distance and bounding box optimization
    def compute_nearest_shoreline_haversine(self, df, shoreline_gdf, buffer=0.5):
        """
        Adds a column 'distance_to_shoreline' to the DataFrame with Haversine distance in km.
        Clips the shoreline GeoDataFrame using a bounding box around the SAR positions.
        Uses vectorized operations to calculate distances.

        Args:
            df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns.
            shoreline_gdf (gpd.GeoDataFrame): GeoDataFrame containing the shoreline polygons.
            buffer (float): Buffer value to extend the bounding box (default is 0.5 degrees).

        Returns:
            pd.DataFrame: Original DataFrame with an added 'distance_to_shoreline' column.
        """
        # Function to create a bounding box from DataFrame coordinates with a buffer
        def create_bounding_box(df, buffer=0.1):
            """
            Create a bounding box around the latitude and longitude values in the DataFrame, with a specified buffer.
            
            Args:
                df (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns.
                buffer (float): Buffer to add to the bounding box in degrees (default is 0.1 degrees).
                
            Returns:
                shapely.geometry.box: A bounding box geometry that covers the data points with added buffer.
            """
            minx, miny = df['longitude'].min() - buffer, df['latitude'].min() - buffer
            maxx, maxy = df['longitude'].max() + buffer, df['latitude'].max() + buffer
            return box(minx, miny, maxx, maxy)
        
        # Haversine function using numpy for vectorized operations
        def haversine_vec(lat1, lon1, lat2, lon2):
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

        # Create a bounding box from the SAR positions with a buffer
        bounding_box = create_bounding_box(df, buffer=buffer)

        # Clip the shoreline GeoDataFrame using the bounding box
        shoreline_clipped = shoreline_gdf.clip(bounding_box)

        # Ensure the clipped shoreline geometry is unified and lineal
        shoreline_lines = shoreline_clipped.exterior.union_all()

        # Convert SAR positions to a GeoDataFrame
        gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))

        # Compute nearest points using vectorized approach
        nearest_points = gdf_points.geometry.apply(lambda point: shoreline_lines.interpolate(shoreline_lines.project(point)))

        # Calculate Haversine distance using vectorized numpy operations
        df['distance_to_shoreline'] = haversine_vec(
            df['latitude'].values, df['longitude'].values, nearest_points.y.values, nearest_points.x.values
        )

        return df