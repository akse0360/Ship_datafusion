import random
import torch
from torch.utils.data import Dataset
import numpy as np


class ais_dataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data: a Pandas DataFrame with AIS data
        Returns: None
        """
        self.max_sog = 102.2  # Max allowable SOG according to AIS specifications
        
        #print(data)
        # Process the input data
        processed_data = self.process_data(data)
        
        # Group the processed data by 'mmsi'
        self.grouped_data = processed_data.groupby('mmsi')
        self.mmsi_list = list(self.grouped_data.groups.keys())
        #print(self.grouped_data)

    def process_data(self, df):
        """
        Normalize the input data using sine and cosine for 'cog', 'latitude', and 'longitude',
        and normalize 'sog'.

        Args:
            df: a Pandas DataFrame with AIS data
        Returns:
            df: a Pandas DataFrame with normalized data
        """
        # Fill missing values for 'cog' and 'sog' with absurd values
        absurd_value_cog = -999.0
        absurd_value_sog = -999.0

        # Replace missing values with absurd values
        df['cog'] = df['cog'].fillna(absurd_value_cog)
        df['sog'] = df['sog'].fillna(absurd_value_sog)

        # Convert 'cog' to radians
        df['cog_sin'] = np.where(
            df['cog'] != absurd_value_cog, 
            np.sin(np.deg2rad(df['cog'])), 
            0.0  # Set to 0 for missing values
        )
        df['cog_cos'] = np.where(
            df['cog'] != absurd_value_cog, 
            np.cos(np.deg2rad(df['cog'])), 
            0.0 
            )

        # Normalize 'sog' 
        df['sog'] = np.where(
            df['sog'] != absurd_value_sog, 
            df['sog'] / self.max_sog, 
            0.0  
        )

        # Compute sine and cosine for latitude and longitude
        df['lat_sin'] = np.sin(np.deg2rad(df['latitude']))
        df['lat_cos'] = np.cos(np.deg2rad(df['latitude']))
        df['lon_sin'] = np.sin(np.deg2rad(df['longitude']))
        df['lon_cos'] = np.cos(np.deg2rad(df['longitude']))

        return df

    def __len__(self):
        return len(self.mmsi_list)

    def __getitem__(self, idx):
        
        mmsi_group = self.grouped_data.get_group(self.mmsi_list[idx]) # Get the mmsi group and extract the necessary AIS points
        
        #track = mmsi_group[['lat_sin', 'lon_cos', 'sog', 'cog_sin', 'cog_cos']].values # Extract the relevant columns including sine and cosine of 'cog'
        track = mmsi_group[['latitude', 'longitude']].values
        if len(track) > 1:

            t = torch.randint(0, len(track) - 1, (1,)).item()
            ais1 = track[t]
            ais2 = track[t + 1]
            
            ais1 = torch.tensor(ais1, dtype=torch.float32)
            ais2 = torch.tensor(ais2, dtype=torch.float32)
        else:
            ais1 = torch.zeros(2)
            ais2 = torch.zeros(2)
        #print(ais1, ais2)

        return ais1, ais2