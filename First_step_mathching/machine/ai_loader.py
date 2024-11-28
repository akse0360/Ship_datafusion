import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from scripts.data_loader import DataLoader as DL
import pandas as pd

class ais_dataset(Dataset):
    def __init__(self, data, grouping_id: str = 'mmsi'):
        self.max_sog = 102.2  # Max allowable SOG according to AIS specifications
        
        # Step 1: Process data to handle missing values and normalize
        processed_data = self.process_data(data)

        # Step 2: Filter out groups with track length < 2 and keep only those MMSIs
        grouped_data = processed_data.groupby(grouping_id)
        self.mmsi_list = [
            mmsi for mmsi, group in grouped_data
            if len(group) >= 2
        ]
        
        # Step 3: Store grouped data for easy access
        self.grouped_data = grouped_data

    def __getitem__(self, idx):
        # Access the group by the index
        mmsi = self.mmsi_list[idx]
        mmsi_group = self.grouped_data.get_group(mmsi)

        # Extract only latitude and longitude columns
        track = mmsi_group[['latitude', 'longitude']].values

        # Sample two points from the track
        t = torch.randint(0, len(track) - 1, (1,)).item()
        ais1 = torch.tensor(track[t], dtype=torch.float32)
        ais2 = torch.tensor(track[t + 1], dtype=torch.float32)

        return ais1, ais2

    def __len__(self):
        return len(self.mmsi_list)

    def process_data(self, df):
        """
        Normalize the input data using sine and cosine for 'cog', 'latitude', and 'longitude',
        and normalize 'sog'.

        Args:
            df: a Pandas DataFrame with AIS data
        Returns:
            df: a Pandas DataFrame with normalized data
        """
        absurd_value_cog = -999.0
        absurd_value_sog = -999.0

        # Replace missing values in 'cog' and 'sog' with group-wise means
        df['cog'] = df['cog'].replace(absurd_value_cog, np.nan)
        df['sog'] = df['sog'].replace(absurd_value_sog, np.nan)

        df['cog'] = df.groupby('mmsi')['cog'].transform(lambda x: x.fillna(x.mean()))
        df['sog'] = df.groupby('mmsi')['sog'].transform(lambda x: x.fillna(x.mean()))

        # Normalize 'sog' and calculate sine/cosine transformations
        df['sog'] = df['sog'] / self.max_sog
        df['cog_sin'] = np.sin(np.deg2rad(df['cog']))
        df['cog_cos'] = np.cos(np.deg2rad(df['cog']))
        
        df['lat_sin'] = np.sin(np.deg2rad(df['latitude']))
        df['lat_cos'] = np.cos(np.deg2rad(df['latitude']))
        df['lon_sin'] = np.sin(np.deg2rad(df['longitude']))
        df['lon_cos'] = np.cos(np.deg2rad(df['longitude']))


        df['latitude'] = df['latitude'].dropna()
        df['longitude'] = df['longitude'].dropna()

        # Dynamically determine min and max for latitude and longitude
        # lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        # lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

        # # Min-max scaling for latitude and longitude
        # df['latitude'] = (df['latitude'] - lat_min) / (lat_max - lat_min)
        # df['longitude'] = (df['longitude'] - lon_min) / (lon_max - lon_min)


        return df
    
    @staticmethod
    def split_dataset(dataset, split_ratio=0.8, seed=1, batch_size=3):
        indices = list(range(len(dataset)))

        num_train = int(split_ratio * len(indices))
        random.seed(seed)
        indices = sorted(indices, key=lambda x: random.random())
        train_indices = indices[:num_train]
        valid_indices = indices[num_train:]

        # Create samplers for training and validation
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        # Create data loaders using the samplers
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, pin_memory=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, drop_last=True, pin_memory=True)
        return train_loader, val_loader
    
    @staticmethod
    def import_data_fn():
        # PATHS, dataframe and shpfile #
        # Define paths
        base_path = "C:\\Users\\abelt\\OneDrive\\Dokumenter\\GitHub\\Ship_datafusion\\data"
        ## File names ##
        # AIS
        ais_files = {
            '02-11-2022': 'ais\\ais_110215.csv',
            '03-11-2022': 'ais\\ais_110315.csv',
            '05-11-2022': 'ais\\ais_1105.csv',
           # "01-03-2024": 'ais\\denmark\\aisdk_2024_03_01.csv',
           # "02-03-2024": 'ais\\denmark\\aisdk_2024_03_02.csv',
           # "03-03-2024": 'ais\\denmark\\aisdk_2024_03_03.csv',
        }

        # LOADING #
        try:
            data_loader = DL(base_path=base_path, ais_files=ais_files)
            ais_loader, _, _ = data_loader.load_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

        # Create a DataFrame with unique tracks
        unique_tracks_df = ais_dataset.create_unique_tracks_dataframe(ais_loader.dfs_ais.copy())
        test = unique_tracks_df.groupby('track_id')
        print(len(test.groups.keys()))
        return unique_tracks_df

    @staticmethod
    def create_unique_tracks_dataframe(dfs_ais):
        unique_tracks = []
        track_id = 1  # Start track ID from 1
        
        for _, df in dfs_ais.items():
            grouped = df.groupby('mmsi')
            for _, track in grouped:
                # Assign the current track_id to each row in the track
                track = track.copy()  # Avoid modifying the original data
                track['track_id'] = track_id
                unique_tracks.append(track)
                track_id += 1  # Increment track_id for the next unique track
        
        return pd.concat(unique_tracks, ignore_index=True)

# Usage example:
# unique_tracks_df = ais_dataset.import_data_fn()
# print(unique_tracks_df)
