import json
import pandas as pd
from roaring_landmask import RoaringLandmask

class DataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.dfs_ais = {}
        self.dfs_sar = {}
        self.dfs_norsat = {}

    # Method to load and format AIS data
    def load_ais_data(self, ais_files):
        self.dfs_ais = {date: pd.read_csv(f"{self.base_path}{file}") for date, file in ais_files.items()}
        for df in self.dfs_ais.values():
            df.rename(columns={'bs_ts': 'time', 'lat': 'latitude', 'lon': 'longitude'}, inplace=True)
            df['source'] = 'ais'
            df["TimeStamp"] = pd.to_datetime(df['time'])

    # Method to load and format SAR data
    def load_sar_data(self, sar_files):
        self.dfs_sar = {date: pd.read_json(f"{self.base_path}{file}", orient='index') for date, file in sar_files.items()}
        for df in self.dfs_sar.values():
            df['source'] = 'sar'
            df['Start'] = pd.to_datetime(df['Start'])
            df['End'] = pd.to_datetime(df['End'])

    # Method to load and format Norsat data
    def load_norsat_data(self, norsat_files):
        self.dfs_norsat = {date: pd.read_json(f"{self.base_path}{file}") for date, file in norsat_files.items()}
        for date, df in self.dfs_norsat.items():
            df['source'] = 'norsat'
            self.dfs_norsat[date] = self.norsat_formatting(df)

    ############ Norsat FILTERS ############
    # Integrated method to format Norsat data
    def norsat_formatting(self, df):
        df['latitude'] = df['NRDEmitterPosition'].apply(lambda x: x.get('Latitude') if isinstance(x, dict) else None)
        df['longitude'] = df['NRDEmitterPosition'].apply(lambda x: x.get('Longitude') if isinstance(x, dict) else None)
        return df
    
    ############ SAR FILTERS ############
    # Method to expand SAR objects for a given date
    def expand_objects_for_date(self, date_key: str) -> pd.DataFrame:
        """
        Expands the objects from a specific date in the dfs_sar dictionary into a DataFrame.

        Parameters:
        - date_key: The specific date (key) in the format 'DD-MM-YYYY' for which to expand the objects.

        Returns:
        - A DataFrame with the expanded objects for the specified date.
        """
        if date_key not in self.dfs_sar:
            raise ValueError(f"Date {date_key} not found in dfs_sar.")

        df = self.dfs_sar[date_key]
        expanded_data = []

        for _, row in df.iterrows():
            start_time = row['Start']
            end_time = row['End']
            objects = row['Objects']

            # Convert the JSON objects to a dict if stored as a string
            if isinstance(objects, str):
                objects = json.loads(objects)

            # Expand each object in the 'Objects' field
            for obj_id, obj_data in objects.items():
                expanded_row = {
                    'Start': start_time,
                    'End': end_time,
                    'Object_ID': obj_id,
                    'x': obj_data['x'],
                    'y': obj_data['y'],
                    'width': obj_data['width'],
                    'height': obj_data['height'],
                    'class': obj_data['class'],
                    'latitude': obj_data.get('latitude'),
                    'longitude': obj_data.get('longitude'),
                    'probabilities': obj_data.get('probabilities'),
                    'encoded_image': obj_data.get('encoded_image')
                }
                expanded_data.append(expanded_row)

        return pd.DataFrame(expanded_data)

    def filter_sar_landmask(self, filtered_sar_df):
        filtered_sar_df = filtered_sar_df.copy()

        filtered_sar_df.loc[:, 'on_land'] = RoaringLandmask.new().contains_many(
            filtered_sar_df['latitude'].to_numpy(), 
            filtered_sar_df['longitude'].to_numpy()
        )
        return filtered_sar_df

    ############ AIS FILTERS ############
    # Method to filter AIS data based on SAR timestamps
    def filter_ais_data(self, date_key, delta_time):
        ais_df = self.dfs_ais[date_key]
        sar_df = self.dfs_sar[date_key]
        return ais_df[
            (ais_df['TimeStamp'] >= sar_df['Start'][0] - delta_time)
            & (ais_df['TimeStamp'] <= sar_df['Start'][0] + delta_time)
        ]

    # Method to convert data types
    def convert_data_types(self, df, date_columns, numeric_columns):
        for col in date_columns:
            df.loc[:, col] = pd.to_datetime(df[col])
        for col in numeric_columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    # Method to clean data (remove rows with NaN in specific columns)
    def clean_data(self, df, columns_to_check):
        return df.dropna(subset=columns_to_check)

    # Method to display data structure
    def display_data_structure(self):
        print(f"AIS:\n{self.dfs_ais.keys()}\nColumns: {self.dfs_ais['02-11-2022'].columns}")
        print(f"SAR:\n{self.dfs_sar.keys()}\nColumns: {self.dfs_sar['02-11-2022'].columns}")
        print(f"Norsat:\n{self.dfs_norsat.keys()}\nColumns: {self.dfs_norsat['02-11-2022'].columns}")