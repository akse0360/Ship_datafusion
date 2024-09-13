# data_processor.pyx
import pandas as pd
import json
import cython
from cython.libc.stdlib cimport malloc, free

cdef class DataProcessor:
    cdef str base_path
    cdef dict dfs_ais
    cdef dict dfs_sar
    cdef dict dfs_norsat

    def __init__(self, base_path):
        self.base_path = base_path
        self.dfs_ais = {}
        self.dfs_sar = {}
        self.dfs_norsat = {}

    def load_ais_data(self, ais_files):
        self.dfs_ais = {date: pd.read_csv(f"{self.base_path}{file}") for date, file in ais_files.items()}
        for df in self.dfs_ais.values():
            df.rename(columns={'bs_ts': 'time', 'lat': 'latitude', 'lon': 'longitude'}, inplace=True)
            df['source'] = 'ais'
            df["TimeStamp"] = pd.to_datetime(df['time'])

    def load_sar_data(self, sar_files):
        self.dfs_sar = {date: pd.read_json(f"{self.base_path}{file}", orient='index') for date, file in sar_files.items()}
        for df in self.dfs_sar.values():
            df['source'] = 'sar'
            df['Start'] = pd.to_datetime(df['Start'])
            df['End'] = pd.to_datetime(df['End'])

    def load_norsat_data(self, norsat_files):
        self.dfs_norsat = {date: pd.read_json(f"{self.base_path}{file}") for date, file in norsat_files.items()}
        for date, df in self.dfs_norsat.items():
            df['source'] = 'norsat'
            self.dfs_norsat[date] = self.norsat_formatting(df)

    def norsat_formatting(self, df):
        cdef int n = df.shape[0]
        cdef list latitudes = []
        cdef list longitudes = []
        for i in range(n):
            position = df['NRDEmitterPosition'].iloc[i]
            latitudes.append(position.get('Latitude') if isinstance(position, dict) else None)
            longitudes.append(position.get('Longitude') if isinstance(position, dict) else None)
        df['latitude'] = latitudes
        df['longitude'] = longitudes
        return df

    def expand_objects_for_date(self, date_key: str) -> pd.DataFrame:
        if date_key not in self.dfs_sar:
            raise ValueError(f"Date {date_key} not found in dfs_sar.")

        df = self.dfs_sar[date_key]
        expanded_data = []

        for _, row in df.iterrows():
            start_time = row['Start']
            end_time = row['End']
            objects = row['Objects']

            if isinstance(objects, str):
                objects = json.loads(objects)

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

    def filter_ais_data(self, date_key, delta_time):
        ais_df = self.dfs_ais[date_key]
        sar_df = self.dfs_sar[date_key]
        return ais_df[
            (ais_df['TimeStamp'] >= sar_df['Start'][0] - delta_time)
            & (ais_df['TimeStamp'] <= sar_df['Start'][0] + delta_time)
        ]

    def convert_data_types(self, df, date_columns, numeric_columns):
        for col in date_columns:
            df.loc[:, col] = pd.to_datetime(df[col])
        for col in numeric_columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

    def clean_data(self, df, columns_to_check):
        return df.dropna(subset=columns_to_check)

    def display_data_structure(self):
        print(f"AIS:\n{self.dfs_ais.keys()}\nColumns: {self.dfs_ais['02-11-2022'].columns}")
        print(f"SAR:\n{self.dfs_sar.keys()}\nColumns: {self.dfs_sar['02-11-2022'].columns}")
        print(f"Norsat:\n{self.dfs_norsat.keys()}\nColumns: {self.dfs_norsat['02-11-2022'].columns}")
