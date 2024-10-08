import pandas as pd
import os 

############ AIS LOADER ############
class Load_ais:
    """
    Loads and formats AIS data files for further analysis.

    This class is responsible for reading CSV files specified in a dictionary, renaming 
    columns for consistency, adding a source identifier, and converting the time column 
    to a datetime format. It also standardizes the DataFrame columns to the expected format.

    Args:
        base_path (str): The base path where the AIS files are located.
        ais_files (dict): A dictionary mapping dates to corresponding AIS file names.

    Attributes:
        dfs_ais (dict): A dictionary containing loaded AIS DataFrames with dates as keys.

    Methods:
        get_ais_data() -> dict: Returns the loaded AIS DataFrames as a dictionary.
    """

    # Expected columns for the final DataFrame
    expected_columns = ['ais_id', 'mmsi', 'TimeStamp', 'latitude', 'longitude', 
                        'length', 'width', 'sog', 'cog', 'source']

    def __init__(self, base_path: str, ais_files: dict) -> None:
        """
        Initializes a new instance of the LoadAIS class by loading and formatting AIS data files.

        This constructor reads CSV files specified in the ais_files dictionary, renames columns 
        for consistency, adds a source identifier, and converts the time column to a datetime 
        format for further processing. It also ensures that the DataFrame has the expected columns.

        Args:
            base_path (str): The base path where the AIS files are located.
            ais_files (dict): A dictionary mapping dates to corresponding AIS file names.

        Returns:
            None
        """
        # Load AIS CSV files into a dictionary of DataFrames
        self.dfs_ais = {
            date: pd.read_csv(os.path.join(base_path, file))
            .rename(columns={
                'bs_ts': 'time',
                'lat': 'latitude',
                'lon': 'longitude',
                'SOG': 'sog',
                'COG': 'cog',
                'MMSI': 'mmsi'
            })
            .assign(
                source='ais',
                TimeStamp=lambda df: pd.to_datetime(df['time'], utc=True) if 'time' in df.columns else pd.NaT
            )
            .reset_index()
            .rename(columns={'index': 'ais_id'})
            .reindex(columns=self.expected_columns, fill_value=pd.NA)
            for date, file in ais_files.items()
        }

        # Print keys and columns of all loaded DataFrames for verification

        print(f"AIS Data Loaded:\n{self.dfs_ais.keys()}")
        print(f"Columns for the first DataFrame: {self.dfs_ais[list(self.dfs_ais.keys())[0]].columns}")

    def get_ais_data(self) -> dict:
        """
        Returns the loaded AIS DataFrames.

        Returns:
            dict: A dictionary containing loaded AIS DataFrames with dates as keys.
        """
        return self.dfs_ais
