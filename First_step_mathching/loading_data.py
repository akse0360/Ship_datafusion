## LOADING DATA and filtering ##
# Libraries
import pandas as pd

from scripts.data_loader import Load_ais, Load_sar, Load_norsat
from scripts.data_formatter import DataProcessor as dp
from scripts.matching import Matching as ma

from datetime import datetime

def loading_main():
    now = datetime.now()

    current_time = now.strftime("%d%m_%H%M%S")

    # Define date and time filter
    date_key = '03-11-2022'
    delta_time = pd.Timedelta(days = 0, hours = 5, minutes = 0)

    # Time match AIS to sar and norsat
    time_diff = pd.Timedelta(days=0, hours=0, minutes=60)

    # Define paths
    base_path = "C:\\Users\\abelt\\OneDrive\\Desktop\\Kandidat\\"
    ## File names ##
    # AIS
    ais_files = {
        '02-11-2022': 'ais\\ais_110215.csv',
        '03-11-2022': 'ais\\ais_110315.csv',
        '05-11-2022': 'ais\\ais_1105.csv'
    }
    # SAR
    sar_files = {
        '02-11-2022': 'sar\\Sentinel_1_detection_20221102T1519.json',
        '03-11-2022': 'sar\\Sentinel_1_detection_20221103T154515.json',
        '05-11-2022': 'sar\\Sentinel_1_detection_20221105T162459.json'
    }
    # Norsat
    norsat_files = {
        '02-11-2022': 'norsat\\Norsat3-N1-JSON-Message-DK-2022-11-02T151459Z.json',
        '03-11-2022': 'norsat\\Norsat3-N1-JSON-Message-DK-2022-11-03T152759Z.json',
        '05-11-2022': 'norsat\\Norsat3-N1-JSON-Message-DK-2022-11-05T155259Z.json'
    }

    # Storing data in a dictionary with the date as the key:
    AIS_data = Load_ais(base_path = base_path, ais_files = ais_files).dfs_ais
    SAR_data = Load_sar(base_path = base_path, sar_files = sar_files).dfs_sar
    norsat_data = Load_norsat(base_path = base_path, norsat_files = norsat_files).dfs_norsat

    # Expanded SAR processing with landmask:
    SAR_expanded = dp.filter_landmask(df = dp.expand_SAR_df_for_date(dfs_dict = SAR_data, date_key = date_key))
    SAR_on_sea = SAR_expanded[SAR_expanded['on_land'] == False]


## LOADING DATA and filtering ##
# Libraries
import pandas as pd

from scripts.data_loader import Load_ais, Load_sar, Load_norsat
from scripts.data_formatter import DataProcessor as dp
from scripts.matching import Matching as ma

from datetime import datetime

def loading_main(base_path: str = None, ais_files: dict = None, sar_files: dict = None, norsat_files: dict = None, date_key: str = None):
    """
    Loads and processes AIS, SAR, and Norsat data, matching AIS data with SAR and Norsat based on time. This function returns structured data for further analysis.

    Args:
        base_path (str, optional): The base path for loading data files.
        ais_files (dict, optional): A dictionary containing AIS file paths.
        sar_files (dict, optional): A dictionary containing SAR file paths.
        norsat_files (dict, optional): A dictionary containing Norsat file paths.
        date_key (str, optional): The key used to filter data by date.

    Returns:
        tuple: A tuple containing:
            - ais_data (dict): A dictionary with AIS data and matched vessels.
            - sar_data (dict): A dictionary with SAR data and processed information.
            - norsat_data (dict): The Norsat data.

    """
    # Initialize the variables to None, so they can be safely referenced later.
    AIS_data, SAR_data, SAR_expanded, SAR_on_sea, norsat_data = None, None, None, None, None
    time_matching_ais_sar, time_matching_ais_norsat = None, None
    
    # Define the time difference threshold for matching
    time_diff = pd.Timedelta(minutes=60)

    # Load AIS data if provided
    if ais_files:
        AIS_data = Load_ais(base_path=base_path, ais_files=ais_files).dfs_ais

    # Load and process SAR data if provided
    if sar_files:
        SAR_data = Load_sar(base_path=base_path, sar_files=sar_files).dfs_sar
        # Expanded SAR processing with landmask:
        SAR_expanded = dp.expand_SAR_df_for_date(dfs_dict=SAR_data, date_key=date_key)
        SAR_expanded = dp.filter_landmask(df=SAR_expanded)
        SAR_on_sea = SAR_expanded[SAR_expanded['on_land'] == False]

    # Load Norsat data if provided
    if norsat_files:
        norsat_data = Load_norsat(base_path=base_path, norsat_files=norsat_files).dfs_norsat

    # Perform time matching between AIS and SAR if both are available
    if AIS_data is not None and SAR_data is not None:
        time_matching_ais_sar = ma.ais_find_matching_vessels(AIS_data, SAR_data, date_key, time_diff)

    # Perform time matching between AIS and Norsat if both are available
    if AIS_data is not None and norsat_data is not None:
        time_matching_ais_norsat = ma.ais_find_matching_vessels(AIS_data, norsat_data, date_key, time_diff)

    # Prepare the data dictionaries for output
    ais_data = {'ais': AIS_data, 'tm_ais_sar': time_matching_ais_sar, 'tm_ais_norsat': time_matching_ais_norsat}
    sar_data = {'sar': SAR_data, 'sar_objects': SAR_expanded, 'sar_landmasked': SAR_on_sea}

    return ais_data, sar_data, norsat_data
