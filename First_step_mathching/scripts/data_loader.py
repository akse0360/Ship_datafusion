# Build-in Libs

# Indivicual data loaders:
from scripts.loaders.Load_sar import Load_sar
from scripts.loaders.Load_norsat import Load_norsat
from scripts.loaders.Load_ais import Load_ais

############ DAS LOADER ############
# Define a new class for the integrated data loader
class DataLoader:
    """
    The DataLoader class is responsible for loading and processing AIS, SAR, and Norsat data.
    It integrates AIS, SAR, and Norsat loading into a single interface and provides
    methods to process, match, and analyze the data.

    Methods:
        load_data: Load and process AIS, SAR, and Norsat data.
    """

    def __init__(self, base_path: str = None, ais_files: dict = None, sar_files: dict = None, norsat_files: dict = None, date_key: str = None):
        """
        Loads and processes AIS, SAR, and Norsat data based on the specified files, matching AIS data
        with SAR and Norsat using temporal and spatial parameters. This function returns the corresponding
        loaders that can be used for further analysis and data manipulation.

        Returns:
            tuple: A tuple containing three elements:
            - ais_loader (Load_ais or None): An instance of the `Load_ais` class containing loaded AIS data. 
            Returns `None` if no AIS files are provided.
            - sar_loader (Load_sar or None): An instance of the `Load_sar` class containing loaded SAR data.
            Returns `None` if no SAR files are provided.
            - norsat_loader (Load_norsat or None): An instance of the `Load_norsat` class containing loaded 
            Norsat data. Returns `None` if no Norsat files are provided.
        """
        self.base_path = base_path
        self.ais_files = ais_files
        self.sar_files = sar_files
        self.norsat_files = norsat_files
        self.date_key = date_key
                
    def load_data(self):
        """
        Loads and processes AIS, SAR, and Norsat data, matching AIS data with SAR and Norsat based on time.
        This function returns structured data for further analysis.

        Returns:
            tuple: A tuple containing:
                - ais_data (dict): A dictionary with AIS data and matched vessels.
                - sar_data (dict): A dictionary with SAR data and processed information.
                - norsat_data (dict): The Norsat data.
        """
        # Initialize the variables to None, so they can be safely referenced later.
        ais_loader, sar_loader, norsat_loader = None, None, None

        # Load AIS data if provided
        if self.ais_files:
            ais_loader = Load_ais(base_path=self.base_path, ais_files=self.ais_files)
        # Load Norsat data if provided
        if self.norsat_files:
            norsat_loader = Load_norsat(base_path=self.base_path, norsat_files=self.norsat_files)
        # Load SAR data if provided
        if self.sar_files:
            sar_loader = Load_sar(base_path=self.base_path, sar_files=self.sar_files)
        
        return ais_loader, sar_loader, norsat_loader
