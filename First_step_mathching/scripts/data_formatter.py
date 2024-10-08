import json
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d, CubicSpline
from roaring_landmask import RoaringLandmask

class DataProcessor:
	"""
	Processes and manipulates data within DataFrames for various analytical tasks.

	The DataProcessor class provides methods to clean data by removing rows with missing values, 
	convert data types for specified columns, filter data based on geographical land presence, 
	and expand SAR DataFrames by extracting and flattening object data for specific dates. 
	These methods facilitate the preparation and transformation of data for further analysis.

	Methods:
		clean_data(df, columns_to_check):
			Cleans the input DataFrame by removing rows with missing values in specified columns.

		convert_data_types(df, date_columns, numeric_columns):
			Converts specified columns in a DataFrame to appropriate data types.

		filter_landmask(df):
			Filters a DataFrame to indicate whether points are on land based on a landmask.

		filter_df_to_image_by_time(df, image_df, delta_time):
			Filters a DataFrame to include only rows within a specified time range relative to an image's start time.

		expand_SAR_df_for_date(dfs_dict, date_key):
			Expands the SAR DataFrame for a specific date by extracting and flattening object data.

	"""

	def clean_data(df, columns_to_check) -> None:
		"""
		Cleans the input DataFrame by removing rows with missing values in specified columns.

		This function takes a DataFrame and a list of columns to check for missing values, 
		and it removes any rows that contain NaN values in those columns. The operation modifies 
		the DataFrame in place.

		Args:
			df (pd.DataFrame): The DataFrame to be cleaned.
			columns_to_check (list): A list of column names to check for missing values.

		Returns:
			None
		"""
		return df.dropna(subset=columns_to_check)
	
	def convert_data_types(df, date_columns, numeric_columns) -> None:
		"""
		Converts specified columns in a DataFrame to appropriate data types.

		This function takes a DataFrame and lists of column names for date and numeric types, 
		and converts the specified columns to datetime and numeric formats, respectively. 
		The operation modifies the DataFrame in place.

		Args:
			df (pd.DataFrame): The DataFrame whose columns are to be converted.
			date_columns (list): A list of column names to be converted to datetime.
			numeric_columns (list): A list of column names to be converted to numeric types.

		Returns:
			None
		"""

		for col in date_columns:
			df.loc[:, col] = pd.to_datetime(df[col])
		for col in numeric_columns:
			df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')

	def filter_landmask(df : pd.DataFrame) -> pd.DataFrame:
		"""
		Filters a DataFrame to indicate whether points are on land based on a landmask.

		This function takes a DataFrame containing longitude and latitude coordinates, 
		checks each point against a predefined landmask, and adds a new column indicating 
		whether each point is on land.

		Args:
			df (pd.DataFrame): The input DataFrame containing 'longitude' and 'latitude' columns.

		Returns:
			pd.DataFrame: A new DataFrame with an additional 'on_land' column indicating land presence.
		"""

		landmask = RoaringLandmask.new()
		filtered_df = df.copy()

		filtered_df.loc[:, 'on_land'] = landmask.contains_many(
			filtered_df['longitude'].to_numpy(), filtered_df['latitude'].to_numpy())
		return filtered_df

	def filter_df_to_image_by_time(self, df : pd.DataFrame, image_df :  pd.DataFrame, delta_time : pd.Timedelta) -> pd.DataFrame:
		"""
		Filters a DataFrame to include only rows within a specified time range relative to an image's start time.

		This function takes a DataFrame containing timestamps and an image DataFrame with a start time, 
		and returns a new DataFrame that includes only the rows where the timestamps fall within a 
		specified time delta before and after the image's start time.

		Args:
			df (pd.DataFrame): The input DataFrame containing a 'TimeStamp' column.
			image_df (pd.DataFrame): The DataFrame containing a 'Start' column with the image start time.
			delta_time (pd.Timedelta): The time range to filter the DataFrame around the image start time.

		Returns:
			pd.DataFrame: A new DataFrame containing only the rows within the specified time range.
		"""

		filtered_df = df.copy()
		return filtered_df[
			(filtered_df['TimeStamp'] >= image_df['Start'][0] - delta_time)
			& (filtered_df['TimeStamp'] <= image_df['Start'][0] + delta_time)
		]

	def expand_SAR_df_for_date(dfs_dict: dict, date_key: str) -> pd.DataFrame:
		"""
		Expands the SAR DataFrame for a specific date by extracting and flattening object data.

		This function retrieves the DataFrame corresponding to the provided date key, checks 
		if the date exists, and then iterates through each row to expand the 'Objects' field 
		into individual rows. The resulting DataFrame contains detailed information about each 
		object, including its position, dimensions, and associated metadata.

		Args:
			dfs_dict (dict): Dictionary containing Dataframes
			date_key (str): The key representing the date for which to expand the SAR DataFrame.

		Returns:
			pd.DataFrame: A new DataFrame containing expanded object data for the specified date.

		Raises:
			ValueError: If the provided date_key is not found in the dfs_sar.
		"""
		# Check if the specified date exists in the dictionary
		if date_key not in dfs_dict:
			raise ValueError(f"Date {date_key} not found in dfs_sar.")

		# Retrieve the DataFrame for the specified date
		df = dfs_dict[date_key]
		expanded_data = []

		# Iterate over each row in the DataFrame
		for _, row in df.iterrows():
			start_time = row['TimeStamp']
			objects = row['Objects']

			# Convert the JSON objects to a dict if stored as a string
			if isinstance(objects, str):
				objects = json.loads(objects)

			# Expand each object in the 'Objects' field
			for obj_id, obj_data in objects.items():
				expanded_row = {
					'TimeStamp': start_time,
					'sar_id': f"{date_key}_{obj_id}",  # Unique identifier combining date and object ID
					'latitude': obj_data.get('latitude'),
					'longitude': obj_data.get('longitude'),
					'width': obj_data['width'],
					'height': obj_data['height'],
					'probabilities': obj_data.get('probabilities'),
					'source': 'SAR'
				}
				expanded_data.append(expanded_row)

		# Create the expanded DataFrame
		expanded_df = pd.DataFrame(expanded_data)
		return expanded_df


	def ais_interpolate_mmsi_points(ais_data, date_key, interpolation_time_col):
		"""
		Interpolates MMSI points from the given DataFrame in ais_data using the specified date_key.

		Parameters:
		- ais_data: A dictionary of DataFrames containing AIS data.
		- date_key: The key to access the specific DataFrame in ais_data.
		- interpolation_time: The time to interpolate to (should be a pandas Timestamp).

		Returns:
		- A tuple containing:
			- A dictionary with interpolated x and y coordinates for each MMSI at time = interpolation_time.
			- A list of MMSI numbers with insufficient data points.
		"""
		if isinstance(ais_data, dict):
			# Grouping MMSI points for temporal matching
			ais_mmsi = ais_data[date_key].groupby(by=['mmsi'])
		elif isinstance(ais_data, pd.core.groupby.DataFrameGroupBy):
			ais_mmsi = ais_data

		mmsi_numbers = list(ais_mmsi.indices.keys())
		not_enough = []  # List to store MMSI numbers with insufficient data points
		xy_interpolated = {}

		interpolation_time = pd.to_numeric(interpolation_time_col).mean()

		for mmsi in mmsi_numbers:
			tes = ais_mmsi.get_group((mmsi,))

			# Check for NaN values and drop them
			if tes['TimeStamp'].isnull().any() or tes['longitude'].isnull().any() or tes['latitude'].isnull().any():
				continue  # Skip this group if there are NaN values

			# Ensure there are at least two points for interpolation
			if len(tes) < 2:
				not_enough.append(mmsi)  # Append MMSI to the list
				continue

			# Prepare the time and coordinate data for interpolation
			time_numeric = pd.to_numeric(tes['TimeStamp'])
			longitude = tes['longitude'].values
			latitude = tes['latitude'].values

			# Ensure that interpolation_time is within the bounds of time_numeric to avoid extrapolation
			if interpolation_time < time_numeric.min() or interpolation_time > time_numeric.max():
				not_enough.append(mmsi)  # Append MMSI to not_enough if interpolation time is out of bounds
				continue
			# Choose interpolation method based on the number of points
			if len(tes) == 2 or len(tes) <= 3:
				interpol = interp1d(time_numeric, np.c_[longitude, latitude], axis=0, kind='linear')
			else:
				interpol = CubicSpline(time_numeric, np.c_[longitude, latitude])
			# Perform interpolation for the specified interpolation time
			x_interp, y_interp = interpol(interpolation_time).T
			xy_interpolated[mmsi] = {'x': x_interp, 'y': y_interp}

		# After processing, you can check the not_enough list
		print(f"MMSI numbers with insufficient data points: {len(not_enough)} out of {len(mmsi_numbers)}")

		return xy_interpolated, not_enough
