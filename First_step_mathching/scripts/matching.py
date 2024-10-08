import pandas as pd
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

class Matching():
	def ais_find_matching_vessels(AIS_data, comparison_data, date_key, delta_time):
		"""
		Find vessels in AIS_data whose timestamps are close to the timestamps in comparison_data within a given delta_time.

		Parameters:
		- AIS_data: DataFrame containing AIS data with 'mmsi' and 'TimeStamp'.
		- comparison_data: DataFrame containing comparison data (e.g., SAR_data or norsat_data) with 'TimeStamp'.
		- date_key: The specific date to use as a key in both AIS_data and comparison_data.
		- delta_time: The pd.Timedelta object representing the time threshold for comparison.

		Returns:
		- grouped_df: DataFrame grouped by 'mmsi' with only the groups where the threshold is met.
		"""

		# Group the AIS data by 'mmsi'
		ais_mmsi = AIS_data[date_key].groupby(by=['mmsi'])

		# Create an empty list to store the results
		results = []

		# Iterate over each group in 'ais_mmsi'
		for mmsi_tuple, group in ais_mmsi:
			# Extract the mmsi value from the tuple
			mmsi = mmsi_tuple  # Since you're grouping by a single column, this is already a scalar

			matched = False  # Flag to check if there is any match for this group

			# Iterate over each timestamp in the group (could be multiple for a single mmsi)
			for ais_timestamp in group['TimeStamp']:
				# Iterate over each timestamp in comparison_data
				for comp_timestamp in comparison_data[date_key]['TimeStamp']:
					# Calculate the time difference and check if it is within the delta_time threshold
					difference = abs(ais_timestamp - comp_timestamp)
					if difference <= delta_time:
						matched = True
						# If a match is found, append the whole group to the results
						group['time_difference'] = difference
						group['comparison_timestamp'] = comp_timestamp
						break  # Stop checking once a match is found

				if matched:
					# Append the entire group to results
					results.append(group)
					break  # Stop checking other timestamps for this mmsi if a match is found

		# Create a new DataFrame from the results
		if results:
			# Concatenate all the matched groups into a single DataFrame
			matching_df = pd.concat(results, ignore_index=True)
			# Group the DataFrame by 'mmsi'
			grouped_df = matching_df.groupby(by=['mmsi'])
			return grouped_df
		else:
			# If no results, return an empty DataFrame grouped by 'mmsi'
			return pd.DataFrame([]).groupby(by=['mmsi'])

	def haversine(lat1, lon1, lat2, lon2):
		# Convert latitude and longitude to radians
		lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

		# Calculate the difference between the two coordinates
		dlat = lat2 - lat1
		dlon = lon2 - lon1

		# Apply the haversine formula
		a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
		c = 2 * math.asin(math.sqrt(a))

		# Calculate the radius of the Earth
		r = 6371 # radius of Earth in kilometers

		# Return the distance
		return c * r
	
	
class hungarian_method():
	def __init__(self, ship_dict : dict, sar_expanded : pd.DataFrame):
		self.matches, self.cost_matrix  = self.match_ships_hungarian_vectorized(ship_dict, sar_expanded)

	# Vectorized Haversine function
	def haversine_vec(self, lat1, lon1, lat2, lon2):
		# Convert latitude and longitude to radians
		lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

		# Calculate the difference between the two coordinates
		dlat = lat2 - lat1
		dlon = lon2 - lon1

		# Apply the haversine formula
		a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
		c = 2 * np.arcsin(np.sqrt(a))

		# Radius of Earth in kilometers
		r = 6371

		# Return the distance in km
		return c * r
	
	# Create a vectorized cost matrix (haversine distances)
	def create_cost_matrix_vectorized(self, ship_dict, sar_expanded):
		ship_coords = np.array([(coords['y'], coords['x']) for coords in ship_dict.values()])
		sar_coords = sar_expanded[['latitude', 'longitude']].to_numpy()

		# Vectorized haversine distance calculation for each pair of ships and SAR detections
		ship_lats, ship_lons = ship_coords[:, 0], ship_coords[:, 1]
		sar_lats, sar_lons = sar_coords[:, 0], sar_coords[:, 1]

		# Broadcasting the ship and SAR coordinates to calculate all distances at once
		ship_lats_b, sar_lats_b = np.broadcast_arrays(ship_lats[:, None], sar_lats[None, :])
		ship_lons_b, sar_lons_b = np.broadcast_arrays(ship_lons[:, None], sar_lons[None, :])

		cost_matrix = self.haversine_vec(ship_lats_b, ship_lons_b, sar_lats_b, sar_lons_b)
		ship_keys = list(ship_dict.keys())

		return cost_matrix, ship_keys

	# Function to match ships to SAR data using the Hungarian algorithm
	def match_ships_hungarian_vectorized(self, ship_dict, sar_expanded):
		cost_matrix, ship_keys = self.create_cost_matrix_vectorized(ship_dict, sar_expanded)

		# Apply Hungarian algorithm to minimize the total cost (distance)
		ship_indices, sar_indices = linear_sum_assignment(cost_matrix)

		# Create a list of matches
		matches = []
		for ship_idx, sar_idx in zip(ship_indices, sar_indices):
			mmsi = ship_keys[ship_idx]
			match = {
				'mmsi': mmsi,
				'sar_idx' : sar_idx,
				'distance_km': cost_matrix[ship_idx, sar_idx]
			}
			matches.append(match)

		return pd.DataFrame(matches), cost_matrix