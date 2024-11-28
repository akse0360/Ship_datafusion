from math import radians, sin, cos, sqrt, atan2
import numpy as np
from shapely.geometry import box

class functions:
    @staticmethod
    def haversine_distance(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
            """
            Calculate the Haversine distance between two points in vectorized form using numpy.

            Args:
                lat1, lon1, lat2, lon2: Arrays or Series representing latitude and longitude.

            Returns:
                np.ndarray: Haversine distance in kilometers.
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
    
    