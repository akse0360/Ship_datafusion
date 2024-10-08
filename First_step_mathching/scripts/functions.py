from math import radians, sin, cos, sqrt, atan2
import numpy as np
from shapely.geometry import box

class functions:
    def haversine(self,lat1, lon1, lat2, lon2):
        
        # Radius of the Earth in kilometers
        R = 6371.0
        
        # Convert latitude and longitude from degrees to radians
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        return R * c
    # Haversine function using numpy for vectorized operations
    
    
    # Function to create a bounding box from DataFrame coordinates with a buffer
    