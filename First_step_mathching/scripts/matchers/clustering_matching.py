import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN, HDBSCAN
from scipy.spatial import cKDTree
from typing import List 

# MLP MATCHING ALGORITHM #
import torch
from machine.mlp import MyModel 

class ClusteringMatcher:
## UTILITY FUNCTIONS ##
    # HAVERSINE DISTANCE #
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
    
    # HBSCAN CLUSTERING #
    @staticmethod
    def hdbscan_clustering(df1: pd.DataFrame, df2: pd.DataFrame, ids: List[str], sources: List[str], min_cluster_size: int, min_samples: int = None) -> pd.DataFrame:
        """
        Perform HDBSCAN clustering on combined DataFrames.

        Parameters:
        - df1: First DataFrame containing ['id', 'latitude', 'longitude'].
        - df2: Second DataFrame containing ['id', 'latitude', 'longitude'].
        - ids: List of column names representing 'id' in both DataFrames.
        - sources: List containing source names for df1 and df2.
        - min_cluster_size: Minimum cluster size for HDBSCAN.
        - min_samples: Number of samples in a neighborhood for a point to be considered a core point.

        Returns:
        - Combined DataFrame with cluster labels.
        """
        df1 = df1.copy()
        df2 = df2.copy()
        df1.rename(columns={ids[0]: 'id'}, inplace=True)
        df2.rename(columns={ids[1]: 'id'}, inplace=True)

        if not {'latitude', 'longitude'}.issubset(df1.columns) or not {'latitude', 'longitude'}.issubset(df2.columns):
            if {'int_latitude', 'int_longitude'}.issubset(df1.columns):
                df1.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            elif {'int_latitude', 'int_longitude'}.issubset(df2.columns):
                df2.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            else:
                raise ValueError("Input DataFrames must contain 'latitude' and 'longitude' columns.")

        combined_data = pd.concat([
            df1[['id', 'latitude', 'longitude']].assign(source=sources[0]),
            df2[['id', 'latitude', 'longitude']].assign(source=sources[1])
        ], ignore_index=True)

        latitudes = combined_data['latitude'].values
        longitudes = combined_data['longitude'].values

        # Calculate Haversine distances for the entire matrix
        distances = np.zeros((len(latitudes), len(latitudes)))
        for i in range(len(latitudes)):
            for j in range(len(latitudes)):
                distances[i, j] = ClusteringMatcher.haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

        # Perform HDBSCAN clustering
        clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distances)

        combined_data['cluster'] = labels

        return combined_data

    # DBSCAN CLUSTERING #
    @staticmethod
    def dbscan_clustering(df1: pd.DataFrame, df2: pd.DataFrame, ids: List[str], sources: List[str], eps: float, min_samples: int) -> pd.DataFrame:
        """
        Perform DBSCAN clustering once for both DataFrames.

        Parameters:
        - df1: First DataFrame containing ['id', 'latitude', 'longitude'].
        - df2: Second DataFrame containing ['id', 'latitude', 'longitude'].
        - eps: Maximum distance between two samples for them to be considered in the same neighborhood (in kilometers).
        - min_samples: The number of samples in a neighborhood for a point to be considered a core point.

        Returns:
        - Combined DataFrame with cluster labels.
        """
        df1 = df1.copy()
        df2 = df2.copy()
        df1.rename(columns={ids[0]: 'id'}, inplace=True)
        df2.rename(columns={ids[1]: 'id'}, inplace=True)

        # Ensure the DataFrames contain 'latitude' and 'longitude' columns        
        if not {'latitude', 'longitude'}.issubset(df1.columns) or not {'latitude', 'longitude'}.issubset(df2.columns):
            if {'int_latitude', 'int_longitude'}.issubset(df1.columns):
                df1.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            elif {'int_latitude', 'int_longitude'}.issubset(df2.columns):
                df2.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            else:
                raise ValueError("Input DataFrames must contain 'latitude' and 'longitude' columns.")
            
        # Ensure the DataFrames contain 'latitude' and 'longitude' columns
        combined_data = pd.concat([
            df1[['id', 'latitude', 'longitude']].assign(source=sources[0]),
            df2[['id', 'latitude', 'longitude']].assign(source=sources[1])
        ], ignore_index=True)

        latitudes = combined_data['latitude'].values
        longitudes = combined_data['longitude'].values

        distances = np.zeros((len(latitudes), len(latitudes)))
        for i in range(len(latitudes)):
            for j in range(len(latitudes)):
                distances[i, j] = ClusteringMatcher.haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distances)

        combined_data['cluster'] = labels

        return combined_data

    # COMPUTE DISTANCE MATRIX #
    @staticmethod
    def compute_distance_matrix(model_path: str, df1: pd.DataFrame, df2: pd.DataFrame, model_params: dict) -> torch.Tensor:
        """
        This function loads a pre-trained model, computes the distance matrix between two pandas DataFrames,
        and then finds the optimal matching using the Hungarian algorithm.

        Args:
            model_path (str): Path to the saved model file (e.g., 'model_epoch543.pth').
            df1 (pd.DataFrame): First DataFrame containing latitude and longitude columns.
            df2 (pd.DataFrame): Second DataFrame containing latitude and longitude columns.
            model_params (dict): Dictionary containing model parameters such as input_dim, hidden_dim, output_dim, and depth.

        Returns:
            (torch.Tensor, torch.Tensor): Indices of the optimal matching from df1 and df2.
        """
        
        # Unpack model parameters from the dictionary
        input_dim = model_params.get('input_dim', 2)
        hidden_dim = model_params.get('hidden_dim', 1024)
        output_dim = model_params.get('output_dim', 2)
        depth = model_params.get('depth', 5)
        
        # Step 1: Load the trained model with the unpacked parameters
        model = MyModel(input_dim, hidden_dim, output_dim, depth)
        
        # Load only the weights of the model
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode

        # Step 2: Prepare the data
        points1 = torch.tensor(df1[['latitude', 'longitude']].values, dtype=torch.float32)
        points2 = torch.tensor(df2[['latitude', 'longitude']].values, dtype=torch.float32)

        # Step 3: Run inference
        with torch.no_grad():  # Disable gradient calculations
            emb1 = model(points1)
            emb2 = model(points2)
 
        return torch.matmul(emb1, emb2.T) 

    # HUNGARIAN ALGORITHM #
    @staticmethod
    def hungarian(d : torch.Tensor) -> torch.Tensor:
        """
        Applies the Hungarian algorithm to the given distance matrix.

        Args:
            d (torch.Tensor): The distance matrix.

        Returns:
            torch.Tensor: Indices from the first and second sets that are optimally matched.
        """
        d_np = d.cpu().detach().numpy()  # Convert to NumPy array
        idx1, idx2 = linear_sum_assignment(d_np)  # Apply Hungarian algorithm
        return torch.tensor(idx1), torch.tensor(idx2)
    
## BASELINE MODELS ## 
    # NEAREST NEIGHBOR MATCHING #
    @staticmethod
    def NNM_clustering(clustered_data: pd.DataFrame, ids: List[str], sources: List[str]) -> pd.DataFrame:
        """
        Matches df1 and df2 data using nearest neighbor matching with pre-clustered data.

        Parameters:
        - clustered_data: DataFrame containing pre-clustered data from both sources.
        - ids (list): List containing the identifier column names for df1 and df2 respectively (e.g., ['mmsi', 'sar_id']).
        - sources (list): List containing the source names to assign for df1 and df2 respectively (e.g., ['ais', 'sar']).

        Returns:
        - pd.DataFrame: DataFrame containing the matched points and their distances, ensuring unique matches.
        """
        def lat_lon_to_cartesian(lat, lon):
            """
            Convert latitude and longitude to Cartesian coordinates.

            Args:
                lat (np.ndarray): Array of latitudes in radians.
                lon (np.ndarray): Array of longitudes in radians.

            Returns:
                np.ndarray: Cartesian coordinates as (x, y, z).
            """
            R = 6371.0  # Radius of the Earth in kilometers
            x = R * np.cos(lat) * np.cos(lon)
            y = R * np.cos(lat) * np.sin(lon)
            z = R * np.sin(lat)
            return np.vstack([x, y, z]).T

        # Initialize a list to store all matches
        all_matches = []

        # Get unique cluster labels
        unique_clusters = clustered_data['cluster'].unique()

        # Iterate over clusters
        for cluster in unique_clusters:
            if cluster == -1:
                # Skip noise points
                continue

            # Extract points in the current cluster for df1 and df2 separately
            cluster_df1 = clustered_data[(clustered_data['cluster'] == cluster) & (clustered_data['source'] == sources[0])]
            cluster_df2 = clustered_data[(clustered_data['cluster'] == cluster) & (clustered_data['source'] == sources[1])]

            # Ensure there are points from both df1 and df2 in the cluster
            if len(cluster_df1) > 0 and len(cluster_df2) > 0:
                # Convert latitude and longitude to radians for nearest neighbor search
                df1_cartesian = lat_lon_to_cartesian(
                    np.radians(cluster_df1['latitude'].values), np.radians(cluster_df1['longitude'].values)
                )
                df2_cartesian = lat_lon_to_cartesian(
                    np.radians(cluster_df2['latitude'].values), np.radians(cluster_df2['longitude'].values)
                )

                # Build KDTree for df2 points in the cluster
                tree = cKDTree(df2_cartesian)

                # Find nearest neighbors for each df1 point in the cluster
                _, indices = tree.query(df1_cartesian)

                # Create a DataFrame to store matches in the current cluster
                cluster_matches = pd.DataFrame({
                    ids[0]: cluster_df1['id'].values.astype(int),
                    'df1_lat': cluster_df1['latitude'].values,
                    'df1_lon': cluster_df1['longitude'].values,
                    ids[1]: cluster_df2.iloc[indices]['id'].values.astype(int),
                    'df2_lat': cluster_df2.iloc[indices]['latitude'].values,
                    'df2_lon': cluster_df2.iloc[indices]['longitude'].values,
                    'distance_km': ClusteringMatcher.haversine_distance(
                        cluster_df1['latitude'].values, cluster_df1['longitude'].values,
                        cluster_df2.iloc[indices]['latitude'].values, cluster_df2.iloc[indices]['longitude'].values
                    ),
                    'cluster': cluster,
                    'df2_index': cluster_df2.index[indices]  # Track df2 indices for unique matching
                })

                # Remove duplicate df2 matches to ensure unique matching within the cluster
                cluster_matches = cluster_matches.sort_values('distance_km').drop_duplicates(subset=['df2_index'], keep='first')
                # Remove rows where df1_id has multiple entries (df1 should have unique matches)
                cluster_matches = cluster_matches.drop_duplicates(subset=[ids[0]], keep='first')
                # Drop the helper 'df2_index' column
                cluster_matches = cluster_matches.drop(columns='df2_index')

                # Append cluster matches to the overall list of matches
                all_matches.append(cluster_matches)

        # Concatenate all cluster matches into a single DataFrame
        final_matches = pd.concat(all_matches, ignore_index=True)

        return final_matches

    # HUNGARIAN ALGORITHM #
    @staticmethod
    def HUM_clustering(clustered_data: pd.DataFrame, ids: List[str], sources: List[str]) -> pd.DataFrame:
        """
        Matches data from two sources within each cluster using the Hungarian algorithm.

        Parameters:
        - clustered_data: DataFrame containing pre-clustered data with cluster labels.
        - ids: List of identifier column names for df1 and df2 respectively (e.g., ['mmsi', 'sar_id']).
        - sources: List of source names for df1 and df2 respectively (e.g., ['ais', 'sar']).

        Returns:
        - A DataFrame containing the matched results.
        """
        all_matches = []
        unique_clusters = clustered_data['cluster'].unique()

        for cluster in unique_clusters:
            if cluster == -1:
                continue  # Skip noise points

            cluster_data = clustered_data[clustered_data['cluster'] == cluster]
            cluster_df1 = cluster_data[cluster_data['source'] == sources[0]]
            cluster_df2 = cluster_data[cluster_data['source'] == sources[1]]

            if len(cluster_df1) > 0 and len(cluster_df2) > 0:
                df1_coords = cluster_df1[['latitude', 'longitude']].to_numpy()
                df2_coords = cluster_df2[['latitude', 'longitude']].to_numpy()

                cost_matrix = ClusteringMatcher.haversine_distance(
                    df1_coords[:, 0][:, None], df1_coords[:, 1][:, None], 
                    df2_coords[:, 0][None, :], df2_coords[:, 1][None, :]
                )

                ship_indices, df2_indices = linear_sum_assignment(cost_matrix)

                for ship_idx, df2_idx in zip(ship_indices, df2_indices):
                    match = {
                        ids[0]: cluster_df1.iloc[ship_idx]['id'],
                        'df1_lat': cluster_df1.iloc[ship_idx]['latitude'],
                        'df1_lon': cluster_df1.iloc[ship_idx]['longitude'],
                        ids[1]: cluster_df2.iloc[df2_idx]['id'],
                        'df2_lat': cluster_df2.iloc[df2_idx]['latitude'],
                        'df2_lon': cluster_df2.iloc[df2_idx]['longitude'],
                        'distance_km': cost_matrix[ship_idx, df2_idx]
                    }
                    all_matches.append(match)

        return pd.DataFrame(all_matches)

## MACHINE LEARNING ##
    # MACHINE LEARNING DISTANCE MATRIX TO NEAREST NEIGHBOR MATCHING #
    @staticmethod
    def MNNM_clustering(clustered_data: pd.DataFrame, ids: List[str], sources: List[str], model_params: dict, model_path: str) -> pd.DataFrame:
        """
        Matches AIS and SAR data by applying nearest neighbor matching with the distance matrix from a pre-trained model.

        Parameters:
            clustered_data (pd.DataFrame): DataFrame containing pre-clustered data from both sources.
            ids (list): List of unique identifiers for df1 and df2. Example: ['mmsi', 'sar_id'].
            sources (list): List of sources for df1 and df2. Example: ['ais', 'sar'].
            model_path (str): Path to the model for generating the distance matrix.

        Returns:
            pd.DataFrame: DataFrame containing matched points between df1 and df2 using nearest neighbor matching.
        """

        # Get unique cluster labels from the pre-clustered data
        unique_clusters = clustered_data['cluster'].unique()

        # Initialize a list to store all matches across clusters
        all_matches = []

        # Loop over each cluster
        for cluster in unique_clusters:
            if cluster == -1:
                # Skip noise points (cluster label -1 in DBSCAN)
                continue

            # Extract points within the current cluster
            cluster_data = clustered_data[clustered_data['cluster'] == cluster]
            cluster_df1 = cluster_data[cluster_data['source'] == sources[0]]  # AIS points
            cluster_df2 = cluster_data[cluster_data['source'] == sources[1]]  # SAR points

            # Ensure there are both df1 and df2 points in the cluster
            if len(cluster_df1) > 0 and len(cluster_df2) > 0:
                # Extract coordinates for df1 and df2 in the current cluster
                df1_coords = cluster_df1[['latitude', 'longitude']]
                df2_coords = cluster_df2[['latitude', 'longitude']]

                # Compute the distance matrix using the pre-trained model
                distance_matrix = ClusteringMatcher.compute_distance_matrix(model_path=model_path, df1=df1_coords, df2=df2_coords, model_params=model_params)

                # Perform nearest neighbor matching
                nearest_neighbor_indices = torch.argmin(distance_matrix, dim=1).numpy()  # Get the index of the nearest neighbor for each df1 point

                # Create matches for the current cluster
                for i, nn_idx in enumerate(nearest_neighbor_indices):
                    # Extract match details from df1 and df2
                    df1_match = cluster_df1.iloc[i]
                    df2_match = cluster_df2.iloc[nn_idx]

                    # Create a match record as a dictionary
                    match = {
                        ids[0]: df1_match['id'],  # e.g., 'mmsi'
                        'df1_lat': df1_match['latitude'],
                        'df1_lon': df1_match['longitude'],
                        ids[1]: df2_match['id'],  # e.g., 'sar_id'
                        'df2_lat': df2_match['latitude'],
                        'df2_lon': df2_match['longitude'],
                        'distance_km': distance_matrix[i, nn_idx].item()  # Get the distance value
                    }
                    
                    # Append the match to the list of all matches
                    all_matches.append(match)

        # Convert the list of matches into a DataFrame
        all_matches = pd.DataFrame(all_matches)
        # Remove duplicate df2 matches to ensure unique matching within the cluster
        all_matches = all_matches.sort_values('distance_km').drop_duplicates(subset=[ids[1]], keep='first')
                # Remove rows where df1_id has multiple entries (df1 should have unique matches)
        all_matches = all_matches.drop_duplicates(subset=[ids[0]], keep='first')
                # Drop the helper 'df2_index' column
        #all_matches = all_matches.drop(columns=ids[1])
        return all_matches

    # MACHINE LEARNING DISTANCE MATRIX TO HUNGARIAN ALGORITHM #
    @staticmethod
    def MHUM_clustering(clustered_data: pd.DataFrame, ids: List[str], sources: List[str], model_params: dict, model_path: str) -> pd.DataFrame:
        """
        Matches AIS and SAR data by applying the Hungarian algorithm with pre-clustered data.

        Parameters:
            clustered_data (pd.DataFrame): DataFrame containing pre-clustered data from both sources.
            ids (list): List of unique identifiers for df1 and df2. Example: ['mmsi', 'sar_id'].
            sources (list): List of sources for df1 and df2. Example: ['ais', 'sar'].
            model_path (str): Path to the model for the Hungarian algorithm.

        Returns:
            pd.DataFrame: DataFrame containing matched points between df1 and df2.
        """
        
        # Get unique cluster labels from the pre-clustered data
        unique_clusters = clustered_data['cluster'].unique()

        # Initialize a list to store all matches across clusters
        all_matches = []

        # Loop over each cluster
        for cluster in unique_clusters:
            if cluster == -1:
                # Skip noise points (cluster label -1 in DBSCAN)
                continue

            # Extract points within the current cluster
            cluster_data = clustered_data[clustered_data['cluster'] == cluster]
            cluster_df1 = cluster_data[cluster_data['source'] == sources[0]]  # AIS points
            cluster_df2 = cluster_data[cluster_data['source'] == sources[1]]  # SAR points

            # Ensure there are both df1 and df2 points in the cluster
            if len(cluster_df1) > 0 and len(cluster_df2) > 0:
                # Extract coordinates for df1 and df2 in the current cluster
                df1_coords = cluster_df1[['latitude', 'longitude']]
                df2_coords = cluster_df2[['latitude', 'longitude']]

                # Apply Hungarian algorithm to find optimal matches
                distance_matrix = ClusteringMatcher.compute_distance_matrix(model_path=model_path, df1=df1_coords, df2=df2_coords, model_params=model_params)
                idx1, idx2 = ClusteringMatcher.hungarian(d = distance_matrix)

                # Convert the tensors to numpy arrays or lists for indexing pandas DataFrames
                ship_indices = idx1.numpy()
                df2_indices = idx2.numpy()

                # Create matches for the current cluster
                for ship_idx, df2_idx in zip(ship_indices, df2_indices):
                    # Extract match details from df1 and df2
                    df1_match = cluster_df1.iloc[ship_idx]
                    df2_match = cluster_df2.iloc[df2_idx]

                    # Extract relevant columns
                    df1_id = df1_match['id']  # e.g., 'mmsi'
                    df1_lat = df1_match['latitude']
                    df1_lon = df1_match['longitude']
                    df2_id = df2_match['id']  # e.g., 'sar_id'
                    df2_lat = df2_match['latitude']
                    df2_lon = df2_match['longitude']
                    distance = ClusteringMatcher.haversine_distance(
                        np.array([df1_lat]), np.array([df1_lon]),
                        np.array([df2_lat]), np.array([df2_lon]))
                    # Create a match record as a dictionary
                    match = {
                        ids[0]: df1_id,  # e.g., 'mmsi'
                        'df1_lat': df1_lat,
                        'df1_lon': df1_lon,
                        ids[1]: df2_id,  # e.g., 'sar_id'
                        'df2_lat': df2_lat,
                        'df2_lon': df2_lon,
                        'distance_km': distance[0]
                    }

                    # Append the match to the list of all matches
                    all_matches.append(match)

        # Convert the list of matches into a DataFrame
        return pd.DataFrame(all_matches)
    

