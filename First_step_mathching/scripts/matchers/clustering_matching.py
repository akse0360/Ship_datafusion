import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from typing import List 

# MLP MATCHING ALGORITHM #
import torch
from mlp import MyModel 

class ClusteringMatcher:
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
    
    # Function to apply DBSCAN clustering to two dataframes
    @staticmethod
    def apply_dbscan_clustering(df1: pd.DataFrame, df2: pd.DataFrame, ids : List[str], sources : List[str] ,eps: float, min_samples: int) -> pd.DataFrame:
        """
        Applies DBSCAN clustering to two DataFrames based on their spatial coordinates.
        
        Parameters:
        - df1: First DataFrame containing data with columns ['id', 'latitude', 'longitude'].
        - df2: Second DataFrame containing data with columns ['id', 'latitude', 'longitude'].
        - eps: Maximum distance between two samples for them to be considered as in the same neighborhood (in kilometers).
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

        # Combine the two DataFrames, adding a source column to differentiate them
        combined_data = pd.concat([
            df1[['id', 'latitude', 'longitude']].assign(source=sources[0]),
            df2[['id', 'latitude', 'longitude']].assign(source=sources[1])
        ], ignore_index=True)

        # Extract the coordinates for clustering
        latitudes = combined_data['latitude'].values
        longitudes = combined_data['longitude'].values

        # Create the Haversine distance matrix between all points
        distances = np.zeros((len(latitudes), len(latitudes)))
        for i in range(len(latitudes)):
            for j in range(len(latitudes)):
                distances[i, j] = ClusteringMatcher.haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

        # Convert the distance matrix to radians for DBSCAN with Haversine distance
        eps_rad = eps 

        # Apply DBSCAN with precomputed distance matrix
        clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distances)

        # Assign cluster labels to the combined data
        combined_data['cluster'] = labels
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
            
        return combined_data
        
    @staticmethod
    def HAM_clustering(df1: pd.DataFrame, df2: pd.DataFrame, ids: List[str], sources: List[str], eps: float = 0.5, min_samples: int = 1) -> pd.DataFrame:
        """
        Matches df1 and df2 data within each cluster using the Hungarian algorithm.

        Parameters:
        - df1: First DataFrame containing [id1_col, 'latitude', 'longitude'] and other necessary columns.
        - df2: Second DataFrame containing [id2_col, 'latitude', 'longitude'] and other necessary columns.
        - ids: List of identifier column names for df1 and df2 respectively (e.g., ['mmsi', 'sar_id']).
        - sources: List of source names to assign for df1 and df2 respectively (e.g., ['ais', 'sar']).
        - eps: Maximum distance between two samples for them to be considered as in the same neighborhood (DBSCAN parameter).
        - min_samples: The number of samples in a neighborhood for a point to be considered a core point (DBSCAN parameter).

        Returns:
        - A DataFrame containing the matched results.
        """        

        df1 = df1.copy()
        df2 = df2.copy()
        
        # Ensure the DataFrames contain 'latitude' and 'longitude' columns        
        if not {'latitude', 'longitude'}.issubset(df1.columns) or not {'latitude', 'longitude'}.issubset(df2.columns):
            if {'int_latitude', 'int_longitude'}.issubset(df1.columns):
                df1.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            elif {'int_latitude', 'int_longitude'}.issubset(df2.columns):
                df2.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            else:
                raise ValueError("Input DataFrames must contain 'latitude' and 'longitude' columns.")

        # Apply DBSCAN clustering
        combined_data = ClusteringMatcher.apply_dbscan_clustering(df1, df2, ids, sources, eps, min_samples)

        # Initialize a list to store all matches
        all_matches = []

        # Get unique cluster labels
        unique_clusters = combined_data['cluster'].unique()

        for cluster in unique_clusters:
            if cluster == -1:
                # Skip noise points (cluster label -1 in DBSCAN)
                continue

            # Extract points within the current cluster
            cluster_data = combined_data[combined_data['cluster'] == cluster]
            cluster_df1 = cluster_data[cluster_data['source'] == sources[0]]
            cluster_df2 = cluster_data[cluster_data['source'] == sources[1]]

            # Ensure there are both df1 and df2 points in the cluster
            if len(cluster_df1) > 0 and len(cluster_df2) > 0:
                # Extract coordinates for df1 and df2 in the current cluster
                df1_coords = cluster_df1[['latitude', 'longitude']].to_numpy()
                df2_coords = cluster_df2[['latitude', 'longitude']].to_numpy()

                # Create cost matrix using Haversine distance
                cost_matrix = ClusteringMatcher.haversine_distance(
                    df1_coords[:, 0][:, None], df1_coords[:, 1][:, None], 
                    df2_coords[:, 0][None, :], df2_coords[:, 1][None, :]
                )

                # Apply Hungarian algorithm
                ship_indices, df2_indices = linear_sum_assignment(cost_matrix)

                # Create matches for the current cluster
                for ship_idx, df2_idx in zip(ship_indices, df2_indices):
                    # Extract match details
                    df1_id = cluster_df1.iloc[ship_idx]['id']
                    df1_lat = cluster_df1.iloc[ship_idx]['latitude']
                    df1_lon = cluster_df1.iloc[ship_idx]['longitude']
                    df2_id = cluster_df2.iloc[df2_idx]['id']
                    df2_lat = cluster_df2.iloc[df2_idx]['latitude']
                    df2_lon = cluster_df2.iloc[df2_idx]['longitude']
                    distance_km = cost_matrix[ship_idx, df2_idx]

                    # Create a match record with the required columns
                    match = {
                        ids[0]: df1_id.astype(int),
                        'df1_lat': df1_lat,
                        'df1_lon': df1_lon,
                        ids[1]: df2_id.astype(int),
                        'df2_lat': df2_lat,
                        'df2_lon': df2_lon,
                        'distance_km': distance_km
                    }
                    all_matches.append(match)

        # Convert all matches to a DataFrame
        return pd.DataFrame(all_matches)

    @staticmethod
    def NNM_clustering(df1: pd.DataFrame, df2: pd.DataFrame, ids: list, sources: list, eps: float = 0.5, min_samples: int = 1) -> pd.DataFrame:
        """
        Matches df1 and df2 data using nearest neighbor matching after applying DBSCAN clustering.

        Parameters:
        - df1 (pd.DataFrame): First DataFrame containing [ids[0], 'latitude', 'longitude'] and other necessary columns.
        - df2 (pd.DataFrame): Second DataFrame containing [ids[1], 'latitude', 'longitude'] and other necessary columns.
        - ids (list): List containing the identifier column names for df1 and df2 respectively (e.g., ['mmsi', 'sar_id']).
        - sources (list): List containing the source names to assign for df1 and df2 respectively (e.g., ['ais', 'sar']).
        - eps (float): Maximum distance between two samples for them to be considered in the same neighborhood (DBSCAN parameter).
        - min_samples (int): The number of samples in a neighborhood for a point to be considered a core point (DBSCAN parameter).

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

        # Create copies of the dataframes to avoid modifying the original DataFrames
        df1 = df1.copy()
        df2 = df2.copy()
        # Ensure the DataFrames contain 'latitude' and 'longitude' columns        
        if not {'latitude', 'longitude'}.issubset(df1.columns) or not {'latitude', 'longitude'}.issubset(df2.columns):
            if {'int_latitude', 'int_longitude'}.issubset(df1.columns):
                df1.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            elif {'int_latitude', 'int_longitude'}.issubset(df2.columns):
                df2.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            else:
                raise ValueError("Input DataFrames must contain 'latitude' and 'longitude' columns.")

        # Assign source labels
        df1.loc[:, 'source'] = sources[0]
        df2.loc[:, 'source'] = sources[1]

        # Apply DBSCAN clustering
        combined_df = ClusteringMatcher.apply_dbscan_clustering(df1, df2, ids, sources, eps, min_samples)

        # Initialize a list to store all matches
        all_matches = []

        # Get unique cluster labels
        unique_clusters = combined_df['cluster'].unique()

        for cluster in unique_clusters:
            if cluster == -1:
                continue

            # Extract points in the current cluster for df1 and df2 separately
            cluster_df1 = combined_df[(combined_df['cluster'] == cluster) & (combined_df['source'] == sources[0])]
            cluster_df2 = combined_df[(combined_df['cluster'] == cluster) & (combined_df['source'] == sources[1])]

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

    @staticmethod
    def hungarian(d):
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
    @staticmethod
    def compute_distance_matrix_with_hungarian(model_path, df1, df2, input_dim=2, hidden_dim=256, output_dim=2, depth=5) -> torch.Tensor:
        """
        This function loads a pre-trained model, computes the distance matrix between two pandas DataFrames,
        and then finds the optimal matching using the Hungarian algorithm.

        Args:
            model_path (str): Path to the saved model file (e.g., 'model_epoch543.pth').
            df1 (pd.DataFrame): First DataFrame containing latitude and longitude columns.
            df2 (pd.DataFrame): Second DataFrame containing latitude and longitude columns.
            input_dim (int): Dimension of the input features (default: 2 for latitude and longitude).
            hidden_dim (int): Hidden dimension size of the model (default: 256).
            output_dim (int): Output dimension size of the model embeddings (default: 2).
            depth (int): Number of layers in the model (default: 5).

        Returns:
            (torch.Tensor, torch.Tensor): Indices of the optimal matching from df1 and df2.
        """

        # Step 1: Load the trained model
        model = MyModel(input_dim, hidden_dim, output_dim, depth)
        #model.load_state_dict(torch.load(model_path))

        # Load only the weights of the model (safer and future-proof)
        state_dict = torch.load(model_path, weights_only=True)

    #    Load the state dict into your model
        model.load_state_dict(state_dict)
        model.eval()  # Set the model to evaluation mode

        # Step 2: Prepare the data
        points1 = torch.tensor(df1[['latitude', 'longitude']].values, dtype=torch.float32)
        points2 = torch.tensor(df2[['latitude', 'longitude']].values, dtype=torch.float32)

        # Step 3: Run inference
        with torch.no_grad():  # Disable gradient calculations
            emb1 = model(points1)
            emb2 = model(points2)

        # Step 4: Compute the distance (or similarity) matrix
        distance_matrix = torch.matmul(emb1, emb2.T)

        # Step 5: Apply the Hungarian algorithm to find the optimal matching
        idx1, idx2 = ClusteringMatcher.hungarian(distance_matrix)

        return idx1, idx2
    
    @staticmethod
    def match_clusters_with_hungarian(df1, df2, ids, sources, eps=7, min_samples=2, model_path = r'C:\Users\abelt\OneDrive\Dokumenter\GitHub\Ship_datafusion\models\model_epoch.pth'):
        """
        Matches AIS and SAR data by applying DBSCAN clustering followed by the Hungarian algorithm.
        
        Parameters:
        - df1: DataFrame containing the AIS points.
        - df2: DataFrame containing the SAR points.
        - ids: List of unique identifiers for df1 and df2. Example: ['mmsi', 'sar_id'].
        - sources: List of sources for df1 and df2. Example: ['ais', 'sar'].
        - eps: The maximum distance between two samples for them to be considered as in the same neighborhood (for DBSCAN).
        - min_samples: The number of points required to form a cluster (for DBSCAN).
        - model_path: Path to the model for the Hungarian algorithm (optional).
        
        Returns:
        - matching_df: DataFrame containing matched points between df1 and df2.
        """

        # Ensure the DataFrames contain 'latitude' and 'longitude' columns        
        if not {'latitude', 'longitude'}.issubset(df1.columns) or not {'latitude', 'longitude'}.issubset(df2.columns):
            if {'int_latitude', 'int_longitude'}.issubset(df1.columns):
                df1.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            elif {'int_latitude', 'int_longitude'}.issubset(df2.columns):
                df2.rename(columns={'int_latitude': 'latitude', 'int_longitude': 'longitude'}, inplace=True)
            else:
                raise ValueError("Input DataFrames must contain 'latitude' and 'longitude' columns.")
        
        # Apply DBSCAN clustering (using a placeholder function)
        clustered_df = ClusteringMatcher.apply_dbscan_clustering(df1, df2, ids=ids, sources=sources, eps=eps, min_samples=min_samples)

        # Get unique cluster labels
        unique_clusters = clustered_df['cluster'].unique()

        # Initialize a list to store all matches across clusters
        all_matches = []

        for cluster in unique_clusters:
            if cluster == -1:
                # Skip noise points (cluster label -1 in DBSCAN)
                continue

            # Extract points within the current cluster
            cluster_data = clustered_df[clustered_df['cluster'] == cluster]
            cluster_df1 = cluster_data[cluster_data['source'] == sources[0]]  # AIS points
            cluster_df2 = cluster_data[cluster_data['source'] == sources[1]]  # SAR points

            # Ensure there are both df1 and df2 points in the cluster
            if len(cluster_df1) > 0 and len(cluster_df2) > 0:
                # Extract coordinates for df1 and df2 in the current cluster
                df1_coords = cluster_df1[['latitude', 'longitude']]
                df2_coords = cluster_df2[['latitude', 'longitude']]
                
                # Apply Hungarian algorithm to find optimal matches
                idx1, idx2 = ClusteringMatcher.compute_distance_matrix_with_hungarian(model_path, df1_coords, df2_coords)
                
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
                    
                    # You can also add distance or other matching criteria if you have them
                    # distance_km = cost_matrix[ship_idx, df2_idx]

                    # Create a match record as a dictionary
                    match = {
                        ids[0]: df1_id,  # e.g., 'mmsi'
                        'df1_lat': df1_lat,
                        'df1_lon': df1_lon,
                        ids[1]: df2_id,  # e.g., 'sar_id'
                        'df2_lat': df2_lat,
                        'df2_lon': df2_lon,
                        # Uncomment this if you calculate distance
                        # 'distance_km': distance_km
                    }

                    # Append the match to the list of all matches
                    all_matches.append(match)

        # Convert the list of matches into a DataFrame
        return pd.DataFrame(all_matches)
