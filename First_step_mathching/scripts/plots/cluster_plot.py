# CLUSTERING PLOT #
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
## Plotting
import matplotlib.cm as cm
import matplotlib.pyplot as plt





class ClusterPlot:
    
    # Haversine distance function
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers (6371)
        return 6371 * c

    # Function to apply DBSCAN clustering to two dataframes
    @staticmethod
    def apply_dbscan_clustering(df1: pd.DataFrame, df2: pd.DataFrame, ids: list, sources: list, eps: float, min_samples: int) -> pd.DataFrame:
        """
        Applies DBSCAN clustering to two DataFrames based on their spatial coordinates.

        Parameters:
        - df1: First DataFrame containing data with columns ['id', 'latitude', 'longitude'].
        - df2: Second DataFrame containing data with columns ['id', 'latitude', 'longitude'].
        - ids: List of column names representing unique identifiers for df1 and df2.
        - sources: List of source identifiers for df1 and df2, e.g., ['ais', 'sar'].
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
                distances[i, j] = ClusterPlot.haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])

        # Convert the distance matrix to radians for DBSCAN with Haversine distance
        eps_rad = eps #/ 6371.0  # Convert eps from km to radians

        # Apply DBSCAN with precomputed distance matrix
        clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distances)

        # Assign cluster labels to the combined data
        combined_data['cluster'] = labels

        # Post-process clusters to ensure each contains at least one point from each source
        final_clusters = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                # Skip noise points
                continue

            cluster_points = combined_data[combined_data['cluster'] == cluster_id]
            if len(cluster_points['source'].unique()) == 2:  # Check if both sources are represented
                final_clusters.append(cluster_points)

        # Concatenate all valid clusters
        result_df = pd.concat(final_clusters) if final_clusters else pd.DataFrame()

        # Check if result_df is empty once
        if not result_df.empty:
            n_valid_clusters = result_df['cluster'].nunique()
            n_noise = np.sum(labels == -1)
        else:
            n_valid_clusters = 0
            n_noise = len(labels)

        print(f"Estimated number of valid clusters (with both sources): {n_valid_clusters}")
        print(f"Estimated number of noise points: {n_noise}")

        return result_df
    
    # Function to plot clusters with unique colors and noise points in black
    @staticmethod
    def plot_clusters(data: pd.DataFrame, title: str = "DBSCAN Clustering"):
        """
        Plots clustered data points with different colors for each cluster.
        Noise points are plotted in black with an alpha of 0.4.

        Parameters:
        - data: DataFrame with 'latitude', 'longitude', and 'cluster' columns.
        - title: Title of the plot.
        """
        # Get unique cluster labels
        unique_labels = set(data['cluster'])
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)  # Exclude noise from cluster count

        # Create a color map to assign unique colors for each cluster
        colors = cm.get_cmap('Spectral', n_clusters)  # Spectral colormap to support a large number of clusters

        plt.figure(figsize=(10, 8))

        for cluster in unique_labels:
            # Define a label and appearance for the cluster
            if cluster == -1:
                # Noise points (-1 cluster) are black with alpha 0.4
                label = 'Noise'
                color = 'black'
                alpha = 0.4
            else:
                # Get a unique color from the colormap for each cluster
                label = f'Cluster {cluster}'
                color = colors(cluster % n_clusters)  # Assign a unique color from the colormap
                alpha = 0.6

            # Filter data for the current cluster
            cluster_data = data[data['cluster'] == cluster]

            plt.scatter(
                cluster_data['longitude'], cluster_data['latitude'],
                s=100, label=label, alpha=alpha, color=color, edgecolors='w', linewidth=0.5
            )

        # Adding labels and title
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

    # Function to plot clusters with unique colors and noise points in black
    @staticmethod
    def plot_clusters2(data: pd.DataFrame, title: str = "DBSCAN Clustering"):
        """
        Plots clustered data points with different colors for each cluster.
        Noise points are plotted in black with an alpha of 0.4.

        Parameters:
        - data: DataFrame with 'latitude', 'longitude', and 'cluster' columns.
        - title: Title of the plot.
        """
        # Get unique cluster labels
        unique_labels = set(data['cluster'])

        # Create a color map to assign unique colors for each cluster
        colors = cm.get_cmap('tab10', len(unique_labels))  # Using 'tab10' colormap for up to 10 unique clusters

        plt.figure(figsize=(10, 8))

        for cluster in unique_labels:
            # Define a label and appearance for the cluster
            if cluster == -1:
                # Noise points (-1 cluster) are black with alpha 0.4
                label = 'Noise'
                color = 'black'
                alpha = 0.4
            else:
                # Get a unique color from the colormap for each cluster
                label = f'Cluster {cluster}'
                color = colors(cluster)  # Assign a unique color from the colormap
                alpha = 0.6

            # Filter data for the current cluster
            cluster_data = data[data['cluster'] == cluster]

            plt.scatter(
                cluster_data['longitude'], cluster_data['latitude'],
                s=100, label=label, alpha=alpha, color=color, edgecolors='w', linewidth=0.5
            )

        # Adding labels and title
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()