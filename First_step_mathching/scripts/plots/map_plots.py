import os
import random
import pandas as pd

from typing import List

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

import folium

class MapPlots:
## UTILITY FUNCTIONS ##
    # Function to generate a random color #
    def generate_random_color(self):
        """
        Generates a random color in hexadecimal format.

        Returns:
            str: A hex string representing a random color.
        """
        r = lambda: random.randint(0, 255)
        return '#{:02x}{:02x}{:02x}'.format(r(), r(), r())

## PLOTTING FUNCTIONS ##
    # Plotting funcion using cartopy #
    def plot_matched(self, df: pd.DataFrame, date: str, id_columns: List[str], 
                        df1_pos_columns: List[str], df2_pos_columns: List[str],
                        data_labels: List[str], model: str) -> None:
        """
        Plots matched AIS and SAR latitudes and longitudes on the same map using cartopy.
        Each match is displayed using the same randomly generated color.

        Args:
            df (pd.DataFrame): A DataFrame containing 'df1_lat', 'df1_lon' for AIS data and 'df2_lat', 'df2_lon' for SAR data.
            date (str): A string representing the date, used in the plot title.
            id_columns (List[str]): List of column names for identifying the AIS and SAR data.
            df1_pos_columns (List[str]): List of columns for AIS latitudes and longitudes.
            df2_pos_columns (List[str]): List of columns for SAR latitudes and longitudes.
        """
        # Calculate the number of matches
        num_matches = len(df)

        # Initialize the map
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Determine the extent for the map based on the latitude and longitude ranges
        all_lats = pd.concat([df[df1_pos_columns[0]], df[df2_pos_columns[0]]])
        all_lons = pd.concat([df[df1_pos_columns[1]], df[df2_pos_columns[1]]])

        # Set the extent (slightly extended) for the map
        buffer = 1 # latitude and longitude extension
        ax.set_extent([min(all_lons) - buffer, max(all_lons) + buffer, min(all_lats) - buffer, max(all_lats) + buffer])

        # Add features like coastlines and borders
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Plot AIS and SAR points
        ais_scatter = None
        sar_scatter = None
        for _, row in df.iterrows():
            color = self.generate_random_color()

            # Plot AIS point ('.' marker)
            ais_scatter = ax.scatter(row[df1_pos_columns[1]], row[df1_pos_columns[0]], 
                                    color=color, marker=".", label=id_columns[0], s=15, transform=ccrs.PlateCarree())
            
            # Plot SAR point ('x' marker)
            sar_scatter = ax.scatter(row[df2_pos_columns[1]], row[df2_pos_columns[0]], 
                                    color=color, marker="x", label=id_columns[1], s=15, transform=ccrs.PlateCarree())

        # Add gridlines and labels
        gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12, 'color': 'black'}
        gl.ylabel_style = {'size': 12, 'color': 'black'}

        # Add a legend with the number of matches as a title
        legend_title = f"Matches Found: {num_matches}"
        ax.legend([ais_scatter, sar_scatter], [data_labels[0], data_labels[1]], title=legend_title, loc='upper right')

        # Add a title
        plt.title(f'Matching {data_labels[0]} and {data_labels[1]} Data for {date} with {model}')

        # Show the plot
        plt.show()

    # Plotting funcion using folium #
    def folium_matched(self, df: pd.DataFrame, id_columns: List[str], df1_pos_columns: List[str], 
                       df2_pos_columns: List[str], folder_path : str, filename :str) -> folium.Map:
        """
        Creates an interactive Folium map with AIS and SAR points, where matched points are displayed in the same color.

        Args:
            df (pd.DataFrame): A DataFrame containing 'df1_lat', 'df1_lon' for AIS data and 'df2_lat', 'df2_lon' for SAR data.
            id_columns (List[str]): List of column names for identifying the AIS and SAR data.
            df1_pos_columns (List[str]): List of columns for AIS latitudes and longitudes.
            df2_pos_columns (List[str]): List of columns for SAR latitudes and longitudes.

        Returns:
            folium.Map: A Folium map object.
        """
        # hat
        file_path = f'{os.path.join(folder_path, filename)}.html'

        # Initialize the map at the mean latitude and longitude of both datasets
        center_lat = df[[df1_pos_columns[0], df2_pos_columns[0]]].mean().mean()
        center_lon = df[[df1_pos_columns[1], df2_pos_columns[1]]].mean().mean()
        folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        # Iterate through the DataFrame and plot each pair of AIS and SAR points with the same random color
        for _, row in df.iterrows():
            color = self.generate_random_color()

            # Add AIS point
            folium.Marker(
                icon=folium.DivIcon(html=f"""
                    <div style="width: 10px; height: 10px; background-color: {color}; transform: rotate(45deg);"></div>
                """),  # Injecting the color variable to create a square of that color
                location=[row[df1_pos_columns[0]], row[df1_pos_columns[1]]],
                tooltip=f"{id_columns[0]}: {row[id_columns[0]]} - Latitude: {row[df1_pos_columns[0]]}, Longitude: {row[df1_pos_columns[1]]}"
            ).add_to(folium_map)

            # Add SAR point
            folium.CircleMarker(
                location=[row[df2_pos_columns[0]], row[df2_pos_columns[1]]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                tooltip=f"{id_columns[1]}: {row[id_columns[1]]}- Latitude: {row[df2_pos_columns[0]]}, Longitude: {row[df2_pos_columns[1]]}"
            ).add_to(folium_map)

            # Optionally add lines to connect matched points
            folium.PolyLine(
                locations=[
                    [row[df1_pos_columns[0]], row[df1_pos_columns[1]]],
                    [row[df2_pos_columns[0]], row[df2_pos_columns[1]]]
                ],
                color=color,
                weight=2
            ).add_to(folium_map)
        
        folium_map.save(file_path)
        #return folium_map


