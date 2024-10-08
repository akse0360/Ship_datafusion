import random
import folium
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

# Function to generate random colors 
def generate_random_color():
        r = lambda: random.randint(0, 255)
        return '#{:02x}{:02x}{:02x}'.format(r(), r(), r())
# Function to map the normalized time difference to a color gradient
def generate_color_from_time_diff(time_diff, min_time_diff, max_time_diff):
    # Normalize the time difference to be between 0 and 1
    norm_time_diff = (time_diff - min_time_diff) / (max_time_diff - min_time_diff)

    # Get a color from the colormap (using matplotlib's 'viridis' colormap)
    cmap = plt.get_cmap('viridis')  # You can use other colormaps like 'coolwarm', 'plasma', etc.
    rgba_color = cmap(norm_time_diff)

    return matplotlib.colors.rgb2hex(rgba_color[:3])
class Plotter():
    # Function to create a folium map with ellipses from the 'UncertaintyEllipsePoints' column
    def norsat_plot_uncertainty_ellipses(norsat_data, date_key, zoom_start=4):
        """
        Plots uncertainty ellipses from the 'UncertaintyEllipsePoints' column in the norsat_data[date_key] on a folium map.
        Use: map_object = pl.norsat_plot_uncertainty_ellipses(norsat_data = norsat_data, date_key = date_key)

        Parameters:
        - norsat_data: DataFrame containing the 'UncertaintyEllipsePoints' column with ellipse points.
        - date_key: The specific date key in norsat_data.
        - zoom_start: Initial zoom level for the map (default is 4).
        
        Returns:
        - A folium map object with ellipses plotted from the 'UncertaintyEllipsePoints' column.
        """
        
        map_center = [
        norsat_data[date_key]['NRDEmitterPosition'].apply(lambda x: x['Latitude']).mean(),  # Mean latitude
        norsat_data[date_key]['NRDEmitterPosition'].apply(lambda x: x['Longitude']).mean()  # Mean longitude
        ]
        

        # Create a base map using Folium
        m = folium.Map(location = map_center, zoom_start = zoom_start)
        
        # Loop through the rows of your data and plot ellipses from the 'UncertaintyEllipsePoints' column
        for _, row in norsat_data[date_key].iterrows():
            ellipse_points = row['UncertaintyEllipsePoints']
            
            # Generate a random color for each ellipse
            color = generate_random_color()
            
            # Create a folium Polygon for the ellipse and add it to the map
            polygon = folium.Polygon(
                locations=ellipse_points,
                color=color,
                fill=True,
                fill_opacity=0.4
            ).add_to(m)
            
            # Create a popup object and add it to the polygon
            popup_content = f"Collection: {row['CandidateList']}"
            popup = folium.Popup(popup_content, max_width=300)
            
            # Bind the popup to the polygon
            polygon.add_child(popup)
        # Return the folium map object
        return m
    
    def unified_plot(ais_mmsi = None, sar_data = None, norsat_data = None, interpolated_ais = None, date_key = None, zoom_start = 4):
        # Initialize variables to hold latitude and longitude
        center_latitudes = []
        center_longitudes = []
        
        # Check which datasets are not None and extract lat/lon
        #if ais_data is not None:
        #    center_latitudes.extend(ais_data[data_key]['latitude'])  # Assuming ais_data has 'latitude' key
        #    center_longitudes.extend(ais_data[data_key]['longitude'])  # Assuming ais_data has 'longitude' key
        if sar_data is not None:
            center_latitudes.extend(sar_data['latitude'])  # Assuming SAR_data has 'latitude' key
            center_longitudes.extend(sar_data['longitude'])  # Assuming SAR_data has 'longitude' key
        if norsat_data is not None:
            center_latitudes.extend(norsat_data[date_key]['latitude'])  # Assuming norsat_data has 'latitude' key
            center_longitudes.extend(norsat_data[date_key]['longitude'])  # Assuming norsat_data has 'longitude' key

        # Calculate the map center if we have any lat/lon data
        if center_latitudes and center_longitudes:
            map_center = [
                sum(center_latitudes) / len(center_latitudes),
                sum(center_longitudes) / len(center_longitudes)
            ]
        else:
            map_center = [0, 0]  # Default value if no data is available

        m = folium.Map(location = map_center, zoom_start = zoom_start)

        if interpolated_ais is not None: # Add markers for each AIS/SAR point
            for mmsi, coords in interpolated_ais.items():
                color_ais = generate_random_color()
                folium.Marker(
                    location=[coords['y'], coords['x']],  # Latitude first, then Longitude
                    #radius=1.5,
                    icon=folium.Icon(icon="times", prefix="fa", color="orange"),  # "times" gives an "X" icon
                    #color=color_ais,
                    #fill=True,
                    #fill_opacity=0.7,
                    popup=f"MMSI: {mmsi}",
                    #icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
                


        if ais_mmsi is not None:# and not ais_mmsi.empty:
            mmsi_numbers = list(ais_mmsi.indices.keys())
            # Get all the timestamps for normalization
            all_timestamps = []
            for mmsi in mmsi_numbers:
                tes = ais_mmsi.get_group((mmsi,))
                all_timestamps.extend(tes['TimeStamp'].tolist())

            # Convert to pandas datetime to easily work with timestamps
            all_timestamps = pd.to_datetime(all_timestamps)

            # Get the min and max timestamp to normalize time differences
            min_timestamp = all_timestamps.min()
            max_timestamp = all_timestamps.max()
            
            # Loop through the data to plot each point with color based on time difference
            for mmsi in mmsi_numbers:
                tes = ais_mmsi.get_group((mmsi,))
                for _, row in tes.iterrows():
                    # Calculate the time difference between this row's timestamp and the min timestamp
                    timestamp = pd.to_datetime(row['TimeStamp'])
                    time_diff = (timestamp - min_timestamp).total_seconds()

                    # Generate a color based on the time difference
                    color_ais = generate_color_from_time_diff(time_diff, 0, (max_timestamp - min_timestamp).total_seconds())

                    # Add a marker with the generated color
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=1.5,
                        color=color_ais,
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"AIS MMSI: {row['mmsi']}<br>Timestamp: {row['TimeStamp']}"
                    ).add_to(m)

            #for mmsi in mmsi_numbers:
             #   tes = ais_mmsi.get_group((mmsi,))  # Pass key as a tuple
              #  for _, row in tes.iterrows():
               #         color_ais = generate_random_color()
                #        folium.CircleMarker(
                 #           location=[row['latitude'], row['longitude']],
                  #          radius=1.5,
                   #         color=color_ais,
                    ##       fill_opacity=0.7,
                      #      popup=f"AIS MMSI: {row['mmsi']}"
                       # ).add_to(m)



        # Plot SAR points if the DataFrame is not None and not empty
        if sar_data is not None and not sar_data.empty:
            for _, row in sar_data.iterrows():
                color_sar = generate_random_color()
                folium.RegularPolygonMarker(
                    location=[row['latitude'], row['longitude']],
                    number_of_sides=4,
                    radius=2,
                    rotation=45,
                    color=color_sar,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"SAR Object ID: {row['Object_ID']}"
                ).add_to(m)

        # Plot Norsat points if the DataFrame is not None and not empty
        if norsat_data is not None:
             # Loop through the rows of your data and plot ellipses from the 'UncertaintyEllipsePoints' column
            for _, row in norsat_data[date_key].iterrows():
                ellipse_points = row['UncertaintyEllipsePoints']
                
                # Generate a random color for each ellipse
                color = generate_random_color()
                
                # Create a folium Polygon for the ellipse and add it to the map
                polygon = folium.Polygon(
                    locations=ellipse_points,
                    color=color,
                    fill=True,
                    fill_opacity=0.4
                ).add_to(m)
                
                # Create a popup object and add it to the polygon
                popup_content = f"Collection: {row['CandidateList']}"
                popup = folium.Popup(popup_content, max_width=300)
                
                # Bind the popup to the polygon
                polygon.add_child(popup)
        
        return m
    
    def plot_matches_on_folium(df):
        # Determine if the DataFrame is for SAR or Norsat based on column names
        if 'sar_id' in df.columns:
            lat_col = 'sar_lat'
            lon_col = 'sar_lon'
            id_col = 'sar_id'
            distance_col = 'sar_distance_km'
            target_type = 'SAR'

        elif 'norsat_id' in df.columns:
            lat_col = 'norsat_lat'
            lon_col = 'norsat_lon'
            id_col = 'norsat_id'
            distance_col = 'norsat_distance_km'
            target_type = 'Norsat'
        else:
            raise ValueError("DataFrame doesn't contain SAR or Norsat columns")

        # Create a base map centered around the mean of the latitudes and longitudes
        center_lat = df[['ais_lat', lat_col]].mean().mean()
        center_lon = df[['ais_lon', lon_col]].mean().mean()
        
        # Initialize the Folium map
        base_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        # Iterate over the rows of the dataframe to add markers and lines
        for _, row in df.iterrows():
            # Add a marker for the AIS location
            folium.Marker(
                location=[row['ais_lat'], row['ais_lon']],
                popup=f"AIS, MMSI: {int(row['mmsi'])}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(base_map)
            
            # Add a marker for the target location (SAR or Norsat)
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=f"{target_type}, ID: {row[id_col]} ({target_type})",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(base_map)
            
            # Draw a dotted line connecting the AIS and target locations
            folium.PolyLine(
                locations=[[row['ais_lat'], row['ais_lon']], [row[lat_col], row[lon_col]]],
                color='green',
                weight=2,
                dash_array='5, 5' , # This makes the line dotted
                popup=f"Distance: {row[distance_col]}"
            ).add_to(base_map)
        
        # Return the map object
        return base_map
    
    def plot_triple_matches_on_folium(df):
        """
        Visualizes the positions of vessels from AIS data, Norsat data, and SAR data on a Folium map. 
        The function creates markers for each vessel's location and connects them with lines to illustrate relationships.

        Args:
            df (DataFrame): A pandas DataFrame containing vessel data with latitude and longitude columns for AIS, Norsat, and SAR.

        Returns:
            folium.Map: A Folium map object displaying the vessel locations and connections.

        Examples:
            >>> plot_triple_matches_on_folium(vessel_data)
        """
        # Create a base map centered around the mean of the latitudes and longitudes
        center_lat = df[['ais_lat_x', 'ais_lat_y', 'norsat_lat', 'sar_lat']].mean().mean()
        center_lon = df[['ais_lon_x', 'ais_lon_y', 'norsat_lon', 'sar_lon']].mean().mean()

        # Initialize the Folium map
        base_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)

        # Iterate over the rows of the dataframe to add markers and lines
        for _, row in df.iterrows():
            # Add a marker for the first AIS location (ais_lat_x, ais_lon_x)
            folium.Marker(
                location=[row['ais_lat_x'], row['ais_lon_x']],
                popup=f"MMSI: {row['mmsi']} (AIS X)",
                icon=folium.Icon(color='blue', icon='tint')  # Use ship icon for AIS
            ).add_to(base_map)
            
            # Add a marker for the second AIS location (ais_lat_y, ais_lon_y)
            folium.Marker(
                location=[row['ais_lat_y'], row['ais_lon_y']],
                popup=f"MMSI: {row['mmsi']} (AIS Y)",
                icon=folium.Icon(color='blue', icon='tint')  # Use ship icon for AIS
            ).add_to(base_map)
            
            # Add a marker for the Norsat location (norsat_lat, norsat_lon)
            folium.Marker(
                location=[row['norsat_lat'], row['norsat_lon']],
                popup=f"Norsat ID: {row['norsat_id']} (Norsat)",
                icon=folium.Icon(color='green', icon='cloud')  # Use globe icon for Norsat
            ).add_to(base_map)
            
            # Add a marker for the SAR location (sar_lat, sar_lon)
            folium.Marker(
                location=[row['sar_lat'], row['sar_lon']],
                popup=f"SAR ID: {row['sar_id']} (SAR)",
                icon=folium.Icon(color='red', icon='globe')  # Use cloud icon for SAR
            ).add_to(base_map)

            # Lines to connect the points (same as previous example)
            folium.PolyLine(
                locations=[[row['ais_lat_x'], row['ais_lon_x']], [row['norsat_lat'], row['norsat_lon']]],
                color='blue',
                weight=2,
                dash_array='5, 5',  # Dotted line for AIS X to Norsat
                popup=f"Distance: {row['norsat_distance_km']}"
            ).add_to(base_map)
            
            folium.PolyLine(
                locations=[[row['ais_lat_y'], row['ais_lon_y']], [row['sar_lat'], row['sar_lon']]],
                color='blue',
                weight=2,
                dash_array='5, 5',  # Dotted line for AIS Y to SAR
                popup=f"Distance: {row['sar_distance_km']}"
            ).add_to(base_map)

            # Connect AIS X and AIS Y
            folium.PolyLine(
                locations=[[row['ais_lat_x'], row['ais_lon_x']], [row['ais_lat_y'], row['ais_lon_y']]],
                color='orange',
                weight=2,
                dash_array='5, 5'  # Dotted line for AIS X to AIS Y
            ).add_to(base_map)

        # Return the map object
        return base_map