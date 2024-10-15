

class MapPlots:
    
    pass

### PLOT OG SPLINE ###
def plot_splines_ais_sar_on_map(splines=None, ais_mmsi=None, sar_data=None, norsat_data=None, ais_sar_pol = None):
    # Create a base map centered around the mean of all AIS points
    map_center = [
        sum(value['y'].mean() for _, value in splines.items()) / len(splines),
        sum(value['x'].mean() for _, value in splines.items()) / len(splines),
    ]

    m = folium.Map(location=map_center, zoom_start=6)

    # Plot splines
    #if splines is not None:
     #   for key, value in splines.items():
      #      spline_coords = list(zip(value['y'], value['x']))
       #     folium.PolyLine(spline_coords, color='blue', weight=2.5, opacity=1).add_to(m)

            
    # Extract corresponding AIS data
    if ais_mmsi is not None and not ais_mmsi.empty:
        tes = ais_mmsi.get_group((key,))  # Pass key as a tuple
        for _, row in tes.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=1.5,
                    color='red',
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"AIS MMSI: {row['mmsi']}"
                ).add_to(m)

    if ais_sar_pol is not None: # Add markers for each AIS/SAR point
        for mmsi, coords in ais_sar_pol.items():
            folium.CircleMarker(
				location=[coords['y'], coords['x']],  # Latitude first, then Longitude
                radius=1.5,
                color='blue',
                fill=True,
                fill_opacity=0.7,
				popup=f"MMSI: {mmsi}",
				#icon=folium.Icon(color='blue', icon='info-sign')
			).add_to(m)
			

    # Plot SAR points if the DataFrame is not None and not empty
    if sar_data is not None and not sar_data.empty:
        for _, row in sar_data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=2,
                color='green',
                fill=True,
                fill_opacity=0.7,
                popup=f"SAR Object ID: {row['Object_ID']}"
            ).add_to(m)

    # Plot Norsat points if the DataFrame is not None and not empty
    if norsat_data is not None and not norsat_data.empty:
        for idx, row in norsat_data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color='black',
                fill=True,
                fill_opacity=0.7,
                popup=f"idx: {idx}"
            ).add_to(m)

    return m

# Call the function
#m = plot_splines_ais_sar_on_map(splines = xy_interpolated, ais_mmsi = ais_mmsi, sar_data = filter_df_sar_0, norsat_data = filter_dfs['Norsat'])
m = plot_splines_ais_sar_on_map(splines = xy_interpolated, ais_mmsi = None, sar_data = filter_df_sar_0, norsat_data = None, ais_sar_pol = ais_sar_pol)
m.save(f'./images/splines_ais_sar_map_{date_key}_on_land_false.html')  # Save to an HTML file if needed

# Call the function
m = plot_splines_ais_sar_on_map(splines = xy_interpolated, ais_mmsi = None, sar_data = filter_df_sar_1, norsat_data = None, ais_sar_pol = ais_sar_pol)
m.save(f'./images/splines_ais_sar_map_{date_key}_on_land_true.html')  # Save to an HTML file if needed


# Triple match # 
def plot_triple_matches_on_cartopy(df):
    """
    Visualizes the positions of vessels from AIS data, Norsat data, and SAR data on a Cartopy map. 
    The function creates markers for each vessel's location and connects them with lines to illustrate relationships.

    Args:
        df (DataFrame): A pandas DataFrame containing vessel data with latitude and longitude columns for AIS, Norsat, and SAR.

    Returns:
        None: Displays the map with vessel locations and connections.
    """

    num_matches = len(df)

    # Calculate the extent dynamically based on latitude and longitude columns
    min_lon = df[['ais_lon_x', 'ais_lon_y', 'norsat_lon', 'sar_lon']].min().min()
    max_lon = df[['ais_lon_x', 'ais_lon_y', 'norsat_lon', 'sar_lon']].max().max()
    min_lat = df[['ais_lat_x', 'ais_lat_y', 'norsat_lat', 'sar_lat']].min().min()
    max_lat = df[['ais_lat_x', 'ais_lat_y', 'norsat_lat', 'sar_lat']].max().max()

    # Create a new figure with the European Albers Equal Area projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.AlbersEqualArea(central_longitude=10, central_latitude=52, 
                                                                   standard_parallels=(43, 62)))

    # Set the extent of the map dynamically with a small buffer for better visibility
    ax.set_extent([min_lon - 0.5, max_lon + 0.5, min_lat - 0.5, max_lat + 0.5], crs=ccrs.PlateCarree())
    # Add latitude and longitude labels
    add_lat_lon_labels(ax)
    
    # Add coastlines and other geographical features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='lightgray')
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAKES, color='lightblue')
    ax.add_feature(cfeature.RIVERS)

    # Iterate over the rows of the dataframe to plot markers and lines
    for _, row in df.iterrows():
        # Plot AIS X location (ais_lat_x, ais_lon_x)
        ax.plot(row['ais_lon_x'], row['ais_lat_x'], marker='o', color='blue', label='AIS X' if _ == 0 else "", transform=ccrs.PlateCarree())

        # Plot AIS Y location (ais_lat_y, ais_lon_y)
        ax.plot(row['ais_lon_y'], row['ais_lat_y'], marker='o', color='blue', label='AIS Y' if _ == 0 else "", transform=ccrs.PlateCarree())

        # Plot Norsat location (norsat_lat, norsat_lon)
        ax.plot(row['norsat_lon'], row['norsat_lat'], marker='d', color='red', label='Norsat' if _ == 0 else "", transform=ccrs.PlateCarree())

        # Plot SAR location (sar_lat, sar_lon)
        ax.plot(row['sar_lon'], row['sar_lat'], marker='X', color='green', label='SAR' if _ == 0 else "", transform=ccrs.PlateCarree())

        # Draw lines between AIS X and Norsat
        ax.plot([row['ais_lon_x'], row['norsat_lon']], [row['ais_lat_x'], row['norsat_lat']], color='blue', linestyle='--', transform=ccrs.PlateCarree())
        # Draw lines between AIS Y and SAR
        ax.plot([row['ais_lon_y'], row['sar_lon']], [row['ais_lat_y'], row['sar_lat']], color='blue', linestyle='--', transform=ccrs.PlateCarree())
        # Draw lines connecting AIS X and AIS Y
        ax.plot([row['ais_lon_x'], row['ais_lon_y']], [row['ais_lat_x'], row['ais_lat_y']], color='orange', linestyle='--', transform=ccrs.PlateCarree())

    # Add legend to the map
    plt.legend(loc='upper right')
    plt.title(f'Triple matching, matches: {num_matches}')
    # Display the map
    plt.show()
# Folium
folium_map_sar = pl.plot_matches_on_folium(unique_sar_df_tresholded)
folium_map_norsat = pl.plot_matches_on_folium(unique_norsat_df_tresholded) 
folium_map = pl.plot_triple_matches_on_folium(triple_match_df)

folium_map_sar.save(f"./images/matches_map_sar{current_time}.html")
folium_map_norsat.save(f"./images/matches_map_norsat{current_time}.html")

folium_map.save(f"./images/triple_matches_map_5_{current_time}.html")