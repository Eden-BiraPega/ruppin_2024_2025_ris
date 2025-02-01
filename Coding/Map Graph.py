import folium
from geopy.geocoders import Nominatim
import time
import pandas as pd

def plot_israel_malls(df, mall_col='Mall', region_col='איזור'):
    """
    Plots an interactive map of Israel with markers for each mall.

    Parameters:
      df (pd.DataFrame): DataFrame containing at least two columns:
                         one with the mall name and one with the region.
      mall_col (str): Name of the column containing the mall name.
      region_col (str): Name of the column containing the region (location) in Israel.

    Returns:
      folium.Map: A folium map object with markers for each mall.
    """
    # Initialize the geocoder (using OpenStreetMap's Nominatim)
    geolocator = Nominatim(user_agent="israel_malls_app")

    # Create a base map centered on Israel (approximate center coordinates)
    israel_map = folium.Map(location=[31.0461, 34.8516], zoom_start=8)

    # Cache for geocoding results to avoid duplicate calls
    geocode_cache = {}

    for idx, row in df.iterrows():
        mall_name = row[mall_col]
        region = row[region_col]

        # If region is not in the cache, perform geocoding
        if region not in geocode_cache:
            # Construct an address by appending ", Israel"
            address = f"{region}, Israel"
            try:
                location = geolocator.geocode(address, timeout=10)
                if location:
                    geocode_cache[region] = location
                    # To be courteous with the Nominatim usage policy, add a delay.
                    time.sleep(1)
                else:
                    print(f"Geocode not found for: {address}")
                    geocode_cache[region] = None
            except Exception as e:
                print(f"Error geocoding {address}: {e}")
                geocode_cache[region] = None

        # Retrieve the cached geocoding result
        location = geocode_cache[region]

        # If geocoding was successful, add a marker to the map
        if location:
            folium.Marker(
                location=[location.latitude, location.longitude],
                popup=f"{mall_name} ({region})",
                tooltip=mall_name
            ).add_to(israel_map)
        else:
            print(f"Skipping marker for mall '{mall_name}' in region '{region}' (no location found).")

    return israel_map

# ---------------------------
# Example usage:
# ---------------------------
# Make sure this code is NOT indented further; it should start at the beginning of the line.

# Load your DataFrame (adjust the file name as needed)
df = pd.read_excel("merged_output.xlsx")

# Generate the map
israel_map = plot_israel_malls(df)

# Save the map to an HTML file
israel_map.save("israel_malls_map.html")

# If you're in a Jupyter/Colab environment, displaying the map object will render the map.
israel_map