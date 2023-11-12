import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

# Load the data
measurements = pd.read_csv("sorted_1.csv", encoding='latin1')

# Load the sensor information
# sensors = pd.read_csv("sensors.csv", encoding='latin1')


# Select the data of aircraft number 1716
flight = measurements[measurements['aircraft'] == 2]

# Load the map data
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Plot the map
fig, ax = plt.subplots(figsize=(10, 6))
world.boundary.plot(ax=ax, linewidth=0.5, color='gray')
flight.plot.scatter(x='longitude', y='latitude', c='geoAltitude', cmap='viridis', ax=ax, s=10)
# sensors.plot.scatter(x='longitude', y='latitude', color='coral', ax=ax, s=10)

# Set the map limits
xlims = (flight['longitude'].min() - 1, flight['longitude'].max() + 1)
ylims = (flight['latitude'].min() - 1, flight['latitude'].max() + 1)
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# Remove axis labels and ticks
ax.set_xticks([])
ax.set_yticks([])

plt.show()

