import geopandas as gpd
import json

file_path = "/home/bojan/probe_data/adelboden/test_1ha_oben/merged_rel_shapes.geojson"
with open(file_path, 'r') as f:
    data = json.load(f)
gdf = gpd.GeoDataFrame.from_features(data['features'])

# 1. Extract the Area (using the 'Shape_Area' property already in your data)
# Or calculate it fresh from the geometry:
gdf['calc_area'] = gdf.geometry.area

# 2. Extract X and Y from the Polygon Centroids
gdf['x'] = gdf.geometry.centroid.x
gdf['y'] = gdf.geometry.centroid.y

# 3. Clean up the dataframe to keep only what you need
result = gdf[['x', 'y', 'calc_area', 'AnrissID']]
result.to_csv("/home/bojan/probe_data/adelboden/test_1ha_oben/merged_rel_shapes.geojson")
print(result.head())