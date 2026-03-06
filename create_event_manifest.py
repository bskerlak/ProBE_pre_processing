import geopandas as gpd
import random
import pandas as pd
from pathlib import Path
from shapely.geometry import Point
from shapely.prepared import prep
from itertools import cycle # For cycling through the area list

# --- CONFIGURATION ---
NUM_LOCATIONS = 128
SIMULATION_DATA_ROOT_DIR = Path(f"/home/bojan/probe_data/bern{NUM_LOCATIONS}")
CONTROL_CENTER_ROOT_DIR = Path(f"/home/bojan/probe_control_center")
SWISSBOUNDARIES_GDB = "/home/bojan/probe_pre_processing/data/swissboundaries/swissBOUNDARIES3D_1_5_LV95_LN02.gdb"
SIMULATION_DATA_ROOT_DIR.mkdir(parents=True, exist_ok=True)
AREA_SEQUENCE = [100, 100, 200, 300, 300, 500, 800, 1000]

# Output File Paths
parquet_manifest = CONTROL_CENTER_ROOT_DIR / "input" / f"bern{NUM_LOCATIONS}_event_manifest.parquet"
spatial_gpkg = CONTROL_CENTER_ROOT_DIR / "input" / f"bern{NUM_LOCATIONS}_event_manifest_visualization.gpkg"

print(f"🎲 Generating {NUM_LOCATIONS} random locations in Kanton Bern...")

# --- 1. LOAD KANTON BERN BOUNDARIES ---
gdf_boundaries = gpd.read_file(SWISSBOUNDARIES_GDB, layer="TLM_KANTONSGEBIET")
bern_geom = gdf_boundaries[gdf_boundaries.NAME == "Bern"].geometry.union_all()
bern_prep = prep(bern_geom)
minx, miny, maxx, maxy = bern_geom.bounds

# --- 2. GENERATE LOCATIONS ---
locations = []
while len(locations) < NUM_LOCATIONS:
    rx = random.uniform(minx, maxx)
    ry = random.uniform(miny, maxy)
    
    if bern_prep.contains(Point(rx, ry)):
        locations.append((int(rx), int(ry)))

# --- 3. MAP LOCATIONS TO AREAS ---
data_rows = []
# cycle() will restart the AREA_SEQUENCE list if NUM_LOCATIONS > len(AREA_SEQUENCE)
area_pool = cycle(AREA_SEQUENCE)

for i, (x, y) in enumerate(locations):
    # Get the next area from the cycled list
    area = next(area_pool)
    data_rows.append({
        'event_id': i,
        'x': x,
        'y': y,
        'area': area
    })

# Create DataFrame
df = pd.DataFrame(data_rows)

# --- 4. EXPORT PARQUET MANIFEST (For simulations) ---
print(f"💾 Saving Parquet manifest: {parquet_manifest}")
df.to_parquet(parquet_manifest, index=False)

# --- 5. EXPORT GEOPACKAGE (For QGIS Visualization) ---
print(f"💾 Saving GeoPackage for visualization: {spatial_gpkg}")
# Convert DataFrame to GeoDataFrame
geometry = [Point(xy) for xy in zip(df.x, df.y)]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2056")
gdf['geometry'] = gdf.geometry.buffer(gdf['area'].apply(lambda a: (a/3.14159)**0.5))
gdf.to_file(spatial_gpkg, driver="GPKG", layer="simulation_points")

print(f"✅ Done. Generated {len(df)} total events (location * area).")