import geopandas as gpd
import random

from pathlib import Path
from shapely.geometry import Point
from shapely.geometry import box
from shapely.prepared import prep

# Config
NUM_SAMPLES = 1000
SWISSBOUNDARIES_GDB = "/home/bojan/probe_pre_processing/data/swissboundaries/swissBOUNDARIES3D_1_5_LV95_LN02.gdb"
SIMULATION_DATA_ROOT_DIR = "/home/bojan/probe_data/bern"

print(f"🎲 Generating {NUM_SAMPLES} random locations in Kanton Bern...")

# Load Kanton Bern polygon
gdf_boundaries = gpd.read_file(SWISSBOUNDARIES_GDB, layer="TLM_KANTONSGEBIET")
bern_geom = gdf_boundaries[gdf_boundaries.NAME == "Bern"].geometry.union_all()
bern_prep = prep(bern_geom)
minx, miny, maxx, maxy = bern_geom.bounds

locations = []

# Chunking Setup
chunk_buffer = []
CHUNK_SIZE = 100000  # Write to disk every 100k samples to save RAM
out_gpkg = Path(SIMULATION_DATA_ROOT_DIR) / f"locations_random_{NUM_SAMPLES}.gpkg"
if out_gpkg.exists():
    out_gpkg.unlink()
file_initialized = False

while len(locations) < NUM_SAMPLES:
    rx = random.uniform(minx, maxx)
    ry = random.uniform(miny, maxy)
    
    # Check if inside polygon
    if bern_prep.contains(Point(rx, ry)):
        loc = (int(rx), int(ry))
        locations.append(loc)
        
        # Chunked Writing
        chunk_buffer.append(loc)
        if len(chunk_buffer) >= CHUNK_SIZE:
            print(f"💾 Appending chunk {len(locations)}/{NUM_SAMPLES} to GPKG...")
            gdf_chunk = gpd.GeoDataFrame(
                {'id': range(len(locations) - len(chunk_buffer), len(locations))}, 
                geometry=[Point(xy) for xy in chunk_buffer], 
                crs="EPSG:2056"
            )
            # Append to file (create if first chunk)
            mode = 'a' if file_initialized else 'w'
            gdf_chunk.to_file(out_gpkg, driver="GPKG", layer="samples", mode=mode)
            file_initialized = True
            chunk_buffer = []

# Save points within polygon in buffer
if chunk_buffer:
    print(f"💾 Appending final chunk...")
    gdf_chunk = gpd.GeoDataFrame(
        {'id': range(len(locations) - len(chunk_buffer), len(locations))}, 
        geometry=[Point(xy) for xy in chunk_buffer], 
        crs="EPSG:2056"
    )
    mode = 'a' if file_initialized else 'w'
    gdf_chunk.to_file(out_gpkg, driver="GPKG", layer="samples", mode=mode)

print(f"✅ Saved all {len(locations)} locations to {out_gpkg}")
