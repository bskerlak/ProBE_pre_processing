import os
import subprocess
from pathlib import Path

def extract_simulation_extent_dem(master_tif, output_ascii, min_x, min_y, max_x, max_y):
    """
    Extracts a rectangular subset from the Master 5m DEM and saves it as an ASCII Grid.
    
    Coordinates must be in EPSG:2056 (Swiss LV95).
    Example: min_x=2600000, min_y=1150000, max_x=2610000, max_y=1160000
    """
    
    # Ensure coordinates are snapped to the 5m grid to avoid shifts
    # This aligns your extraction exactly with the pixel boundaries
    snap_min_x = (min_x // 5) * 5
    snap_min_y = (min_y // 5) * 5
    snap_max_x = (max_x // 5) * 5
    snap_max_y = (max_y // 5) * 5

    print(f"🌍 Extracting: [{snap_min_x}, {snap_min_y}] to [{snap_max_x}, {snap_max_y}]")

    # gdal_translate is the best tool for windowed extraction
    # -projwin <ulx> <uly> <lrx> <lry> (Upper Left X, Upper Left Y, Lower Right X, Lower Right Y)
    cmd = [
        "gdal_translate",
        "-of", "AAIGrid",           # Output format: ESRI ASCII
        "-projwin", 
        str(snap_min_x), str(snap_max_y), # Top-Left
        str(snap_max_x), str(snap_min_y), # Bottom-Right
        "-co", "FORCE_CELLSIZE=YES",
        "-co", "DECIMAL_PRECISION=2",
        master_tif,
        output_ascii
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ ASCII subset created: {output_ascii}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Extraction failed: {e.stderr}")

if __name__ == "__main__":
    # Example: A 2km x 2km area near Interlaken/Bern region
    MASTER_DEM = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned.tif"
    OUTPUT_FILE = "/home/bojan/probe_pre_processing/data/simulation_dem/sim_zone_A1.asc"
    
    extract_simulation_extent_dem(
        MASTER_DEM, 
        OUTPUT_FILE, 
        min_x=2630000, min_y=1160000, 
        max_x=2632000, max_y=1162000
    )