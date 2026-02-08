import os
import requests
import subprocess

import pandas as pd
import geopandas as gpd

from tqdm import tqdm
from pathlib import Path


def get_bern_extent(
    swissboundaries_gdb="/home/bojan/probe_pre_processing/data/swissboundaries/swissBOUNDARIES3D_1_5_LV95_LN02.gdb",
    layer="TLM_KANTONSGEBIET",
    driver="OpenFileGDB"
):
    
    # Read Kantonsgebiete from Layer and extract Bern
    gdf = gpd.read_file(swissboundaries_gdb, layer=layer)
    bern=gdf.loc[gdf.NAME == "Bern"]

    # Extent in native CRS (LV95 / EPSG:2056)
    minx, miny, maxx, maxy = bern.total_bounds

    return  bern.crs, (minx, miny, maxx, maxy)


def download_tiles(
    csv_path,
    out_dir="/home/bojan/probe_pre_processing/data/bern_2m_tiles",
    header=0,
    chunk_size=1024 * 1024  # 1 MB
):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path, header=None)  # no header, column name = 0
    df.columns = ['url']
    urls = df.url.dropna().tolist()

    for url in tqdm(urls, desc="Downloading Tiles"):
        filename = os.path.basename(Path(url).name)
        out_path = os.path.join(out_dir, filename)

        # Skip if already downloaded
        if os.path.exists(out_path):
            continue

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)


def create_aligned_5m_mosaic(input_folder, output_tif):
    """
    Creates a single 5m TIF with corners snapped to multiples of 5m.
    """
    input_path = Path(input_folder)
    vrt_file = input_path / "temporary_mosaic.vrt"
    
    # 1. Build the Virtual Mosaic
    tiles = list(input_path.glob("*.tif"))
    if not tiles:
        print(f"❌ No tiles found in {input_folder}")
        return

    print(f"🏗️  Creating VRT for {len(tiles)} tiles...")
    subprocess.run(["gdalbuildvrt", str(vrt_file)] + [str(t) for t in tiles], check=True)

    # 2. Warp with -tap logic
    # -tr 5 5: Set target resolution to 5m
    # -tap: Snap extent to multiples of resolution (5m)
    # -r average: Mathematically clean downsampling from 2m -> 5m
    print(f"🚀 Warping and Snapping to 5m Grid...")
    warp_command = [
        "gdalwarp",
        "-tr", "5", "5",         # 5m resolution
        "-r", "average",         # Compute mean of underlying 2m pixels
        "-tap",                  # <--- THE MAGIC: Force multiples of 5m
        "-t_srs", "EPSG:2056",   # Swiss LV95
        "-co", "COMPRESS=LZW",   # Lossless compression
        "-co", "TILED=YES",      # Fast random access for simulation workers
        "-co", "NUM_THREADS=ALL_CPUS", # Use all available cores
        str(vrt_file), str(output_tif)
    ]
    
    subprocess.run(warp_command, check=True)
    
    # Cleanup VRT
    if vrt_file.exists(): vrt_file.unlink()
    print(f"✅ Success! Aligned DEM saved to {output_tif}")


if __name__ == "__main__":

    download = False
    if download:
        BERN_2M_TILES_SWISSALTI3D = "/home/bojan/probe_pre_processing/data/ch.swisstopo.swissalti3d-2m-bern.csv"
        download_tiles(BERN_2M_TILES_SWISSALTI3D)

    create_5m_tiles = True
    if create_5m_tiles:
        create_aligned_5m_mosaic("/home/bojan/probe_pre_processing/data/bern_2m_tiles", "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned.tif")
