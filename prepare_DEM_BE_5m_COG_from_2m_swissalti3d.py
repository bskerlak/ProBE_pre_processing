import os
import requests
import subprocess
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
CSV_PATH = "/home/bojan/probe_pre_processing/data/ch.swisstopo.swissalti3d-grosses-rechteck.csv"
TILES_DIR = "/home/bojan/probe_pre_processing/data/bern_2m_tiles"
OUTPUT_TIF = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned_5km_buffer_COG.tif"
OUTPUT_TIF_CROPPED = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned_5km_buffer_COG_cropped.tif"

def get_bern_extent(
    swissboundaries_gdb="/home/bojan/probe_pre_processing/data/swissboundaries/swissBOUNDARIES3D_1_5_LV95_LN02.gdb",
    layer="TLM_KANTONSGEBIET",
    buffer_meters: float = 5000.0
):
    """
    Read the canton polygon for Bern and return its extent
    expanded by `buffer_meters`.
    """
    gdf = gpd.read_file(swissboundaries_gdb, layer=layer)
    bern = gdf.loc[gdf.NAME == "Bern"]

    # Buffer and compute bounds
    buffered = bern.geometry.buffer(buffer_meters)
    minx, miny, maxx, maxy = buffered.total_bounds

    return bern.crs, (minx, miny, maxx, maxy)

def get_bern_buffered(
    swissboundaries_gdb="/home/bojan/probe_pre_processing/data/swissboundaries/swissBOUNDARIES3D_1_5_LV95_LN02.gdb",
    layer="TLM_KANTONSGEBIET",
    buffer_meters: float = 5000.0
):
    """Returns the Kanton Bern geometry buffered by buffer_meters."""
    gdf = gpd.read_file(swissboundaries_gdb, layer=layer)
    bern = gdf.loc[gdf.NAME == "Bern"].copy()
    bern["geometry"] = bern.geometry.buffer(buffer_meters)
    return bern

def download_tiles(csv_path, out_dir="/home/bojan/probe_pre_processing/data/bern_2m_tiles"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use 'names' to safely assign column names during read
    df = pd.read_csv(csv_path, header=None, names=['url'])
    urls = df.url.dropna().tolist()

    for url in tqdm(urls, desc="Downloading Tiles"):
        filename = Path(url).name
        out_path = out_dir / filename
        if out_path.exists(): continue

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            # Download to a temporary file first to ensure integrity
            tmp_path = out_path.with_suffix('.tmp')
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk: f.write(chunk)
            # Rename only after successful download
            tmp_path.rename(out_path)

def create_aligned_5m_cog(input_folder, output_tif, extent):
    """
    Creates a single 5m COG with corners snapped to 5m multiples and clipped to extent.
    """
    input_path = Path(input_folder)
    vrt_file = input_path / "temporary_mosaic.vrt"
    minx, miny, maxx, maxy = extent
    
    tiles = list(input_path.glob("*.tif"))
    if not tiles:
        print(f"❌ No tiles found in {input_folder}")
        return

    print(f"🏗️  Creating VRT for {len(tiles)} tiles...")
    
    # Use a file list to avoid "Argument list too long" error
    file_list_path = input_path / "file_list.txt"
    with open(file_list_path, "w") as f:
        f.write("\n".join(str(t) for t in tiles))

    subprocess.run(["gdalbuildvrt", "-input_file_list", str(file_list_path), str(vrt_file)], check=True)
    if file_list_path.exists(): file_list_path.unlink()

    # 2. Warp to COG with -tap and -te (Target Extent)
    # -of COG: Use the dedicated Cloud Optimized GeoTIFF driver
    # -te: Clip to the 5km buffered extent of Bern
    print(f"🚀 Warping to Cloud Optimized GeoTIFF (COG) with 5km Buffer...")
    warp_command = [
        "gdalwarp",
        "-of", "COG",              # Change output format to COG
        "-tr", "5", "5",           # 5m resolution
        "-r", "average",           # Downsampling method
        "-tap",                    # Snap to 5m grid
        "-te", str(minx), str(miny), str(maxx), str(maxy), # <--- CLIP TO BUFFERED EXTENT
        "-t_srs", "EPSG:2056",
        "-co", "COMPRESS=DEFLATE", # Standard COG compression
        "-co", "PREDICTOR=2",      # Optimization for DEM data
        "-co", "NUM_THREADS=ALL_CPUS",
        str(vrt_file), str(output_tif)
    ]
    
    subprocess.run(warp_command, check=True)
    if vrt_file.exists(): vrt_file.unlink()
    print(f"✅ Success! COG DEM with 5km buffer saved to {output_tif}")

if __name__ == "__main__":
    DOWNLOAD = False
    if DOWNLOAD:
        download_tiles(CSV_PATH, out_dir=TILES_DIR)

    PROCESS = False
    if PROCESS:
        crs, extent = get_bern_extent()
        print(f"📍 Buffered Extent (LV95): {extent}")
        
        create_aligned_5m_cog(
            TILES_DIR, 
            OUTPUT_TIF,
            extent
        )

    CROP = True
    if CROP:
        # crop the aligned 5m COG to kanton Bern (Polygon) +5 km buffer
        print(f"✂️  Cropping {OUTPUT_TIF} to Kanton Bern polygon (+5km buffer)...")
        
        # 1. Get buffered polygon
        bern_buffered = get_bern_buffered()
        
        # 2. Save cutline to temporary file
        cutline_path = Path(OUTPUT_TIF).parent / "cutline_bern.gpkg"
        bern_buffered.to_file(cutline_path, driver="GPKG")
        
        # 3. Warp with cutline
        cmd = [
            "gdalwarp",
            "-cutline", str(cutline_path),
            "-crop_to_cutline",
            "-dstnodata", "-9999",
            "-of", "COG",
            "-co", "COMPRESS=DEFLATE",
            "-co", "PREDICTOR=2",
            "-co", "NUM_THREADS=ALL_CPUS",
            "-overwrite",
            str(OUTPUT_TIF),
            str(OUTPUT_TIF_CROPPED)
        ]
        
        subprocess.run(cmd, check=True)
        
        # 4. Cleanup
        if cutline_path.exists():
            cutline_path.unlink()
            
        print(f"✅ Cropped COG saved to {OUTPUT_TIF_CROPPED}")