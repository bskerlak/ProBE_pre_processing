import os
import shutil
import subprocess
import math
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

def run_multi_area_sims(master_dem, release_x, release_y, areas_m2, template_path, simulation_input_root_path):
    """
    Creates separate AvaFrame directories for different release areas.
    """
    for area in areas_m2:
        # Calculate radius
        radius = math.sqrt(area / math.pi)
        sim_id = f"AnrissX{release_x}Y{release_y}A{int(area)}m2"
        
        # 1. Setup Directories

        base_dir = Path(simulation_input_root_path) / sim_id
        input_dir = base_dir / "Inputs"

        # AvaFrame standard subdirectories
        subdirs = [
            "CONFIG", "ENT", "LINES", "POINTS", 
            "REL", "RELTH", "RES", "SECREL"
        ]
        
        if base_dir.exists():
            shutil.rmtree(base_dir)
        for s in subdirs:
            os.makedirs(input_dir / s, exist_ok=True)

        print(f"\n--- Preparing {sim_id} (Radius: {radius:.2f}m) ---")

        # 2. Extract 5m DEM (4x4 km)
        # We snap to 5m to ensure the ASCII header is clean
        min_x, max_x = release_x - 2000, release_x + 2000
        min_y, max_y = release_y - 2000, release_y + 2000
        
        snap_min_x, snap_max_y = (min_x // 5) * 5, (max_y // 5) * 5
        snap_max_x, snap_min_y = (max_x // 5) * 5, (min_y // 5) * 5
        
        output_asc = input_dir / f"dem_{sim_id}.asc"
        
        # Using a safer subprocess call to see errors
        cmd = [
            "gdal_translate", "-of", "AAIGrid",
            "-projwin", str(snap_min_x), str(snap_max_y), 
            str(snap_max_x), str(snap_min_y),
            "-co", "FORCE_CELLSIZE=YES",
            "-co", "DECIMAL_PRECISION=2",
            "-co", "WRITE_PRJ=NO", # <--- Stops the .prj file creati
            "--config", "GDAL_PAM_ENABLED", "NO", # Stops the .aux.xml file creation
            master_dem,
            str(output_asc)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ DEM extracted.")
        except subprocess.CalledProcessError as e:
            print(f"❌ GDAL Fail for {sim_id}: {e.stderr}")
            continue

        # 3. Create Release Polygon
        circle = Point(release_x, release_y).buffer(radius)
        gdf = gpd.GeoDataFrame([{'geometry': circle, 'id': sim_id}], crs="EPSG:2056")
        
        # AvaFrame looks for .shp in the Release folder
        gdf.to_file(input_dir / "REL" / f"{sim_id}.shp")
        print(f"✅ Release shapefile created.")

        # 4. Copy config template
        target_cfg = input_dir / "CONFIG" / "cfgCom1DFA.ini"
        shutil.copy(template_path, target_cfg)
        print(f"✅ Config copied: {target_cfg.name}")

        # 5. Optional: Create the empty 'Outputs' folder (AvaFrame likes this)
        os.makedirs(base_dir / "Outputs", exist_ok=True)

    print("\n🚀 All directories ready for AvaFrame.")

if __name__ == "__main__":

    MASTER_TIF = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned.tif"
    TEMPLATE_INI = "/home/bojan/probe_pre_processing/cfgCom1DFA_template.ini"
    SIMULATION_INPUT_ROOT_PATH = "/home/bojan/probe_pre_processing/data/simulation_inputs"
    
    # Coordinates in LV95
    RELEASE_X = 2631000 
    RELEASE_Y = 1161000 
    
    # Input areas in square meters
    TARGET_AREAS = [200, 400, 1500]
    
    run_multi_area_sims(MASTER_TIF, RELEASE_X, RELEASE_Y, TARGET_AREAS, TEMPLATE_INI, SIMULATION_INPUT_ROOT_PATH)