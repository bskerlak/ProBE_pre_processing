import os
import shutil
import subprocess
import math
import json
import rasterio
import numpy as np
import geopandas as gpd
import configparser
import shutil
import csv
import random
import ray
import tqdm
import itertools

from pathlib import Path
from shapely.geometry import Point
from shapely.geometry import box
from shapely.prepared import prep

import avaframe
from avaframe.com1DFA import com1DFA
import time
from datetime import datetime

class AvaFrameAnrissManager:
    def __init__(self, master_dem_path, config_template_path, base_path, worst_case_parameters):
        self.master_dem_path = master_dem_path
        self.base_path = Path(base_path)
        self.config_template_path = config_template_path
        self.worst_case_parameters = worst_case_parameters
        self.os_env = os.environ.copy()
        self.os_env["GDAL_PAM_ENABLED"] = "NO"
        self.os_env["GDAL_OUT_PRJ"] = "NO"

    def setup_dirs(self, root_dir):
        """Creates standard AvaFrame directory structure."""
        input_dir = root_dir / "Inputs"
        subdirs = ["CONFIG", "ENT", "LINES", "POINTS", "REL", "RELTH", "RES", "SECREL"]
        for s in subdirs:
            os.makedirs(input_dir / s, exist_ok=True)
        os.makedirs(root_dir / "Outputs", exist_ok=True)
        return input_dir

    def extract_dem(self, out_path, extent):
        """Extracts a snapped 5m DEM using gdal_translate."""
        # Snapping logic to ensure 5m alignment
        # extent = [min_x, min_y, max_x, max_y]
        sx_min = (extent[0] // 5) * 5  # min_x
        sy_min = (extent[1] // 5) * 5  # min_y
        sx_max = (extent[2] // 5) * 5  # max_x
        sy_max = (extent[3] // 5) * 5  # max_y

        cmd = [
            "gdal_translate", "-of", "AAIGrid",
            "-projwin", 
            str(sx_min), str(sy_max), # Top-Left (Min X, Max Y)
            str(sx_max), str(sy_min), # Bottom-Right (Max X, Min Y)
            "-co", "FORCE_CELLSIZE=YES",
            "-co", "DECIMAL_PRECISION=2",
            self.master_dem_path, str(out_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, env=self.os_env)
        prj_file = out_path.with_suffix('.prj')
        if prj_file.exists():
            prj_file.unlink()


    def create_release(self, rel_dir, x, y, area):
        """Creates circular release area shapefile."""
        radius = math.sqrt(area / math.pi)
        circle = Point(x, y).buffer(radius, quad_segs=16)  # 4x16 = 64 Punkte für 1 Kreis
        gdf = gpd.GeoDataFrame([{'geometry': circle, 'id': 'rel_0'}], crs="EPSG:2056")
        gdf.to_file(rel_dir / f"AnrissX{x}Y{y}A{area}.shp")

    def get_flow_bounds(self, result_ascii, safety_buffer=100):
        """
        The Trim: Reads the simulation result and finds the actual used extent.
        """
        with rasterio.open(result_ascii) as src:
            data = src.read(1)
            # Find pixels with pressure/thickness > 0.01 (ignore tiny numerical noise)
            rows, cols = np.where(data > 0.01)
            
            if rows.size == 0:
                return None
            
            # Convert pixel indices back to coordinates
            # rasterio.transform * (col, row) gives (x, y)
            xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='ul')
            res = src.res[0] # 5.0
            # xs are the LEFT edges of the pixels
            # ys are the TOP edges of the pixels
            tight_left   = min(xs)
            tight_right  = max(xs) + res # Add one cell to cover the full width
            tight_top    = max(ys)
            tight_bottom = min(ys) - res # Subtract one cell to reach the LL corner
                        
            # Return tight bounds with safety buffer
            return [
                tight_left - safety_buffer, # min_x
                tight_bottom - safety_buffer, # min_y
                tight_right + safety_buffer, # max_x
                tight_top + safety_buffer  # max_y
            ]
        
    def _extent_to_geopackage(self, extent, output_file):
        """
        Converts extent to a GeoPackage and overwrites any existing file.
        extent order: [min_x, min_y, max_x, max_y]
        """
        # Ensure output_file is a Path object for clean handling
        output_path = Path(output_file)
        
        # 1. Create the bounding box geometry
        geom = box(extent[0], extent[1], extent[2], extent[3])
        
        # 2. Build the GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'center_x': [float(extent[0] + (extent[2]-extent[0])/2)],
                'center_y': [float(extent[1] + (extent[3]-extent[1])/2)],
                'area_m2': [float((extent[2] - extent[0]) * (extent[3] - extent[1]))]
            }, 
            geometry=[geom], 
            crs="EPSG:2056"
        )
        gdf.to_file(f"{output_path}.shp", driver="ESRI Shapefile")

        # 3. Force overwrite with the GPKG driver
        # We use engine='fiona' here because it is more direct about file handles
        gdf.to_file(
            f"{output_path}.gpkg", 
            driver="GPKG", 
            layer="tight_extent", 
            mode="w", 
            engine="fiona"
        )
            
        print(f"📥 Bounding box saved to GeoPackage: {output_file.name}")

    def finalize_extent(self, extent, grid_size=5):
        """
        Validates that extent coordinates are nearly integers, snaps to grid, 
        and converts to pure Python ints.
        """
        int_extent = []
        
        for val in extent:
            # 1. Check if the float is 'essentially' an integer
            # atol=1e-5 handles sub-millimeter floating point drift
            if not np.isclose(val, np.round(val), atol=1e-5):
                raise ValueError(f"Coordinate {val} has significant decimal noise and is not a whole number!")
                
            # 2. Convert to rounded integer
            rounded_val = int(np.round(val))
            
            # 3. Assert grid alignment (Multiple of 5)
            assert rounded_val % grid_size == 0, f"Value {rounded_val} is an integer but not aligned to the {grid_size}m grid!"
            
            int_extent.append(rounded_val)

        # 4. Final Geometry Check
        min_x, min_y, max_x, max_y = int_extent
        assert max_x > min_x and max_y > min_y, "Bounding box has zero or negative area!"
        
        return int_extent

    def run_probe_and_prepare(self, x, y):
        """
        The Full Workflow:
        1. Run Probe (6x6km)
        2. Analyze Output
        3. Define Production Extent
        """
        # --- 1. THE PROBE ---
        worst_case_area = self.worst_case_parameters['area']
        probe_path = self.base_path / f"ProbeAnrissX{int(x)}Y{int(y)}A{worst_case_area}"
        input_dir = self.setup_dirs(probe_path)
        
        # Copy template config
        shutil.copy(self.config_template_path, input_dir / "CONFIG" / "cfgCom1DFA.ini")
        
        buffer_init = 1000
        buffer_increment = 500
        buffer_max = 10000  # Safety cap at 20km width
        buffer = buffer_init
        simulation_successful = False

        while not simulation_successful and buffer <= buffer_max:

            print(f"Creating buffer +-{buffer}m")
            probe_extent = [x-buffer, y-buffer, x+buffer, y+buffer]
        
            self.extract_dem(input_dir / "dem.asc", probe_extent)
            self.create_release(input_dir / "REL", x, y, self.worst_case_parameters['area'])

            print(f"🚀 Probe prepared at {probe_path}. Run simulation now...")
            
            # --- AVAFRAME EXECUTION ---
            # set general settings
            cfgMain = avaframe.in3Utils.cfgUtils.getGeneralConfig()
            cfgMain['MAIN']['avalancheDir'] = str(probe_path)
            cfgMain['MAIN']['nCPU'] = str(1)

            # reading default config and updating parameters to loop over
            run_config = configparser.ConfigParser()
            run_config.optionxform=str # preserve CamelCase
            config_input_template = input_dir / "CONFIG" / "cfgCom1DFA.ini"
            run_config.read(config_input_template)
            
            # set worst case parameters
            run_config.set('GENERAL', 'muvoellmyminshear', str(self.worst_case_parameters['mu']))
            run_config.set('GENERAL', 'xsivoellmyminshear', str(self.worst_case_parameters['xsi']))
            run_config.set('GENERAL', 'tau0voellmyminshear', str(self.worst_case_parameters['tau0']))
            run_config.set('GENERAL', 'relTh', str(self.worst_case_parameters['relTh']))

            # run AvaFrame (run_config overrides cfgMain)
            print(f"Starting AvaFrame com1DFAMain for: {probe_path}")
            try:
                _, _, _, _, time_needed_total = com1DFA.com1DFAMain(cfgMain, cfgInfo=run_config)
                print(f"Simulation successful at buffer +-{buffer}m = {buffer*2/1000}km x {buffer*2/1000}km  - took {time_needed_total}s")
                simulation_successful = True
            except ValueError as e:
                print("⚠️error⚠️: Particles left the domain --> Next buffer")
                buffer += 500

        if not simulation_successful:
            raise RuntimeError(f"Could not contain avalanche even at {buffer_max}m buffer.")
                
        # --- 2. THE TRIM ---
        # Assuming peak flow depth file '*pft.asc' is generated in Outputs
        matches = list(Path(probe_path / "Outputs" / "com1DFA" / "peakFiles").glob("*_pft.asc"))
        assert len(matches) == 1, "Too many or no *_pft.asc file(s)"
        result_pft_file = matches[0]
        tight_extent = self.get_flow_bounds(result_pft_file)
        tight_extent_int = self.finalize_extent(tight_extent)
        
        print(f"✂️ Tight Extent calculated: {tight_extent_int}")
        tight_extent_path = Path(probe_path / "Outputs" / "com1DFA" / "tight_extent")
        self._extent_to_geopackage(tight_extent_int, tight_extent_path)
        # also return the buffer that was required to contain the simulation
        return tight_extent_int, buffer

@ray.remote
def process_batch(batch, master_dem, config_template, root_dir, parameters):
    # Identify worst case (first elements as per instruction)
    worst_case_params = {k: v[0] for k, v in parameters.items()}
    
    manager = AvaFrameAnrissManager(master_dem, config_template, root_dir, worst_case_params)
    
    # Generate all combinations
    keys = parameters.keys()
    values = parameters.values()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results_info = []
    
    for (x, y) in batch:
        try:
            print(f"🔄 Processing location {x}, {y}")
            # 1. Run Probe (Worst Case) -> Get Tight Extent
            tight_extent, buffer_used = manager.run_probe_and_prepare(x, y)
            
            # 2. Extract Tight DEM (Cache)
            cache_dir = Path(root_dir) / "cache_dems"
            cache_dir.mkdir(parents=True, exist_ok=True)
            tight_dem_path = cache_dir / f"tight_dem_{x}_{y}.asc"
            manager.extract_dem(tight_dem_path, tight_extent)
            
            # 3. Run All Simulations on Tight DEM
            # Use a temporary production directory for this coordinate
            prod_dir = Path(root_dir) / f"Production_X{x}_Y{y}"
            input_dir = manager.setup_dirs(prod_dir)
            
            # Copy Tight DEM to Inputs
            shutil.copy(tight_dem_path, input_dir / "dem.asc")
            
            for params in param_combinations:
                # Prepare Release (Clear existing first)
                for f in (input_dir / "REL").glob("*"):
                    f.unlink()
                manager.create_release(input_dir / "REL", x, y, params['area'])
                
                # Prepare Config
                run_config = configparser.ConfigParser()
                run_config.optionxform = str
                run_config.read(config_template)
                run_config.set('GENERAL', 'muvoellmyminshear', str(params['mu']))
                run_config.set('GENERAL', 'xsivoellmyminshear', str(params['xsi']))
                run_config.set('GENERAL', 'tau0voellmyminshear', str(params['tau0']))
                run_config.set('GENERAL', 'relTh', str(params['relTh']))
                
                # Run AvaFrame
                cfgMain = avaframe.in3Utils.cfgUtils.getGeneralConfig()
                cfgMain['MAIN']['avalancheDir'] = str(prod_dir)
                cfgMain['MAIN']['nCPU'] = '1'
                
                com1DFA.com1DFAMain(cfgMain, cfgInfo=run_config)
            
            # Cleanup
            shutil.rmtree(prod_dir)
            if tight_dem_path.exists(): tight_dem_path.unlink()
            
        except Exception as e:
            print(f"❌ Failed location {x}, {y}: {e}")
            
    return results_info

if __name__ == "__main__":
    # Settings
    BERN_DEM_5M = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned_5km_buffer_COG_cropped.tif"
    SWISSBOUNDARIES_GDB = "/home/bojan/probe_pre_processing/data/swissboundaries/swissBOUNDARIES3D_1_5_LV95_LN02.gdb"
    CONFIG_TEMPLATE = "/home/bojan/probe_pre_processing/cfgCom1DFA_template.ini"
    SIMULATION_DATA_ROOT_DIR = "/home/bojan/probe_data/bern"
    timestamp = datetime.now().strftime("%Y%m%d")
    PRE_PROCESSING_LOG = Path(SIMULATION_DATA_ROOT_DIR) / f"pre_processing_{timestamp}_log.csv"

    # Parameters
    # Sorted so that index 0 is the "worst case" (Probe)
    parameters = {
        'area': sorted([200, 400, 1500], reverse=True), # Max area
        'relTh': sorted([0.75, 1, 3], reverse=True),    # Max thickness
        'mu': sorted([0.05, 0.25, 0.375]),              # Min friction
        'xsi': sorted([200, 600, 1250], reverse=True),  # Max turbulence (min friction)
        'tau0': sorted([500, 1500]),                    # Min cohesion
    }

    # Initialize Ray
    ray.init()
    
    # 1. Read Locations from GPKG
    locations_gpkg = Path(SIMULATION_DATA_ROOT_DIR) / "locations_random_1000.gpkg"
    print(f"📂 Reading locations from {locations_gpkg}...")
    gdf = gpd.read_file(locations_gpkg)
    locations = [(int(p.x), int(p.y)) for p in gdf.geometry]
    print(f"✅ Loaded {len(locations)} locations.")

    # 2. Batching
    BATCH_SIZE = 10
    batches = [locations[i:i + BATCH_SIZE] for i in range(0, len(locations), BATCH_SIZE)]
    print(f"📦 Created {len(batches)} batches of size {BATCH_SIZE}.")

    # 3. Submit to Ray
    futures = [process_batch.remote(b, BERN_DEM_5M, CONFIG_TEMPLATE, SIMULATION_DATA_ROOT_DIR, parameters) for b in batches]
    
    # 4. Monitor
    with tqdm(total=len(batches)) as pbar:
        while futures:
            done, futures = ray.wait(futures)
            pbar.update(len(done))