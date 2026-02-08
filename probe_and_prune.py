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

from pathlib import Path
from shapely.geometry import Point
from shapely.geometry import box

import avaframe
from avaframe.com1DFA import com1DFA

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
        
        buffer = 500
        simulation_successful = False
        max_buffer = 10000  # Safety cap at 20km width

        while not simulation_successful and buffer <= max_buffer:

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
            raise RuntimeError(f"Could not contain avalanche even at {max_buffer}m buffer.")
                
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
        return tight_extent_int

if __name__ == "__main__":
    # Settings
    BERN_DEM_5M = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned.tif"
    CONFIG_TEMPLATE = "/home/bojan/probe_pre_processing/cfgCom1DFA_template.ini"
    SIMULATION_DATA_ROOT_DIR = "/home/bojan/probe_data/bern"

    # Parameters
    # default Pro-Mo parameter (from 2024; 3 Anrissflächen und 54 = 3*3*2*3 Parameter)
    parameters = {
        'area': [200, 400, 1500],
        'relTh': [0.75, 1, 3],
        'mu': [0.05, 0.25, 0.375],
        'xsi': [200, 600, 1250],
        'tau0': [500, 1500],
    }

    # get worst case parameter combination
    # grösste Masse = grösstes relTH, grösste Fläche / area
    # kleinste Reibung = kleinstes mü, grösstes xsi, kleinstes tau0  
    worst_case_parameters = {
        'area': max(parameters['area']),
        'relTh': max(parameters['relTh']),
        'mu': min(parameters['mu']),
        'xsi': max(parameters['xsi']),
        'tau0': min(parameters['tau0']),
    }
   
    # Release Coordinates (LV95)
    rel_x, rel_y = 2631000, 1161000
    rel_x, rel_y = 2608200, 1145230
    
    # AvaFrame Anriss Manager: Braucht Kanton BE DEM, Config Template und Output Directory
    manager = AvaFrameAnrissManager(BERN_DEM_5M, CONFIG_TEMPLATE, SIMULATION_DATA_ROOT_DIR, worst_case_parameters)
    
    # 1. Define a path for the 'Master' DEM for this specific coordinate
    # We use the rel_x and rel_y in the filename to keep it unique
    master_dem_name = f"production_dem_X{rel_x}_Y{rel_y}.asc"
    master_dem_path = Path(SIMULATION_DATA_ROOT_DIR) / "cache_dems" / master_dem_name
    master_dem_path.parent.mkdir(exist_ok=True)

    # 2. Calculate tight "worst case" extent and extract DEM only if it doesn't exist
    if not master_dem_path.exists():
        print(f"🌍 Extracting new Master DEM for {rel_x}/{rel_y}...")
        # Run the Probe ("worst case" simulation) and get the tight bounds for this release circle center coordinate (same for all areas)
        production_extent = manager.run_probe_and_prepare(rel_x, rel_y)
        manager.extract_dem(master_dem_path, production_extent)
    else:
        print(f"📦 Using cached Master DEM for {rel_x}/{rel_y}")
    
    # 3. Distribute DEM to production folders & setup input folders
    for area in parameters["area"]:
        mod_dir_name = f"AnrissX{rel_x}Y{rel_y}Area{area}"
        mod_dir_path = Path(Path(SIMULATION_DATA_ROOT_DIR) / mod_dir_name)
        p_input = manager.setup_dirs(mod_dir_path)
        
        # copy already extracted master DEM for this (x/y)
        shutil.copy2(master_dem_path, p_input)
        
        # create release polygon from circle center and area
        manager.create_release(p_input / "REL", rel_x, rel_y, area)
    
        print(f"✅ Production DEM and input folders ready at {mod_dir_path}")

    # 3. Cleanup (remove probe)
    cleanup = True
    if cleanup:
        probe_path = Path(SIMULATION_DATA_ROOT_DIR) / f"ProbeAnrissX{int(rel_x)}Y{int(rel_y)}A{int(worst_case_parameters["area"])}"
        if probe_path.exists() and probe_path.is_dir():
            print(f"🧹 Cleaning up directory: {probe_path}")
            shutil.rmtree(probe_path)
        xml_sidecar = master_dem_path.with_name(master_dem_path.name + ".aux.xml")
        if xml_sidecar.exists():
            xml_sidecar.unlink()
        # note that the results will be kept so we don't have to re-calculate the "worst case" again