import os
import shutil
import subprocess
import math
import json
import rasterio
import numpy as np
import geopandas as gpd
import configparser

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
        gdf.to_file(rel_dir / f"ReleaseX{x}Y{y}A{area}m2.shp")

    def get_flow_bounds(self, result_ascii, buffer_m=100):
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
            xs, ys = rasterio.transform.xy(src.transform, rows, cols)
            
            # Return tight bounds with safety buffer
            return [
                min(xs) - buffer_m, # min_x
                min(ys) - buffer_m, # min_y
                max(xs) + buffer_m, # max_x
                max(ys) + buffer_m  # max_y
            ]
        
    def _bounds_to_geopackage(self, bounds, output_file):
        """
        Converts bounds to a GeoPackage and overwrites any existing file.
        bounds order: [min_x, min_y, max_x, max_y]
        """
        # Ensure output_file is a Path object for clean handling
        output_path = Path(output_file)
        
        # 1. Create the bounding box geometry
        geom = box(float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
        
        # 2. Build the GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'release_x': [float(bounds[0] + (bounds[2]-bounds[0])/2)],
                'release_y': [float(bounds[1] + (bounds[3]-bounds[1])/2)],
                'area_m2': [float((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))]
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

    def run_probe_and_prepare(self, x, y):
        """
        The Full Workflow:
        1. Run Probe (6x6km)
        2. Analyze Output
        3. Define Production Extent
        """
        # --- 1. THE PROBE ---
        worst_case_area = self.worst_case_parameters['area']
        probe_path = self.base_path / f"probe_X{int(x)}_Y{int(y)}_A{worst_case_area}m2"
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
        # Assuming peak pressure file 'pp.asc' is generated in Outputs
        matches = list(Path(probe_path / "Outputs" / "com1DFA" / "peakFiles").glob("*_pft.asc"))
        result_pft_file = matches[0]
        tight_extent = self.get_flow_bounds(result_pft_file)
        
        print(f"✂️ Tight Extent calculated: {tight_extent}")
        tight_extent_path = Path(probe_path / "Outputs" / "com1DFA" / "tight_extent")
        self._bounds_to_geopackage(tight_extent, tight_extent_path)
        return tight_extent

if __name__ == "__main__":
    # Settings
    MASTER_TIF = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned.tif"
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
    manager = AvaFrameAnrissManager(MASTER_TIF, CONFIG_TEMPLATE, SIMULATION_DATA_ROOT_DIR, worst_case_parameters)
    
    # 1. Run the Probe ("worst case" simulation) and get the tight bounds
    production_extent = manager.run_probe_and_prepare(rel_x, rel_y)
    
    # 2. Setup Production run (using the tight extent)
    for area in parameters["area"]:
        mod_dir_name = f"AnrissX{rel_x}Y{rel_y}Area{area}m2"

        prod_path = Path(Path(SIMULATION_DATA_ROOT_DIR) / mod_dir_name)
        p_input = manager.setup_dirs(prod_path)
        manager.extract_dem(p_input / "dem.asc", production_extent)
        manager.create_release(p_input / "REL", rel_x, rel_y, area)
    
        print(f"✅ Production DEM optimized and ready at {prod_path}")

    # 3. Cleanup (remove probe)
    cleanup = False
    if cleanup:
        probe_path = Path(SIMULATION_DATA_ROOT_DIR) / f"probe_X{int(rel_x)}_Y{int(rel_y)}_A{int(worst_case_parameters["area"])}m2"
        if probe_path.exists() and probe_path.is_dir():
            print(f"🧹 Cleaning up directory: {probe_path}")
            shutil.rmtree(probe_path)
        # note that the results will be kept so we don't have to re-calculate the "worst case" again