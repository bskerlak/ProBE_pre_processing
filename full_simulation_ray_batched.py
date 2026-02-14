import os
import shutil
import subprocess
import math
import csv
import ray
import itertools
import time
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
import configparser
from shapely.geometry import Point
from ray.util.actor_pool import ActorPool

import avaframe
from avaframe.com1DFA import com1DFA

class AvaFrameAnrissManager:
    def __init__(self, master_dem_path, config_template_path, root_path, worst_case_parameters):
        self.master_dem_path = master_dem_path
        self.results_base_path = Path(root_path) / "results"
        self.config_template_path = config_template_path
        self.worst_case_parameters = worst_case_parameters
        self.os_env = os.environ.copy()
        self.os_env["GDAL_PAM_ENABLED"] = "NO"
        self.os_env["GDAL_OUT_PRJ"] = "NO"

    def setup_dirs(self, root_dir):
        input_dir = root_dir / "Inputs"
        subdirs = ["CONFIG", "ENT", "LINES", "POINTS", "REL", "RELTH", "RES", "SECREL"]
        for s in subdirs:
            os.makedirs(input_dir / s, exist_ok=True)
        os.makedirs(root_dir / "Outputs", exist_ok=True)
        return input_dir

    def extract_dem(self, out_path, extent):
        sx_min, sy_min, sx_max, sy_max = [(v // 5) * 5 for v in extent]
        cmd = [
            "gdal_translate", "-of", "AAIGrid",
            "-projwin", str(sx_min), str(sy_max),
            str(sx_max), str(sy_min),
            "-co", "FORCE_CELLSIZE=YES",
            "-co", "DECIMAL_PRECISION=2",
            self.master_dem_path, str(out_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, env=self.os_env)
        if out_path.with_suffix('.prj').exists():
            out_path.with_suffix('.prj').unlink()

    def get_flow_bounds(self, result_ascii, safety_buffer=100):
        with rasterio.open(result_ascii) as src:
            data = src.read(1)
            rows, cols = np.where(data > 0.01)
            if rows.size == 0: return None
            xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='ul')
            res = src.res[0]
            return [min(xs) - safety_buffer, min(ys) - safety_buffer, 
                    max(xs) + res + safety_buffer, max(ys) + safety_buffer]

    def run_lead_sim(self, x, y):
        print(f"🚀 Running lead simulation for ({x}, {y})")
        p = self.worst_case_parameters
        sim_id = f"X{x}_Y{y}_A{p['area']}_relTh{p['relTh']}_mu{p['mu']}_xsi{p['xsi']}_tau0{p['tau0']}"
        sim_path = self.results_base_path / f"Sim_{sim_id}"
        input_dir = self.setup_dirs(sim_path)
        
        shutil.copy(self.config_template_path, input_dir / "CONFIG" / "cfgCom1DFA.ini")
        
        buffer, success = 1000, False
        while not success and buffer <= 10000:
            extent = [x-buffer, y-buffer, x+buffer, y+buffer]
            self.extract_dem(input_dir / "dem.asc", extent)
            radius = math.sqrt(p['area'] / math.pi)
            circle = Point(x, y).buffer(radius, quad_segs=16)
            gpd.GeoDataFrame([{'geometry': circle, 'id': 'rel_0'}], crs="EPSG:2056").to_file(input_dir / "REL" / "rel.shp")

            cfgMain = avaframe.in3Utils.cfgUtils.getGeneralConfig()
            cfgMain['MAIN']['avalancheDir'] = str(sim_path)
            
            run_config = configparser.ConfigParser()
            run_config.optionxform = str
            run_config.read(input_dir / "CONFIG" / "cfgCom1DFA.ini")
            for k, v in [('muvoellmyminshear', p['mu']), ('xsivoellmyminshear', p['xsi']), 
                         ('tau0voellmyminshear', p['tau0']), ('relTh', p['relTh'])]:
                run_config.set('GENERAL', k, str(v))

            try:
                com1DFA.com1DFAMain(cfgMain, cfgInfo=run_config)
                success = True
            except ValueError:
                buffer += 1000

        matches = list((sim_path / "Outputs" / "com1DFA" / "peakFiles").glob("*_pft.asc"))
        tight_extent = self.get_flow_bounds(matches[0])
        print(f"✂️ Tight Extent calculated: {tight_extent} - needed buffer: {buffer}")
        return [int(np.round(v // 5 * 5)) for v in tight_extent], buffer

    def create_max_depth_raster(self, x, y, sim_paths, output_path):
        """Stacks all peak flow thickness rasters, aligning them to a common master grid."""
        import rasterio
        from rasterio.warp import reproject, Resampling
        
        max_data = None
        master_profile = None
        master_transform = None
        master_width = None
        master_height = None

        for path in sim_paths:
            matches = list(Path(path / "Outputs" / "com1DFA" / "peakFiles").glob("*_pft.asc"))
            if not matches:
                continue
            
            with rasterio.open(matches[0]) as src:
                # First file encountered sets the 'Master' geometry (usually the Lead Sim)
                if max_data is None:
                    max_data = src.read(1)
                    master_profile = src.profile.copy()
                    master_transform = src.transform
                    master_width = src.width
                    master_height = src.height
                    # Update profile for GeoTIFF output
                    master_profile.update(driver='GTiff', dtype='float32', count=1, compress='lzw')
                else:
                    current_data = src.read(1)
                    
                    # Check if dimensions match; if not, reproject into the master grid
                    if current_data.shape != max_data.shape:
                        # Initialize an empty array of the master's size
                        aligned_data = np.zeros((master_height, master_width), dtype='float32')
                        
                        reproject(
                            source=current_data,
                            destination=aligned_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=master_transform,
                            dst_crs=master_profile['crs'],
                            resampling=Resampling.nearest
                        )
                        max_data = np.maximum(max_data, aligned_data)
                    else:
                        max_data = np.maximum(max_data, current_data)

        if max_data is not None:
            output_path.parent.mkdir(exist_ok=True, parents=True)
            with rasterio.open(output_path, 'w', **master_profile) as dst:
                dst.write(max_data, 1)
            return True
        return False

@ray.remote(num_cpus=1)
class AvaFrameBatchWorker:
    def __init__(self, dem_path, config_template, root_dir, parameters):
        self.parameters = parameters
        self.worst_case_params = {k: v[0] for k, v in parameters.items()}
        self.root_dir = Path(root_dir)
        self.manager = AvaFrameAnrissManager(dem_path, config_template, root_dir, self.worst_case_params)

    def process_batch(self, batch):
        results = []
        for task in batch:
            results.extend(self.process_location(task))
        return results

    def process_location(self, task_data):
        idx, x, y = task_data
        log, successful_sim_paths = [], []
        try:
            # 1. Lead Simulation
            tight_extent, buffer = self.manager.run_lead_sim(x, y)
            
            # 2. Extract & Cache Production DEM
            cache_path = self.root_dir / "cache_dems" / f"prod_X{x}_Y{y}.asc"
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            self.manager.extract_dem(cache_path, tight_extent)

            # 3. Parameter Grid
            keys = self.parameters.keys()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*self.parameters.values())]
            
            for p in combinations:
                sim_id = f"X{x}_Y{y}_A{p['area']}_relTh{p['relTh']}_mu{p['mu']}_xsi{p['xsi']}_tau0{p['tau0']}"
                sim_path = self.root_dir / "results" / f"Sim_{sim_id}"
                
                if p != self.worst_case_params:
                    try:
                        p_in = self.manager.setup_dirs(sim_path)
                        shutil.copy2(cache_path, p_in / "dem.asc")
                        radius = math.sqrt(p['area'] / math.pi)
                        circle = Point(x, y).buffer(radius, quad_segs=16)
                        gpd.GeoDataFrame([{'geometry': circle, 'id': 'rel_0'}], crs="EPSG:2056").to_file(p_in / "REL" / "rel.shp")
                        
                        cfg = configparser.ConfigParser(); cfg.optionxform = str
                        cfg.read(self.manager.config_template_path)
                        for k, v in [('muvoellmyminshear', p['mu']), ('xsivoellmyminshear', p['xsi']), 
                                     ('tau0voellmyminshear', p['tau0']), ('relTh', p['relTh'])]:
                            cfg.set('GENERAL', k, str(v))

                        cfgM = avaframe.in3Utils.cfgUtils.getGeneralConfig(); cfgM['MAIN']['avalancheDir'] = str(sim_path)
                        com1DFA.com1DFAMain(cfgM, cfgInfo=cfg)
                        log.append([idx, x, y, p['area'], p['relTh'], p['mu'], p['xsi'], p['tau0'], "SUCCESS", ""])
                    except Exception as e:
                        log.append([idx, x, y, p['area'], p['relTh'], p['mu'], p['xsi'], p['tau0'], "FAIL", str(e)])
                else:
                    log.append([idx, x, y, p['area'], p['relTh'], p['mu'], p['xsi'], p['tau0'], "SUCCESS_LEAD", ""])
                
                successful_sim_paths.append(sim_path)

            # 4. Merge Results into MaxDepth GeoTIFF
            out_tiff = self.root_dir / "merged_results" / f"max_depth_X{x}_Y{y}.tif"
            self.manager.create_max_depth_raster(x, y, successful_sim_paths, out_tiff)
            print(f"✅ Generated MaxDepth GeoTIFF for ({x}, {y})")

        except Exception as e:
            log.append([idx, x, y, 0, 0, 0, 0, 0, "CRITICAL_ERROR", str(e)])
        return log

if __name__ == "__main__":
    # --- CONFIG & RUN ---
    DEM = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned_5km_buffer_COG_cropped.tif"
    TEMPLATE = "/home/bojan/probe_pre_processing/cfgCom1DFA_template.ini"
    ROOT = "/home/bojan/probe_data/bern"
    LOG_FILE = Path(ROOT) / f"log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    
    # Define and Sort Parameters
    raw_params = {
        'area': [200, 400, 1500],
        'relTh': [0.75, 1, 3],
        'mu': [0.05, 0.25, 0.375],
        'xsi': [200, 600, 1250],
        'tau0': [500, 1500],
    }
    # Sorting ensures index [0] is always the "Worst Case" Lead Sim
    sorted_params = {
        'area': sorted(raw_params['area'], reverse=True),
        'relTh': sorted(raw_params['relTh'], reverse=True),
        'mu': sorted(raw_params['mu']),
        'xsi': sorted(raw_params['xsi'], reverse=True),
        'tau0': sorted(raw_params['tau0']),
    }

    # --- 2. INITIALIZE RAY ---
    if not ray.is_initialized():
        ray.init()

    # --- 3. PREPARE TASKS ---
    gdf = gpd.read_file(Path(ROOT) / "locations_random_1000.gpkg")
    # Define tasks here so it's accessible
    tasks = [(idx, int(p.x), int(p.y)) for idx, p in enumerate(gdf.geometry)]
    tasks = tasks[:10]  # Limit for testing

    # --- 4. SETUP WORKERS ---
    n_workers = int(ray.available_resources().get("CPU", 4))
    workers = [AvaFrameBatchWorker.remote(DEM, TEMPLATE, ROOT, sorted_params) for _ in range(n_workers)]
    pool = ActorPool(workers)

    print(f"🚀 Starting processing for {len(tasks)} locations using {n_workers} workers.")

    # --- 5. EXECUTION WITH TQDM ---
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # Detailed Header
        writer.writerow(['idx', 'x', 'y', 'area', 'relTh', 'mu', 'xsi', 'tau0', 'status', 'error'])

        with tqdm(total=len(tasks), desc="Avalanche Locations", unit="loc", dynamic_ncols=True) as pbar:
            # map_unordered yields results as soon as any worker is done
            # We pass [v] so each worker takes exactly one location task
            for batch_results in pool.map_unordered(lambda a, v: a.process_batch.remote([v]), tasks):
                for row in batch_results:
                    writer.writerow(row)
                    
                    # Log errors to terminal without breaking the bar
                    if "FAIL" in str(row[8]) or "ERROR" in str(row[8]):
                        pbar.write(f"⚠️  Task {row[0]} failed: {row[9]}")
                
                f.flush() # Ensure data is written to disk frequently
                pbar.update(1)

    print(f"🏁 Done! MaxDepth TIFFs are in 'merged_results' and log is at {LOG_FILE}")