import os
import sys
import shutil
import subprocess
import math
import ray
import itertools
import time
import duckdb
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

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

def get_location_batches(gpkg_path, batch_size=1000, max_locations=None, anriss0005_flag=None):
    """
    Docstring for get_location_batches
    
    :param gpkg_path: Description
    :param batch_size: Description
    """

    # --- OVERRIDE LOGIC ---
    if anriss0005_flag is not None:
        # Yield a single batch containing only the target coordinate
        # The structure is a list of tuples: [(id, x, y)]
        print(f"🎯 Anriss0005Flag active: Yielding single test location (2608198, 1145230)")
        yield [(0, 2608198, 1145230)] # Anriss0005 from all performance tests
        return # Stop the generator here
    
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    query = f"""
        SELECT 
            id, 
            CAST(ST_X(geom) AS INTEGER) as x, 
            CAST(ST_Y(geom) AS INTEGER) as y 
        FROM st_read('{gpkg_path}', layer='samples')
    """
    if max_locations is not None:
        query += f" LIMIT {max_locations}"

    cursor = con.execute(query)
    
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        yield batch

class AvaFrameAnrissManager:
    def __init__(self, master_dem_path, config_template_path, root_path, worst_case_parameters):
        self.master_dem_path = master_dem_path
        self.simulation_base_path = Path(root_path) / "Simulations"
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
        start_tc = time.perf_counter()
        print(f"🚀 Running lead simulation for ({x}, {y})")
        p = self.worst_case_parameters
        # TODO skip lead simulation if DEM already present
        sim_path = self.simulation_base_path / f"Sim_X{x}_Y{y}_A{p['area']}_relTh{p['relTh']}_mu{p['mu']}_xsi{p['xsi']}_tau0{p['tau0']}"
        input_dir = self.setup_dirs(sim_path)
        
        shutil.copy(self.config_template_path, input_dir / "CONFIG" / "cfgCom1DFA.ini")
        
        buffer, success = 1000, False
        buffer_delta = 500

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
                buffer += buffer_delta
                print(f"Increased buffer by {buffer_delta}m to +-{buffer}m")

        matches = list((sim_path / "Outputs" / "com1DFA" / "peakFiles").glob("*_pft.asc"))
        tight_extent = self.get_flow_bounds(matches[0])
        lead_duration = time.perf_counter() - start_tc
        print(f"✂️✅ Tight Extent calculated: {tight_extent} - needed buffer: {buffer} (lead sim: {lead_duration:.2f}s)")
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

class AvaFrameBatchWorker:
    def __init__(self, dem_path, config_template, root_path, parameters):
        self.parameters = parameters
        self.simulation_base_path = Path(root_path) / "Simulations"
        self.worst_case_params = {k: v[0] for k, v in parameters.items()}
        self.worst_case_params_str = "_".join([f"{k}{v}" for k, v in self.worst_case_params.items()])
        self.root_dir = Path(root_path)
        self.manager = AvaFrameAnrissManager(dem_path, config_template, root_path, self.worst_case_params)

    def process_batch(self, batch):
        batch_start = time.perf_counter()
        print(f"📦 Processing batch of {len(batch)} locations")
        results = []
        for task in batch:
            results.extend(self.process_location(task))
        batch_duration = time.perf_counter() - batch_start
        print(f"📦 Batch processed in {batch_duration:.2f}s")
        return results

    def process_location(self, task_data):
        idx, x, y = task_data
        log, successful_sim_paths = [], []
        loc_start = time.perf_counter()
        print(f"📍 Start location idx={idx} ({x},{y})")
        try:
            # 1. Lead Simulation
            tight_extent, buffer = self.manager.run_lead_sim(x, y)
            
            # 2. Extract & Cache Production DEM
            cache_path = self.root_dir / f"DEM_cache_{self.worst_case_params_str}" / f"DEM_X{x}_Y{y}.asc"
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            self.manager.extract_dem(cache_path, tight_extent)

            # 3. Parameter Grid
            keys = self.parameters.keys()
            # TODO extend with modifications / rules defined by AWN / TP Modellierung (instead of simple grid)
            combinations = [dict(zip(keys, v)) for v in itertools.product(*self.parameters.values())]
            
            for p in combinations:
                sim_start = time.perf_counter()
                sim_path = self.simulation_base_path / f"Sim_X{x}_Y{y}_A{p['area']}_relTh{p['relTh']}_mu{p['mu']}_xsi{p['xsi']}_tau0{p['tau0']}"
                
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
                        sim_dur = time.perf_counter() - sim_start
                        print(f"   ✅ Sim finished for area={p['area']} relTh={p['relTh']} mu={p['mu']} xsi={p['xsi']} tau0={p['tau0']} in {sim_dur:.2f}s")
                        log.append([idx, x, y, p['area'], p['relTh'], p['mu'], p['xsi'], p['tau0'], "SUCCESS", ""])
                    except Exception as e:
                        sim_dur = time.perf_counter() - sim_start
                        print(f"   ❌ Sim failed for area={p['area']} relTh={p['relTh']} mu={p['mu']} xsi={p['xsi']} tau0={p['tau0']} after {sim_dur:.2f}s: {e}")
                        log.append([idx, x, y, p['area'], p['relTh'], p['mu'], p['xsi'], p['tau0'], "FAIL", str(e)])
                else:
                    sim_dur = time.perf_counter() - sim_start
                    print(f"   🔁 Skipped worst-case sim (✅ lead simulation already calculated) for area={p['area']} relTh={p['relTh']} mu={p['mu']} xsi={p['xsi']} tau0={p['tau0']} ({sim_dur:.2f}s)")
                    log.append([idx, x, y, p['area'], p['relTh'], p['mu'], p['xsi'], p['tau0'], "SUCCESS_LEAD", ""])
                
                successful_sim_paths.append(sim_path)

            # 4. Merge Results into MaxDepth GeoTIFF
            out_tiff = self.root_dir / "merged_results" / f"max_depth_X{x}_Y{y}.tif"
            merge_start = time.perf_counter()
            merged = self.manager.create_max_depth_raster(x, y, successful_sim_paths, out_tiff)
            merge_dur = time.perf_counter() - merge_start
            print(f"✅ Generated MaxDepth GeoTIFF for ({x}, {y}) (merged={merged}, time={merge_dur:.2f}s)")

            loc_dur = time.perf_counter() - loc_start
            print(f"📍 Location idx={idx} completed in {loc_dur:.2f}s")

        except Exception as e:
            log.append([idx, x, y, 0, 0, 0, 0, 0, "CRITICAL_ERROR", str(e)])
        return log
    
if __name__ == "__main__":
    total_start = time.perf_counter()

    BE_DEM_5M = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned_5km_buffer_COG_cropped.tif"
    CONFIG_TEMPLATE = "/home/bojan/probe_pre_processing/cfgCom1DFA_template.ini"
    SIM_ROOT = "/home/bojan/probe_data/bern"
    LOCATIONS = "/home/bojan/probe_data/bern/locations_random_1000.gpkg"
    LOG_FILE = Path(SIM_ROOT) / f"log_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    RAY_MODE = "local_debug"
    N_RAY_WORKERS = 8
    N_LIMIT_LOCATIONS = 1
    LOCAL_DEBUG_MODE = True

    # Simulation parameters
    raw_params = {
        'area': [200, 400, 1500],
        'relTh': [0.75, 1, 3],
        'mu': [0.05, 0.25, 0.375],
        'xsi': [200, 600, 1250],
        'tau0': [500, 1500],
    }
    # Sorting ensures index [0] is always the "Worst Case" = "Lead Simulation"
    sorted_params = {
        'area': sorted(raw_params['area'], reverse=True),
        'relTh': sorted(raw_params['relTh'], reverse=True),
        'mu': sorted(raw_params['mu']),
        'xsi': sorted(raw_params['xsi'], reverse=True),
        'tau0': sorted(raw_params['tau0']),
    }

    # Prepare Simulation by setting up the DuckDB batch generator (batch = locations (X/Y))
    # TODO read from S3
    batch_generator = get_location_batches(LOCATIONS, batch_size=10 , max_locations=N_LIMIT_LOCATIONS, anriss0005_flag=True)
    # next(batch_generator) # get next batch from generator

    MAX_IN_FLIGHT = N_RAY_WORKERS * 2  # queue one task for every worker
    in_flight = 0

    if LOCAL_DEBUG_MODE:
        print("RUNNING IN DEBUG MODE WITHOUT RAY")
        debug_worker = AvaFrameBatchWorker(BE_DEM_5M, CONFIG_TEMPLATE, SIM_ROOT, sorted_params)
        # Manually call the method
        test_location = [(0, 2608198, 1145230)]
        results = debug_worker.process_batch(test_location)
        sys.exit(0)

    # Initialize Ray Cluster
    print("Initializing Ray Cluster...")
    if not ray.is_initialized():
        if RAY_MODE == "local_debug":
            ray.init(local_mode=True)
        elif RAY_MODE == "local_cluster":
            ray.init()

    # N_RAY_WORKERS = int(ray.available_resources().get("CPU", 4))
    print(f"Setting up {N_RAY_WORKERS} Ray Workers ...")
    workers = [AvaFrameBatchWorker.options(name=f"AvaFrameBatchWorker{i}").remote(BE_DEM_5M, CONFIG_TEMPLATE, SIM_ROOT, sorted_params) for i in range(N_RAY_WORKERS)]
    pool = ActorPool(workers)

    # Run simulations
    with tqdm(desc="Locations", unit="Center Coordinates") as pbar:
        while True:
            # A. Feed the Pool until we hit backpressure limit
            while in_flight < MAX_IN_FLIGHT:
                # If the generator has been exhausted it will be set to None
                if batch_generator is None:
                    break
                try:
                    batch = next(batch_generator)  # take the next entry from the batch generator
                    pool.submit(lambda a, v: a.process_batch.remote(v), batch)
                    in_flight += 1
                except StopIteration:
                    # Exhausted the input list (generator "finished")
                    batch_generator = None # remove the generator
                    break
            
            # B. If everything is submitted and finished, exit
            if in_flight == 0:
                break
            
            # C. Collect results as they finish
            try:
                # get_next() waits for the next available result
                batch_results = pool.get_next()
                in_flight -= 1
                
                # D. Incremental Save (Essential!)
                # Write results to a unique parquet file to avoid file locking
                # TODO
                #save_batch_to_parquet(batch_results)
                
                pbar.update(len(batch_results))
                
            except Exception as e:
                print(f"⚠️ Worker error: {e}")
                in_flight -= 1 # Prevent deadlocks on failed tasks

    print("✅ Processing complete.")
    total_dur = time.perf_counter() - total_start
    print(f"⏱️ Total run time: {total_dur:.2f}s")