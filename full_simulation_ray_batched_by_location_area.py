import os
import sys
import shutil
import glob

# math, GIS
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.warp import reproject, Resampling

# orchestration
import duckdb
import itertools
import ray
from ray.util.actor_pool import ActorPool
import subprocess
from tqdm import tqdm

# logging, timing
import configparser
import csv
import logging
import time
import pprint
from pathlib import Path
from datetime import datetime

# AvaFrame
import avaframe
from avaframe.com1DFA import com1DFA

# List of internal AvaFrame loggers that are being chatty
NOISY_LOGGERS = [
    "avaframe",
    "avaframe.com1DFA.com1DFATools",
    "avaframe.com1DFA.checkCfg",
    "avaframe.in3Utils.cfgUtils",
    "avaframe.in3Utils.geoTrans",
    "avaframe.com1DFA.deriveParameterSet",
    "avaframe.in1Data.getInput",
    "avaframe.in3Utils.initializeProject",
    "pyogrio._io"
]
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

def get_location_batches(gpkg_path, batch_size=1000, max_locations=None, anriss0005_flag=None):
    """
    Gets location batches from the location file
    
    :param gpkg_path: Description
    :param batch_size: Description
    """

    # --- OVERRIDE LOGIC ---
    if anriss0005_flag:
        # The structure is a list of tuples: [(id, x, y, area)]
        print(f"🎯 Anriss0005Flag active: Yielding {max_locations} copies of the same test location (2608198, 1145230, 200)")
        # Provide the same (batch_id, batch) structure as normal operation

        if max_locations is not None:
            for _ in range(max_locations):
                yield (0, [(0, 2608198, 1145230, 200)]) # Anriss0005 from all performance tests
        else:
            yield (0, [(0, 2608198, 1145230, 200)]) # Anriss0005 from all performance tests
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
    
    batch_id = 0
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        pass
        yield (batch_id, batch)
        batch_id += 1

def save_batch_to_csv(rows, logfile_path):
    """Append simulation log rows to `logfile_path` as CSV.

    Each row should be: [batch_id, loc_idx, x, y, area, relTh, mu, xsi, tau0, duration, status, message]
    """
    logfile_path = Path(logfile_path)
    header = ['batch_id', 'loc_idx', 'x', 'y', 'area', 'relTh', 'mu', 'xsi', 'tau0', 'duration', 'status', 'message']
    write_header = not logfile_path.exists()
    logfile_path.parent.mkdir(parents=True, exist_ok=True)
    with open(logfile_path, 'a', newline='') as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)
        for r in rows:
            # If row is nested (e.g., list of lists), flatten accordingly
            if isinstance(r, (list, tuple)) and len(r) and isinstance(r[0], (list, tuple)):
                for sub in r:
                    writer.writerow(sub)
            else:
                writer.writerow(r)

class AvaFrameOutputManager:
    def __init__(self, root_dir):
        self.root_path = Path(root_dir)
        self.simulation_base_path = self.root_path / "Simulations"
        self.raster_output_dir = self.simulation_base_path / "summary_stat_rasters"

    def create_stat_rasters(self, x, y, sim_paths):
        """
        Stacks all rasters to calculate Max, Min, and Avg for both Depth and Velocity,
        aligned to the cached DEM master grid.
        """
        import rasterio
        import numpy as np
        from rasterio.warp import reproject, Resampling

        # 1. Anchor to Cached DEM
        worst_case_str = "_".join([f"{k}{v}" for k, v in self.worst_case_parameters.items()])
        dem_cache_path = self.root_path / f"DEM_cache_{worst_case_str}" / f"DEM_X{x}_Y{y}_Area{area}.asc"
        with rasterio.open(dem_cache_path) as master_src:
            m_trans = master_src.transform
            m_width, m_height = master_src.width, master_src.height
            m_profile = master_src.profile.copy()
        
        m_profile.update(
            driver='GTiff',
            dtype='float32',
            compress='lzw',
            tiled=False
            )
        m_profile.pop('blockxsize')
        m_profile.pop('blockysize')

        # Initialize accumulation arrays
        # Depth Stats
        max_d = np.zeros((m_height, m_width), dtype='float32')
        min_d = np.full((m_height, m_width), np.inf, dtype='float32')
        sum_d = np.zeros((m_height, m_width), dtype='float32')
        
        # Velocity Stats (assuming 'pfv' Peak Velocity files)
        max_v = np.zeros((m_height, m_width), dtype='float32')
        min_v = np.full((m_height, m_width), np.inf, dtype='float32')
        sum_v = np.zeros((m_height, m_width), dtype='float32')

        # Count of simulations that hit each pixel (for Avg)
        count_mask = np.zeros((m_height, m_width), dtype='int32')

        # 2. Process all simulation results
        for path in sim_paths:
            # Find Depth (pft) and Velocity (pfv) files
            pft_file = list(Path(path / "Outputs/com1DFA/peakFiles").glob("*_pft.asc"))
            pfv_file = list(Path(path / "Outputs/com1DFA/peakFiles").glob("*_pfv.asc"))
            
            if not pft_file or not pfv_file: continue
            
            for file_path, current_sum, current_max, current_min in [
                (pft_file[0], sum_d, max_d, min_d), 
                (pfv_file[0], sum_v, max_v, min_v)
            ]:
                with rasterio.open(file_path) as src:
                    aligned = np.zeros((m_height, m_width), dtype='float32')
                    reproject(
                        source=src.read(1), destination=aligned,
                        src_transform=src.transform, src_crs="EPSG:2056",
                        dst_transform=m_trans, dst_crs="EPSG:2056",
                        resampling=Resampling.nearest
                    )
                    
                    # Update Stats
                    mask = aligned > 0
                    current_max[mask] = np.maximum(current_max[mask], aligned[mask])
                    current_min[mask] = np.minimum(current_min[mask], aligned[mask])
                    current_sum[mask] += aligned[mask]
                    
                    # We only update the count mask once per simulation path (using Depth as proxy)
                    if file_path == pft_file[0]:
                        count_mask[mask] += 1

        # 3. Finalize and Save
        # Clean up Min arrays (replace Infinity where no flow occurred with 0)
        min_d[min_d == np.inf] = 0
        min_v[min_v == np.inf] = 0
        
        # Calculate averages (avoid division by zero)
        avg_d = np.divide(sum_d, count_mask, out=np.zeros_like(sum_d), where=count_mask > 0)
        avg_v = np.divide(sum_v, count_mask, out=np.zeros_like(sum_v), where=count_mask > 0)

        # Export raster data
        stats = {
            f"X{x}_Y{y}_max_depth.tif": max_d, f"X{x}_Y{y}_min_depth.tif": min_d, f"X{x}_Y{y}_avg_depth.tif": avg_d,
            f"X{x}_Y{y}_max_vel.tif": max_v, f"X{x}_Y{y}_min_vel.tif": min_v, f"X{x}_Y{y}_avg_vel.tif": avg_v
        }

        self.raster_output_dir.mkdir(parents=True, exist_ok=True)
        for name, data in stats.items():
            with rasterio.open(self.raster_output_dir / name, 'w', **m_profile) as dst:
                dst.write(data, 1)

class AvaFrameAnrissManager:
    """
    Prepares Input for AvaFrame
    Extracts DEM into ASCII format used by AvaFrame
    

    :param master_dem_path: 5m DEM for Kanton Bern
    :param config_template_path: AvaFrame Config template
    :param root_dir: Root directory for the current simulation    
    """
    def __init__(self, master_dem_path, config_template_path, root_dir, worst_case_parameters):
        self.master_dem_path = master_dem_path
        self.root_path = Path(root_dir)
        self.simulation_base_path = self.root_path / "Simulations"
        self.config_template_path = config_template_path
        self.worst_case_parameters = worst_case_parameters
        self.os_env = os.environ.copy()
        self.os_env["GDAL_PAM_ENABLED"] = "NO"
        self.os_env["GDAL_OUT_PRJ"] = "NO"

        for logger_name in NOISY_LOGGERS:
            l = logging.getLogger(logger_name)
            l.setLevel(logging.WARNING) # Only show errors or warnings
            l.propagate = False         # Prevent them from sending logs up to your s

    def setup_dirs(self, root_path):
        input_dir = root_path / "Inputs"
        subdirs = ["CONFIG", "ENT", "LINES", "POINTS", "REL", "RELTH", "RES", "SECREL"]
        for s in subdirs:
            os.makedirs(input_dir / s, exist_ok=True)
        os.makedirs(root_path / "Outputs", exist_ok=True)
        return input_dir

    def extract_ascii_raster_from_master_dem(self, out_path, extent):
        sx_min, sx_max, sy_min, sy_max = [(v // 5) * 5 for v in extent]
        cmd = [
            "gdal_translate", "-of", "AAIGrid",
            "-projwin", str(sx_min), str(sy_max), str(sx_max), str(sy_min),
            "-co", "FORCE_CELLSIZE=YES",
            "-co", "DECIMAL_PRECISION=2",
            self.master_dem_path, str(out_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, env=self.os_env)
        if out_path.with_suffix('.prj').exists():
            out_path.with_suffix('.prj').unlink()

    def get_flow_bounds(self, result_ascii, safety_buffer=10):
        # setting default buffer to 10m is a bit arbitrary, just want to avoid problems with boundary pixels
        with rasterio.open(result_ascii) as src:
            data = src.read(1)
            rows, cols = np.where(data > 0.01)
            if rows.size == 0: return None
            xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='ul')
            res = src.res[0]
            return [min(xs) - safety_buffer, max(xs) + res + safety_buffer, min(ys) - safety_buffer, max(ys) + safety_buffer]

    def run_lead_sim(self, x, y, area):
        start_tc = time.perf_counter()
        print(f"🚀 Running lead simulation for (X, Y, area) = ({x}, {y}, {area})")
        p = self.worst_case_parameters
        sim_path = self.simulation_base_path / f"Sim_X{x}_Y{y}_A{area}_relTh{p['relTh']}_mu{p['mu']}_xsi{p['xsi']}_tau0{p['tau0']}"
        input_dir = self.setup_dirs(sim_path)
        
        shutil.copy(self.config_template_path, input_dir / "CONFIG" / "cfgCom1DFA.ini")
        
        # set initial buffer (here we start the search from)
        buffer, success = 1500, False
        buffer_delta = 500

        while not success and buffer <= 7500:
            xmin, xmax = x - buffer, x + buffer
            ymin, ymax = y - buffer, y + buffer
            extent = [xmin, xmax, ymin, ymax]
            extract_start = time.perf_counter()
            self.extract_ascii_raster_from_master_dem(input_dir / "dem.asc", extent)
            extract_dur = time.perf_counter() - extract_start
            print(f"   ⏱️ DEM extraction took {extract_dur:.2f}s")
            
            circle_start = time.perf_counter()
            radius = math.sqrt(area / math.pi)
            circle = Point(x, y).buffer(radius, quad_segs=16)
            gpd.GeoDataFrame([{'geometry': circle, 'id': 'rel_0'}], crs="EPSG:2056").to_file(input_dir / "REL" / "rel.shp")
            circle_dur = time.perf_counter()-circle_start
            print(f"   ⏱️ Circle polygon (area = {area}m2, r = {radius:.2f}m) creation took {circle_dur:.2f}s")

            cfgMain = avaframe.in3Utils.cfgUtils.getGeneralConfig()
            cfgMain['MAIN']['avalancheDir'] = str(sim_path)
            
            run_config = configparser.ConfigParser()
            run_config.optionxform = str
            run_config.read(input_dir / "CONFIG" / "cfgCom1DFA.ini")
            for k, v in [('relTh', p['relTh']),
                         ('muvoellmyminshear', p['mu']),
                         ('xsivoellmyminshear', p['xsi']), 
                         ('tau0voellmyminshear', p['tau0'])]:
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
        print(f"✂️✅ Lead Sim completed in {lead_duration:.2f}s. Tight Extent calculated. Needed buffer: +-{buffer}m")
        return [int(np.round(v // 5 * 5)) for v in tight_extent], buffer

    def get_DEM_for_location_area(self, x, y, area):
        """Return (dem_path, extent, buffer) for the worst-case DEM for (x,y, area).
        Uses a cache directory under the manager's `root_path` named by worst-case params.
        """
        worst_case_str = "_".join([f"{k}{v}" for k, v in self.worst_case_parameters.items()])
        dem_path = self.root_path / f"DEM_cache_X{x}_Y{y}_area{area}" / f"DEM_X{x}_Y{y}_area{area}_{worst_case_str}.asc"

        if dem_path.exists():
            print(f"🔄 Using existing cache for (X, Y, area) = ({x}, {y}, {area}): {dem_path}")
            header = {}
            with open(dem_path, 'r') as f:
                for _ in range(6):
                    line = f.readline().split()
                    if not line:
                        break
                    header[line[0].lower()] = float(line[1])
            try:
                x_min = header.get('xllcorner')
                y_min = header.get('yllcorner')
                ncols = int(header.get('ncols'))
                nrows = int(header.get('nrows'))
                cellsize = header.get('cellsize')
                x_max = x_min + (ncols * cellsize)
                y_max = y_min + (nrows * cellsize)
                extent = [x_min, x_max, y_min, y_max]
                buffer = 0
            except Exception as e:
                print(f"Error when opening DEM {dem_path}: {e}")
            return dem_path, extent, buffer

        # ----------------------------------------------------------------------------------------
        # DEM is not available in cache: run a lead sim to compute tight extent and create the DEM
        dem_path.parent.mkdir(parents=True, exist_ok=True)
        extent, buffer = self.run_lead_sim(x, y, area)
        self.extract_ascii_raster_from_master_dem(dem_path, extent)
        return dem_path, extent, buffer
        # ----------------------------------------------------------------------------------------

class AvaFrameBatchWorker:
    def __init__(self, dem_path, config_template, root_dir, parameters):
        self.parameters = parameters
        self.simulation_base_path = Path(root_dir) / "Simulations"
        self.worst_case_params = {k: v[0] for k, v in parameters.items()}
        self.worst_case_params_str = "_".join([f"{k}{v}" for k, v in self.worst_case_params.items()])
        self.root_path = Path(root_dir)
        self.anriss_manager = AvaFrameAnrissManager(dem_path, config_template, root_dir, self.worst_case_params)

        for logger_name in NOISY_LOGGERS:
            l = logging.getLogger(logger_name)
            l.setLevel(logging.WARNING) # Only show errors or warnings
            l.propagate = False         # Prevent them from sending logs up to your s

    def process_batch(self, batch_data, overwrite=False):
        """ 
        Batch = (batch_id, collection of locations and areas (x, y, area)) 
        """
        batch_id, x_y_area = batch_data
        batch_start = time.perf_counter()

        # Create a unique directory for this batch
        batch_dir_name = f"batch_{batch_id:04d}"
        batch_output_path = self.simulation_base_path / batch_dir_name
        batch_output_path.mkdir(parents=True, exist_ok=True)

        print(f"📦 Processing {batch_dir_name} ({len(x_y_area)} locations x area combinations)")
        # check if batch already processes
        def batch_already_processed(batch_output_path):
            data_lake_silver_path = Path(batch_output_path) / "batch_data_lake_silver" / "batch_merged_silver.parquet"
            data_lake_silver_present = data_lake_silver_path.exists()
            data_lake_bronze_path = Path(batch_output_path) / "batch_data_lake_bronze"
            data_lake_bronze_present = data_lake_bronze_path.exists() and data_lake_bronze_path.is_dir()
            has_Sim_dirs = len(glob.glob("batch_output_path/Sim*")) > 0
            if data_lake_silver_present and not data_lake_bronze_present and not has_Sim_dirs:
                return True
            else:
                return False

        if batch_already_processed(batch_output_path) and not overwrite:
            log_list = []
            log_list.append([batch_id, -999, -999, -999, -999, -999, -999, -999, -999, -999, "SKIPPED_BATCH", "silver data lake already present, overwrite = False"])
            return log_list
        else:
            pass  # overwrite batch

        results = []
        for location_area in x_y_area:
            # Pass the batch_dir_name and numeric batch_id down to process_location
            results.extend(self.process_location_area(location_area, batch_dir_name, batch_id))
        batch_duration = time.perf_counter() - batch_start
        print(f"📦 Batch ID {batch_id} in dir {batch_dir_name} ({len(x_y_area)} locations) processed in {batch_duration:.2f}s")
        return results
    
    def process_location_area(self, location_area_data, batch_dir_name, batch_id):
        loc_idx, x, y, area = location_area_data
        log_list, successful_sim_paths = [], []
        loc_start = time.perf_counter()
        print(f"📍 Start location idx {loc_idx}, (X, Y, area) = ({x}, {y}, {area})")
        try:          
            # 1. Get DEM for this location
            self.anriss_manager.simulation_base_path = self.simulation_base_path / batch_dir_name # add batch name to anriss manager so the lead simulation gets stored in the right batch
            DEM_worst_case_simulation_path, worst_case_simulation_extent, buffer = self.anriss_manager.get_DEM_for_location_area(x, y, area)
            self.anriss_manager.extract_ascii_raster_from_master_dem(DEM_worst_case_simulation_path, worst_case_simulation_extent)

            # 2. Parameter Grid
            keys = self.parameters.keys()
            # TODO extend with modifications / rules defined by AWN / TP Modellierung (instead of simple grid)
            parameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*self.parameters.values())]
            print(f"   Parameter grid: {len(parameter_combinations)} combinations. First 5: {parameter_combinations[:5]}")

            for parameter_combination in parameter_combinations:
                logger.debug(f"   -> Preparing sim for params: {parameter_combination}")
                sim_start = time.perf_counter()
                p = parameter_combination
                sim_path = self.simulation_base_path / batch_dir_name / f"Sim_X{x}_Y{y}_A{area}_relTh{p['relTh']}_mu{p['mu']}_xsi{p['xsi']}_tau0{p['tau0']}"
                
                if parameter_combination != self.worst_case_params or True:
                    try:
                        input_dir = self.anriss_manager.setup_dirs(sim_path)
                        shutil.copy2(DEM_worst_case_simulation_path, input_dir / "dem.asc")
                        radius = math.sqrt(area / math.pi)
                        circle = Point(x, y).buffer(radius, quad_segs=16)
                        rel_path = input_dir / "REL" / "rel.shp"
                        gpd.GeoDataFrame([{'geometry': circle, 'id': 'rel_0'}], crs="EPSG:2056").to_file(rel_path)
                        cfg = configparser.ConfigParser(); cfg.optionxform = str
                        cfg.read(self.anriss_manager.config_template_path)
                        for k, v in [('muvoellmyminshear', p['mu']), ('xsivoellmyminshear', p['xsi']), 
                                     ('tau0voellmyminshear', p['tau0']), ('relTh', p['relTh'])]:
                            cfg.set('GENERAL', k, str(v))
                        cfgM = avaframe.in3Utils.cfgUtils.getGeneralConfig(); cfgM['MAIN']['avalancheDir'] = str(sim_path)
                        logger.debug(f"      Starting com1DFA for sim {sim_path}")
                        # ------------------------------------------------------------------------------------------------
                        com1DFA.com1DFAMain(cfgM, cfgInfo=cfg)
                        # ------------------------------------------------------------------------------------------------
                        sim_dur = time.perf_counter() - sim_start
                        print(f"   ✅ Sim finished for area={area} relTh={p['relTh']} mu={p['mu']} xsi={p['xsi']} tau0={p['tau0']} in {sim_dur:.2f}s")
                        log_list.append([batch_id, loc_idx, x, y, area, p['relTh'], p['mu'], p['xsi'], p['tau0'], sim_dur, "SUCCESS", ""])

                    except Exception as e:
                        sim_dur = time.perf_counter() - sim_start
                        print(f"   ❌ Sim failed for area={area} relTh={p['relTh']} mu={p['mu']} xsi={p['xsi']} tau0={p['tau0']} after {sim_dur:.2f}s: {e}")
                        log_list.append([batch_id, loc_idx, x, y, area, p['relTh'], p['mu'], p['xsi'], p['tau0'], sim_dur, "FAIL", str(e)])
                else:
                    pass 
                    # TODO implement this logic if needed (probably not worth the effort)
                    sim_dur = time.perf_counter() - sim_start
                    print(f"   🔁 Skipped worst-case sim (✅ lead simulation already calculated) for area={area} relTh={p['relTh']} mu={p['mu']} xsi={p['xsi']} tau0={p['tau0']} ({sim_dur:.2f}s)")
                    log_list.append([batch_id, loc_idx, x, y, area, p['relTh'], p['mu'], p['xsi'], p['tau0'], sim_dur, "SUCCESS_LEAD", ""])
                
                successful_sim_paths.append(sim_path)

            loc_dur = time.perf_counter() - loc_start
            print(f"📍 Simulations for location idx={loc_idx} (x,y)=({x},{y}) completed in {loc_dur:.2f}s")

        except Exception as e:
            # duration unknown for critical errors - record as 0
            log_list.append([batch_id, loc_idx, x, y, 0, 0, 0, 0, 0, 0, "CRITICAL_ERROR", str(e)])
            raise
        return log_list
    
if __name__ == "__main__":

    # ===========================================================================================================
    # 0) CONFIG
    # ===========================================================================================================
    ROOT_DIR = "/home/bojan/probe_data/bern8"
    LOCAL_SINGLE_THREAD_DEBUG_MODE = False
    ANRISS0005_FLAG = True
    if LOCAL_SINGLE_THREAD_DEBUG_MODE:
        ROOT_DIR = "/home/bojan/probe_data/local_debug"
    total_start = time.perf_counter()
    BE_DEM_5M = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned_5km_buffer_COG_cropped.tif"
    CONFIG_TEMPLATE = "/home/bojan/probe_pre_processing/cfgCom1DFA_template.ini"
    LOCATIONS = "/home/bojan/probe_data/bern/locations_random_1000.gpkg"
    LOG_FILE = Path(ROOT_DIR) / f"log_started_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    RAY_MODE = "local_cluster"
    N_RAY_WORKERS = 8
    MAX_IN_FLIGHT = N_RAY_WORKERS * 2  # queue one task for every worker
    N_LIMIT_LOCATIONS = 8
    N_LOCATIONS_IN_BATCH = 1
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[PID %(process)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # ===========================================================================================================
    # 1) PREPARE SIMULATION
    # ===========================================================================================================

    # Simulation parameters
    raw_params = {
        'relTh': [0.75, 1, 3],
        'mu': [0.05, 0.25, 0.375],  
        'xsi': [200, 600, 1250],
        'tau0': [500, 1500],
    }
    # Sorting ensures index [0] is always the "Worst Case" = "Lead Simulation"
    sorted_params = {
        'relTh': sorted(raw_params['relTh'], reverse=True),
        'mu': sorted(raw_params['mu']),
        'xsi': sorted(raw_params['xsi'], reverse=True),
        'tau0': sorted(raw_params['tau0']),
    }

    # ===========================================================================================================
    # 2) LOCAL SINGLE-THREADED DEBUG
    # ==========================================================================================================
    if LOCAL_SINGLE_THREAD_DEBUG_MODE:
        print("RUNNING IN DEBUG MODE WITHOUT RAY")
        print("using AvaFrameBatchWorker directly, no remote, no ncpu")
        WorkerClass = AvaFrameBatchWorker
        # Manually call the method
        manual_override_params_sorted = {
            'relTh': [3, 1, 0.75],
            'mu': [0.05, 0.25, 0.375],
            'xsi': [1250, 600, 200],
            'tau0': [500, 1500],
        }
        # (batch id [(location id, x, y)])
        test_location_area = (0, [(0, 2608198, 1145230, 200)]) # Anriss0005 from all performance tests
    
        debug_worker = WorkerClass(BE_DEM_5M, CONFIG_TEMPLATE, ROOT_DIR, manual_override_params_sorted)
        results = debug_worker.process_batch(test_location_area)
        try:
            written = save_batch_to_csv(results, LOG_FILE)
            print(f"✅ Wrote debug log to {LOG_FILE} ({written} rows)")
            # show sample of log rows
            print("Sample log rows:")
            for r in results[:5]:
                print("  ", r)
        except Exception as e:
            print(f"⚠️ Failed to write debug log: {e}")
        sys.exit(0)

    # ===========================================================================================================
    # 3) INITIALIZE RAY CLUSTER
    # ==========================================================================================================
    print("Initializing Ray Cluster...")
    if not ray.is_initialized():
        if RAY_MODE == "local_debug":
            ray.init(local_mode=True)
        elif RAY_MODE == "local_cluster":
            ray.init()
        else:
            ray.init()

    def create_worker(*args, **kwargs):
        # wrapping to enable both local debug without decorator and remote with decorator
        return WorkerClass.remote(*args, **kwargs)

    # N_RAY_WORKERS = int(ray.available_resources().get("CPU", 4))
    print(f"Setting up {N_RAY_WORKERS} Ray Workers ...")
    WorkerClass = ray.remote(num_cpus=1)(AvaFrameBatchWorker)
    workers = [create_worker(BE_DEM_5M, CONFIG_TEMPLATE, ROOT_DIR, sorted_params) for i in range(N_RAY_WORKERS)]
    pool = ActorPool(workers)

    # ===========================================================================================================
    # 4) RUN SIMULATIONS
    # ==========================================================================================================

    # Prepare INPUT for simulation by setting up the DuckDB batch generator (batch = locations (X/Y))
    # TODO read from S3
    # TODO set total n locations in case N_LIMIT_LOCATIONS is not set
    batch_generator = get_location_batches(LOCATIONS, batch_size=N_LOCATIONS_IN_BATCH , max_locations=N_LIMIT_LOCATIONS, anriss0005_flag=ANRISS0005_FLAG)
    # dry run: get the first 10 entries of the batch generator and print them
    print(f"Running dry run: showing first 10 batches")
    for i in range(10):
        try:
            batch = next(batch_generator)
        except StopIteration:
            print('--- StopIteration: generator exhausted ---')
            break
        except Exception as e:
            print('NEXT_ERROR', e)
            break
        print(f'--- Batch {i+1} (len={len(batch)}) ---')
        pprint.pprint(batch)
    # real run
    print(f"Now starting real run")
    batch_generator = get_location_batches(LOCATIONS, batch_size=N_LOCATIONS_IN_BATCH , max_locations=N_LIMIT_LOCATIONS, anriss0005_flag=ANRISS0005_FLAG)
    
    # 1. SEND: Use map_unordered to distribute the generator across the 8 actors
    # This automatically maintains backpressure and keeps all 8 actors busy.
    # The 'job_data' here is now the (batch_id, batch_list) tuple
    result_generator = pool.map_unordered(
       lambda worker, job_data: worker.process_batch.remote(job_data), batch_generator
    )

    with tqdm(desc="Simulation", unit=" simulations", total=N_LIMIT_LOCATIONS) as pbar:
        # 2. COLLECT: Iterate directly over the result generator
        # This loop only advances when an actor finishes a batch
        for batch_results in result_generator:
            try:
                # 3. Incremental Save
                written = save_batch_to_csv(batch_results, LOG_FILE)
                
                # 4. Update Progress
                pbar.update(len(batch_results))
                
            except Exception as e:
                print(f"⚠️ Error bresult batch: {e}")

    total_dur = time.perf_counter() - total_start
    print(f"✅ Processing all complete. Total run time: {total_dur:.2f}s ⏱️")