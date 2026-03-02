import os
import sys
import shutil
import math
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from rasterio.windows import from_bounds
import duckdb
import itertools
import ray
from ray.util.actor_pool import ActorPool
import configparser
import csv
import logging
from pathlib import Path
from datetime import datetime
import time

# --- CONSTANTS & LOGGING ---
NOISY_LOGGERS = [
    "avaframe", "avaframe.com1DFA.com1DFATools", "avaframe.com1DFA.checkCfg",
    "avaframe.in3Utils.cfgUtils", "avaframe.in3Utils.geoTrans",
    "avaframe.com1DFA.deriveParameterSet", "avaframe.in1Data.getInput",
    "avaframe.in3Utils.initializeProject", "pyogrio._io"
]

def silence_loggers():
    """Stops AvaFrame from flooding the console during parallel execution."""
    for name in NOISY_LOGGERS:
        l = logging.getLogger(name)
        l.setLevel(logging.ERROR)
        l.propagate = False

# --- CORE UTILITIES ---

def extract_dem_fast(master_dem_path, out_path, extent):
    """
    Extracts a sub-region from a large GeoTIFF directly to ASCII.
    Avoids GDAL subprocess overhead.
    """
    # Align extent to 5m grid
    sx_min, sx_max, sy_min, sy_max = [(v // 5) * 5 for v in extent]
    
    with rasterio.open(master_dem_path) as src:
        window = from_bounds(sx_min, sy_min, sx_max, sy_max, src.transform)
        data = src.read(1, window=window)
        
        # AAIGrid manual write
        with open(out_path, 'w') as f:
            f.write(f"ncols         {data.shape[1]}\n")
            f.write(f"nrows         {data.shape[0]}\n")
            f.write(f"xllcorner     {sx_min}\n")
            f.write(f"yllcorner     {sy_min}\n")
            f.write(f"cellsize      {src.res[0]}\n")
            f.write(f"NODATA_value  {src.nodatavals[0] or -9999}\n")
            np.savetxt(f, data, fmt='%.2f')

def get_flow_bounds(result_ascii, safety_buffer=50):
    """Calculates the tight bounding box of avalanche flow (>1cm depth)."""
    with rasterio.open(result_ascii) as src:
        data = src.read(1)
        rows, cols = np.where(data > 0.01)
        if rows.size == 0: return None
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='ul')
        res = src.res[0]
        return [min(xs) - safety_buffer, max(xs) + res + safety_buffer, 
                min(ys) - safety_buffer, max(ys) + safety_buffer]

# --- RAY WORKER ---

@ray.remote(num_cpus=1)
class AvaFrameBatchWorker:
    def __init__(self, dem_path, config_template, root_dir, parameters):
        silence_loggers()
        self.master_dem = dem_path
        self.config_template = Path(config_template)
        self.root_path = Path(root_dir)
        self.parameters = parameters
        # Sort to find "Worst Case" (highest relTh, lowest mu, highest xsi)
        self.worst_case = {
            'relTh': sorted(parameters['relTh'])[-1],
            'mu': sorted(parameters['mu'])[0],
            'xsi': sorted(parameters['xsi'])[-1],
            'tau0': sorted(parameters['tau0'])[0]
        }

    def process_batch(self, batch_data):
        """Batch size of 1 recommended for optimal Ray load balancing."""
        batch_id, x_y_area_list = batch_data
        all_logs = []
        for loc_data in x_y_area_list:
            all_logs.extend(self.run_location_pipeline(batch_id, loc_data))
        return all_logs

    def run_location_pipeline(self, batch_id, loc_data):
        loc_idx, x, y, area = loc_data
        loc_logs = []
        
        # 1. Setup Master Paths for this Location
        loc_folder = self.root_path / f"Loc_{loc_idx}_A{area}"
        loc_folder.mkdir(parents=True, exist_ok=True)
        master_dem = loc_folder / "dem_final.asc"
        master_rel = loc_folder / "rel.shp"
        
        try:
            from avaframe.com1DFA import com1DFA
            from avaframe.in3Utils import cfgUtils

            # 2. RUN LEAD SIMULATION (To define tight DEM extent)
            if not master_dem.exists():
                lead_path = loc_folder / "LEAD_SIM"
                self._run_single_sim(lead_path, x, y, area, self.worst_case, initial_buffer=2000)
                
                # Calculate tight extent from lead result
                pft_file = next(lead_path.glob("Outputs/com1DFA/peakFiles/*_pft.asc"))
                tight_extent = get_flow_bounds(pft_file)
                extract_dem_fast(self.master_dem, master_dem, tight_extent)
                
                # Move lead release to master location
                for f in lead_path.glob("Inputs/REL/rel.*"):
                    shutil.move(f, loc_folder / f.name)
                shutil.rmtree(lead_path)

            # 3. RUN PARAMETER GRID
            keys = self.parameters.keys()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*self.parameters.values())]
            
            for p in combinations:
                sim_id = f"Th{p['relTh']}_mu{p['mu']}_xsi{p['xsi']}_tau0{p['tau0']}"
                sim_path = loc_folder / sim_id
                
                start_t = time.perf_counter()
                self._run_single_sim(sim_path, x, y, area, p, existing_dem=master_dem, existing_rel=master_rel)
                dur = time.perf_counter() - start_t
                
                loc_logs.append([batch_id, loc_idx, x, y, area, p['relTh'], p['mu'], p['xsi'], p['tau0'], dur, "SUCCESS", ""])

        except Exception as e:
            loc_logs.append([batch_id, loc_idx, x, y, area, 0, 0, 0, 0, 0, "CRITICAL", str(e)])
        
        return loc_logs

    def _run_single_sim(self, path, x, y, area, p, initial_buffer=None, existing_dem=None, existing_rel=None):
        """Internal helper to execute com1DFA."""
        from avaframe.com1DFA import com1DFA
        from avaframe.in3Utils import cfgUtils

        # Setup directory structure
        for d in ["Inputs/CONFIG", "Inputs/REL", "Inputs/DEM", "Outputs"]:
            (path / d).mkdir(parents=True, exist_ok=True)

        # Handle DEM (Symlink if possible)
        dem_target = path / "Inputs/DEM/dem.asc"
        if existing_dem:
            os.symlink(existing_dem, dem_target)
        else:
            extent = [x - initial_buffer, x + initial_buffer, y - initial_buffer, y + initial_buffer]
            extract_dem_fast(self.master_dem, dem_target, extent)

        # Handle Release
        rel_target = path / "Inputs/REL/rel.shp"
        if existing_rel:
            for s in ['.shp', '.shx', '.dbf', '.prj']:
                os.symlink(existing_rel.with_suffix(s), rel_target.with_suffix(s))
        else:
            radius = math.sqrt(area / math.pi)
            circle = Point(x, y).buffer(radius, quad_segs=16)
            gpd.GeoDataFrame([{'geometry': circle}], crs="EPSG:2056").to_file(rel_target)

        # Configure AvaFrame
        cfg = configparser.ConfigParser(); cfg.optionxform = str
        cfg.read(self.config_template)
        for k, v in [('muvoellmyminshear', p['mu']), ('xsivoellmyminshear', p['xsi']), 
                     ('tau0voellmyminshear', p['tau0']), ('relTh', p['relTh'])]:
            cfg.set('GENERAL', k, str(v))
        
        cfgM = cfgUtils.getGeneralConfig()
        cfgM['MAIN']['avalancheDir'] = str(path)
        com1DFA.com1DFAMain(cfgM, cfgInfo=cfg)

# --- MAIN ORCHESTRATOR ---

if __name__ == "__main__":
    # --- 1. CONFIGURATION ---
    ROOT_DIR = Path("/home/bojan/probe_data/final_production")
    BE_DEM = "/home/bojan/probe_pre_processing/data/Kanton_BE_5m_aligned_5km_buffer_COG_cropped.tif"
    CONFIG_TEMPLATE = "/home/bojan/probe_pre_processing/cfgCom1DFA_template.ini"
    LOCATIONS_GPKG = "/home/bojan/probe_data/bern/locations_random_1000.gpkg"
    N_CORES = 8
    LIMIT = 24

    params = {
        'relTh': [3.0, 1.0, 0.75],
        'mu': [0.05, 0.25, 0.375],
        'xsi': [1250, 600, 200],
        'tau0': [500, 1500],
    }

    # --- 2. INITIALIZATION ---
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    ray.init(num_cpus=N_CORES)
    
    workers = [AvaFrameBatchWorker.remote(BE_DEM, CONFIG_TEMPLATE, ROOT_DIR, params) for _ in range(N_CORES)]
    pool = ActorPool(workers)

    # Deterministic batch generator (Batch size 1 for granularity)
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    query = f"""
        SELECT l.id, CAST(ST_X(l.geom) AS INTEGER) as x, CAST(ST_Y(l.geom) AS INTEGER) as y, a.area
        FROM st_read('{LOCATIONS_GPKG}', layer='samples') l
        CROSS JOIN (SELECT UNNEST(ARRAY[100, 200, 500, 1000]) AS area) a
        ORDER BY HASH(l.id::VARCHAR || a.area::VARCHAR || '42')
        LIMIT {LIMIT}
    """
    batch_gen = ((i, [row]) for i, row in enumerate(con.execute(query).fetchall()))

    # --- 3. EXECUTION LOOP ---
    LOG_FILE = ROOT_DIR / f"results_{datetime.now().strftime('%H%M%S')}.csv"
    results_stream = pool.map_unordered(lambda w, b: w.process_batch.remote(b), batch_gen)

    print(f"🚀 Starting {LIMIT} location pipelines across {N_CORES} workers...")
    
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch', 'id', 'x', 'y', 'area', 'relTh', 'mu', 'xsi', 'tau0', 'dur', 'status', 'msg'])
        
        from tqdm import tqdm
        for result_list in tqdm(results_stream, total=LIMIT):
            writer.writerows(result_list)
            f.flush() # Ensure data is written even if script is interrupted

    print(f"\n✅ Done. Output: {LOG_FILE}")