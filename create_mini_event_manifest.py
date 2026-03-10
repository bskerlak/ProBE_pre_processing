import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from shapely.geometry import Point
from shapely.prepared import prep
from itertools import cycle # For cycling through the area list

# --- CONFIGURATION ---
NUM_LOCATIONS = 1000
SIMULATION_DATA_ROOT_DIR = Path(f"/home/bojan/probe_data/MINI{NUM_LOCATIONS}")
CONTROL_CENTER_ROOT_DIR = Path(f"/home/bojan/probe_control_center")
MINI_SHAPEFILE_ROOT = Path("/home/bojan/probe_control_center/input/ProBE_MINI_1000_v2")
MINI_TEIL_A_SHAPEFILE = MINI_SHAPEFILE_ROOT / "Anrissflaechen_sel1000.shp"
SIMULATION_DATA_ROOT_DIR.mkdir(parents=True, exist_ok=True)

# Output File Paths
mini_reibungsparameter_parquet_manifest = CONTROL_CENTER_ROOT_DIR / "input" / f"MINI{NUM_LOCATIONS}_reibungsparameter_event_manifest.parquet"
mini_reibungsparameter_gpkg_viz = CONTROL_CENTER_ROOT_DIR / "input" / f"MINI{NUM_LOCATIONS}_reibungsparameter_event_manifest_visualization.gpkg"

print(f"🎲 Generating {NUM_LOCATIONS} random locations in Kanton Bern...")

# MINI Teil a) Reibungsparameter
teil_a = False
if teil_a:
    mini_events_raw = gpd.read_file(MINI_TEIL_A_SHAPEFILE)
    mini_events = mini_events_raw[["RS_ID", "Anrh", "geometry"]]
    assert len(mini_events) == 1000
    print(f"💾 Saving MINI Teil a) Reibungsparameter - Parquet manifest: {mini_reibungsparameter_parquet_manifest}")
    mini_events.to_parquet(mini_reibungsparameter_parquet_manifest, index=False)
    mini_events.to_file(mini_reibungsparameter_gpkg_viz, driver="GPKG", layer="MINI 1000 Reibungsparameter")

# MINI Teil b) Sensitivität
teil_b = True
if teil_b:
    for area in [25, 70, 130, 250, 400, 700, 1100]:
        mini_events_raw = gpd.read_file(MINI_SHAPEFILE_ROOT / f"Anrisskreis{area}m2.shp")
        mini_events = mini_events_raw[["RS_ID", "Anrh", "geometry"]]
        assert len(mini_events) == 1000
        scales = [
            ('hmin', 0.5),
            ('hmean', 1.0),
            ('hmax', 1.5)
        ]
        df_expanded = mini_events.loc[mini_events.index.repeat(len(scales))].copy()
        suffix_labels = [s[0] for s in scales] * len(mini_events_raw)
        multipliers = [s[1] for s in scales] * len(mini_events_raw)
        df_expanded['Anrh'] = df_expanded['Anrh'] * multipliers
        df_expanded['suffix'] = suffix_labels
        df_expanded['RS_ID_extended'] = df_expanded.apply(
            lambda row: f"{row['RS_ID']}_A{area}_{row['suffix']}",
            axis=1
        )
        # Regel: unter 0.3m werden nur min und max behalten
        idx_to_be_filtered_out = (df_expanded["Anrh"] < 0.3) & (df_expanded["suffix"] == "hmean")
        print(f"Found {sum(idx_to_be_filtered_out)} entries to be filtered out")
        df_filtered = df_expanded.loc[~idx_to_be_filtered_out].copy()
        n_remaining = len(scales)*1000-sum(idx_to_be_filtered_out)
        assert len(df_filtered) == n_remaining
        df_filtered.drop(columns=["suffix"], inplace=True)
        df_filtered.loc[:, "Anrh_rounded"] = np.round(df_filtered.Anrh, 2)
        df_filtered['RS_ID_extended'] = df_filtered.apply(
            lambda row: f"{row['RS_ID_extended']}{row['Anrh_rounded']}",
            axis=1
        )
        df_filtered.loc[df_filtered.RS_ID == 179339]
        df_filtered.loc[:, "Anrh"] = df_filtered.Anrh_rounded
        df_filtered.drop(columns=["Anrh_rounded"], inplace=True)

        filename_out = CONTROL_CENTER_ROOT_DIR / "input" / f"MINI{NUM_LOCATIONS}_sensitivität_area{area}_event_manifest.parquet"
        print(f"💾 Saving MINI Teil b) Sensitivität - area {area}m2 - Parquet manifest ({n_remaining} rows): {filename_out}")
        df_filtered.to_parquet(filename_out, index=False)