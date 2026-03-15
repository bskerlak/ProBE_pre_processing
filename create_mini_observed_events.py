import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


def extract_point_data(beobachted_gdb, beobachted_punktdaten_extracted):
    def extract_grid_points(row):
        """Extracts 5x5m grid centers from a single polygon row."""
        poly = row.geometry
        rs_id = row['anrissprozessfl_beob_RS_ID_DokuProz']
        
        # 1. Get bounding box
        minx, miny, maxx, maxy = poly.bounds
        
        # 2. Align bounds to the nearest grid center (ending in 2.5 or 7.5)
        # Logic: round down to nearest 5, then add 2.5
        start_x = np.floor(minx / 5) * 5 + 2.5
        start_y = np.floor(miny / 5) * 5 + 2.5
        
        # 3. Create coordinate ranges
        x_coords = np.arange(start_x, maxx + 5, 5)
        y_coords = np.arange(start_y, maxy + 5, 5)
        
        # 4. Create meshgrid and filter
        xv, yv = np.meshgrid(x_coords, y_coords)
        points = np.vstack([xv.ravel(), yv.ravel()]).T
        
        # Use vectorized 'contains' check for speed
        # We wrap points in a Series to use shapely's vectorized logic
        potential_points = [Point(p) for p in points]
        mask = [poly.contains(p) for p in potential_points]
        
        # 5. Return as DataFrame
        valid_points = points[mask]
        return pd.DataFrame({
            'x': valid_points[:, 0],
            'y': valid_points[:, 1],
            'RS_ID': rs_id
        })
    
    # Apply to your GDB dataframe
    anrissprozessfl_beob = gpd.read_file(beobachted_gdb, layer='anrissprozessfl_beob')
    # If you have multiple polygons, this will concatenate all their grid points (siehe Überlagerungs-Thema)
    all_observed_points = pd.concat([extract_grid_points(row) for _, row in anrissprozessfl_beob.iterrows()])
    all_observed_points.to_csv(beobachted_punktdaten_extracted, index=False)
    print(all_observed_points.head())
    return all_observed_points


if __name__ == '__main__':
    CONTROL_CENTER_INPUT = Path("/home/bojan/probe_control_center/input/")
    beobachted_gdb  = CONTROL_CENTER_INPUT / "20260309_ablaufparameter_beobachtet/Beobachtet.gdb"
    beobachted_punktdaten_extracted = CONTROL_CENTER_INPUT / "20260309_beobachtete_events_punktdaten.csv"
    beobachted_df = extract_point_data(beobachted_gdb, beobachted_punktdaten_extracted)