"""
03_build_basin_databases.py
===========================
Splits last_grounded_point into one SQLite database per Zwally basin.

For each basin (1–26), a separate .db file is created in BASIN_DB_DIR with
a single table (basin_data) containing only the rows whose (x, y) coordinates
fall within that basin according to Basins_Zwally_16km.nc.

The basin analysis (theoretical fluxes, plotting) is handled separately
by basin_analysis.py, which reads from these per-basin databases.
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import xarray as xr

import config


# =============================================================================
# HELPERS
# =============================================================================

def _load_basin_grid(nc_path: str) -> tuple:
    """Returns (basin_id 2D array, x_coords 1D, y_coords 1D)."""
    ds = xr.open_dataset(nc_path)
    basin_id = ds["Basin_ID"].values
    x_coords = ds["x"].values
    y_coords = ds["y"].values
    ds.close()
    return basin_id, x_coords, y_coords


def _basin_xy_sets(basin_id: np.ndarray,x_coords: np.ndarray,y_coords: np.ndarray) -> dict:
    """
    Returns a dict mapping basin index → set of (x, y) tuples
    belonging to that basin.
    """
    basins = {}
    ids = np.unique(basin_id[~np.isnan(basin_id.astype(float))]).astype(int)
    for i in ids:
        rows, cols = np.where(basin_id == i)
        basins[i] = set(zip(x_coords[cols].tolist(), y_coords[rows].tolist()))
    return basins


CREATE_BASIN_TABLE = """
CREATE TABLE IF NOT EXISTS basin_data (
    x                    REAL,
    y                    REAL,
    simulation           TEXT,
    experiment           TEXT,
    time                 INTEGER,
    flux                 REAL,
    thickness            REAL,
    velocity             REAL,
    drag                 REAL,
    surface              REAL,
    base                 REAL,
    bed                  REAL,
    flotaison            REAL,
    R_drag               REAL,
    driving_stress       REAL,
    slope_flux           REAL,
    slope_max            REAL,
    viscosity            REAL,
    buttressing          REAL,
    buttressing_natural  REAL,
    n_simulations        INTEGER,
    n_timesteps          INTEGER
);
"""


# MAIN
def build_basin_databases(reso: int) -> None:
    db_path = config.get_db_path(reso)
    os.makedirs(config.BASIN_DB_DIR, exist_ok=True)

    print("Loading basin grid …", flush=True)
    basin_id, x_coords, y_coords = _load_basin_grid(config.NC_BASINS)
    basin_xy = _basin_xy_sets(basin_id, x_coords, y_coords)

    print("Loading last_grounded_point …", flush=True)
    conn_src = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM last_grounded_point", conn_src)
    conn_src.close()

    for basin_idx, xy_set in sorted(basin_xy.items()):
        mask     = df.apply(lambda row: (row["x"], row["y"]) in xy_set, axis=1)
        df_basin = df[mask]

        if df_basin.empty:
            print(f"  Basin {basin_idx:02d}: no data, skipping.", flush=True)
            continue

        basin_db = os.path.join(config.BASIN_DB_DIR, f"basin_{basin_idx:02d}_{reso}km.db")
        conn_dst = sqlite3.connect(basin_db)
        cursor = conn_dst.cursor()
        cursor.execute(CREATE_BASIN_TABLE)
        df_basin.to_sql("basin_data", conn_dst, if_exists="replace", index=False)
        conn_dst.close()

        print(f"  Basin {basin_idx:02d}: {len(df_basin)} rows → {basin_db}", flush=True)


for reso in config.RESOLUTIONS:
    print(f"\n=== Resolution {reso} km ===", flush=True)
    build_basin_databases(reso)

print("Done.", flush=True)
