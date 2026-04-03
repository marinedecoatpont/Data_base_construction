"""
04_build_bedmachine_database.py
================================
Creates a SQLite database containing only the rows from last_grounded_point
whose (x, y) coordinates match pixels on the BedMachine grounding line.

The grounding line is identified from the BedMachine mask field:
  mask == 1 → grounded ice
  mask == 0 → ocean / ice shelf
Grounding-line pixels are those where at least one immediate neighbour has
a different mask value (i.e. the transition zone).

The output database has the same schema as the per-basin databases and can be
used directly by basin_analysis.py for observational reference.
"""

import os
import sqlite3

import numpy as np
import pandas as pd
import xarray as xr

import config


# =============================================================================
# BEDMACHINE GROUNDING LINE DETECTION
# =============================================================================

def _get_bedmachine_gl_coords(bm_path: str,
                               target_x: np.ndarray,
                               target_y: np.ndarray) -> set:
    """
    Identifies grounding-line pixels in BedMachine, interpolates the mask
    onto the simulation grid, and returns a set of (x, y) tuples.

    Parameters
    ----------
    bm_path   : path to BedMachineAntarctica.nc
    target_x  : 1-D x coordinates of the simulation grid
    target_y  : 1-D y coordinates of the simulation grid
    """
    ds   = xr.open_dataset(bm_path)
    mask = ds["mask"]  # 0 = ocean/shelf, 1 = grounded, 2 = floating, 3 = rock

    # Interpolate onto the simulation grid (nearest neighbour)
    mask_interp = mask.interp(x=target_x, y=target_y, method="nearest")
    m = mask_interp.values.astype(int)

    # Grounding-line pixels: grounded cells adjacent to floating/ocean
    gl_mask = np.zeros_like(m, dtype=bool)
    grounded = (m == 1)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(grounded, (di, dj), axis=(0, 1))
        gl_mask |= (grounded & ~shifted)

    ds.close()
    rows, cols = np.where(gl_mask)
    return set(zip(target_x[cols].tolist(), target_y[rows].tolist()))


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

CREATE_BM_TABLE = """
CREATE TABLE IF NOT EXISTS bedmachine_gl (
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
    residence            INTEGER,
    n_simulations        INTEGER,
    n_timesteps          INTEGER
);
"""


# =============================================================================
# MAIN
# =============================================================================

def build_bedmachine_database(reso: int) -> None:
    db_src  = config.get_db_path(reso)
    db_out  = os.path.join(config.SAVE_PATH, f"BedMachine_GL_{reso}km.db")
    grid_nc = config.GRID_FILES[reso]

    # Reference grid coordinates
    ds_ref   = xr.open_dataset(grid_nc)
    target_x = ds_ref["x"].values
    target_y = ds_ref["y"].values
    ds_ref.close()

    print("Detecting BedMachine grounding-line pixels …", flush=True)
    gl_coords = _get_bedmachine_gl_coords(
        config.BEDMACHINE_FILE, target_x, target_y
    )
    print(f"  {len(gl_coords)} GL pixels found.", flush=True)

    print("Loading last_grounded_point …", flush=True)
    conn_src = sqlite3.connect(db_src)
    df       = pd.read_sql("SELECT * FROM last_grounded_point", conn_src)
    conn_src.close()

    mask    = df.apply(lambda row: (row["x"], row["y"]) in gl_coords, axis=1)
    df_gl   = df[mask].copy()
    print(f"  {len(df_gl)} rows match BedMachine GL.", flush=True)

    conn_dst = sqlite3.connect(db_out)
    cursor   = conn_dst.cursor()
    cursor.execute(CREATE_BM_TABLE)
    df_gl.to_sql("bedmachine_gl", conn_dst, if_exists="replace", index=False)
    conn_dst.close()

    print(f"  Output: {db_out}", flush=True)


if __name__ == "__main__":
    for reso in config.RESOLUTIONS:
        print(f"\n=== Resolution {reso} km ===", flush=True)
        build_bedmachine_database(reso)

    print("Done.", flush=True)
