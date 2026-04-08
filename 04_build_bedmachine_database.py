"""
04_build_bedmachine_database.py
================================
Creates a SQLite database containing only the rows from last_grounded_point
whose (x, y) coordinates match non-zero pixels in the BedMachine grounding-line
flux file (ligroundf_bedmachine_all_test.nc).

The BedMachine flux field is used directly as the reference: pixels where
ligroundf != 0 define the observed grounding line. The script then filters
last_grounded_point to those coordinates and stores the result in a new database
whose schema is identical to the per-basin databases.

A flux_bed column (BedMachine flux value) is added to each matched row.
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

def _grid(da: xr.DataArray, reso: int) -> xr.DataArray:
    """Interpolates da onto the reference simulation grid for a given resolution."""
    ref = xr.open_dataset(config.GRID_FILES[reso])
    da_interp = da.interp(x=ref.x, y=ref.y)
    ref.close()
    return da_interp


def _load_bedmachine_flux(bm_flux_path: str, reso: int) -> xr.DataArray:
    """
    Opens the BedMachine grounding-line flux file and interpolates it
    onto the simulation grid.
    Returns a DataArray with NaN everywhere the flux is zero.
    """
    ds = xr.open_dataset(bm_flux_path)
    flux_bed = ds.ligroundf
    flux_bed = xr.where(flux_bed != 0, flux_bed, np.nan)
    flux_interp = _grid(flux_bed, reso)
    ds.close()
    return flux_interp


def _extract_gl_points(flux_bed: xr.DataArray) -> tuple:
    """
    Returns arrays (x, y, flux) for all non-zero BedMachine GL pixels.
    """
    mask   = np.isfinite(flux_bed.values) & (flux_bed.values != 0)
    ii, jj = np.where(mask)
    xf     = flux_bed.x.values[jj]
    yf     = flux_bed.y.values[ii]
    flux_f = flux_bed.values[ii, jj]
    return xf, yf, flux_f


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
    n_timesteps          INTEGER,
    flux_bed             REAL
);
"""


# =============================================================================
# MAIN
# =============================================================================

def build_bedmachine_database(reso: int,
                               bm_flux_path: str,
                               db_src: str,
                               db_out: str) -> None:
    """
    Filters last_grounded_point to BedMachine GL pixels and writes the
    result (with an added flux_bed column) to db_out.
    """
    print(f"Loading BedMachine flux from {bm_flux_path} …", flush=True)
    flux_bed = _load_bedmachine_flux(bm_flux_path, reso)
    xf, yf, flux_f = _extract_gl_points(flux_bed)
    print(f"  {len(xf)} non-zero BedMachine GL pixels found.", flush=True)

    # Build a lookup: (x, y) -> flux_bed value
    bed_lookup = {(float(x), float(y)): float(f) for x, y, f in zip(xf, yf, flux_f)}

    print("Loading last_grounded_point …", flush=True)
    conn_src = sqlite3.connect(db_src)
    df       = pd.read_sql("SELECT * FROM last_grounded_point", conn_src)
    conn_src.close()

    # Keep only rows whose (x, y) appear in the BedMachine GL pixel set
    mask   = df.apply(lambda row: (row["x"], row["y"]) in bed_lookup, axis=1)
    df_gl  = df[mask].copy()
    df_gl["flux_bed"] = df_gl.apply(
        lambda row: bed_lookup[(row["x"], row["y"])], axis=1
    )
    print(f"  {len(df_gl)} simulation rows match BedMachine GL coordinates.", flush=True)

    conn_dst = sqlite3.connect(db_out)
    cursor   = conn_dst.cursor()
    cursor.execute(CREATE_BM_TABLE)
    df_gl.to_sql("bedmachine_gl", conn_dst, if_exists="replace", index=False)
    conn_dst.close()

    print(f"  Output database: {db_out}", flush=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    bm_flux_path = os.path.join(config.SAVE_PATH, "ligroundf_bedmachine_all_test.nc")

    for reso in config.RESOLUTIONS:
        db_src = config.get_db_path(reso)
        db_out = os.path.join(config.SAVE_PATH, f"BedMachine_GL_{reso}km.db")

        print(f"\n=== Resolution {reso} km ===", flush=True)
        build_bedmachine_database(reso, bm_flux_path, db_src, db_out)

    print("Done.", flush=True)
