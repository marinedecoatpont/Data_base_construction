"""
04_build_bedmachine_database.py
================================
Creates a SQLite database containing only the rows from last_grounded_point
whose (x, y) coordinates match non-zero pixels in the BedMachine grounding-line
flux file (ligroundf_bedmachine_all_test.nc).

A flux_bed column (BedMachine flux value) is added to each matched row.

FIX: the join between simulation rows and BedMachine pixels is now done with a
pandas merge on rounded coordinates instead of a slow row-by-row apply().
This avoids floating-point mismatches that caused zero rows to be matched.
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
    ref       = xr.open_dataset(config.GRID_FILES[reso])
    da_interp = da.interp(x=ref.x, y=ref.y)
    ref.close()
    return da_interp


def _load_bedmachine_flux(bm_flux_path: str, reso: int) -> xr.DataArray:
    """
    Opens the BedMachine grounding-line flux file and interpolates it
    onto the simulation grid.
    Returns a DataArray with NaN everywhere the flux is zero.
    """
    ds         = xr.open_dataset(bm_flux_path)
    flux_bed   = ds.ligroundf
    flux_bed   = xr.where(flux_bed != 0, flux_bed, np.nan)
    flux_interp = _grid(flux_bed, reso)
    ds.close()
    return flux_interp


def _extract_gl_points(flux_bed: xr.DataArray) -> pd.DataFrame:
    """
    Returns a DataFrame with columns (x, y, flux_bed) for all non-zero
    BedMachine GL pixels.
    Coordinates are rounded to the nearest metre to avoid float drift.
    """
    mask   = np.isfinite(flux_bed.values) & (flux_bed.values != 0)
    ii, jj = np.where(mask)
    xf     = flux_bed.x.values[jj]
    yf     = flux_bed.y.values[ii]
    flux_f = flux_bed.values[ii, jj]

    df = pd.DataFrame({"x": xf, "y": yf, "flux_bed": flux_f})
    # Round to nearest metre so the merge with simulation coords is exact
    df["x"] = df["x"].round(0)
    df["y"] = df["y"].round(0)
    return df


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
    flux_bed   = _load_bedmachine_flux(bm_flux_path, reso)
    df_bed     = _extract_gl_points(flux_bed)
    print(f"  {len(df_bed)} non-zero BedMachine GL pixels found.", flush=True)

    print("Loading last_grounded_point …", flush=True)
    conn_src = sqlite3.connect(db_src)
    df       = pd.read_sql("SELECT * FROM last_grounded_point", conn_src)
    conn_src.close()
    print(f"  {len(df):,} rows in last_grounded_point.", flush=True)

    # Round simulation coordinates the same way so the merge is exact
    df["x"] = df["x"].round(0)
    df["y"] = df["y"].round(0)

    # Join: keeps only simulation rows that have a BedMachine GL pixel
    df_gl = df.merge(df_bed, on=["x", "y"], how="inner")
    print(f"  {len(df_gl):,} simulation rows match BedMachine GL coordinates.", flush=True)

    if df_gl.empty:
        print("  WARNING: no matching rows — check coordinate systems / resolution.", flush=True)

    conn_dst = sqlite3.connect(db_out)
    # Use if_exists="replace" so the table is always rebuilt cleanly
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
