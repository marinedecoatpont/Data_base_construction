"""
05_basin_analysis.py
====================
Three-step pipeline for analysing grounding-line flux per Zwally basin.

  Step 1 — extract_basins_to_csv()
      Reads last_grounded_point from the main SQLite database, assigns each
      (x, y) point to its Zwally basin, and writes one CSV per basin.
      flux_bed (BedMachine) is joined from the BedMachine_GL database.

  Step 2 — compute_theoretical_fluxes()
      For each basin CSV, computes Weertman (m=1, m=3) and Coulomb/Tsai
      theoretical fluxes and appends the results to the CSV.

Usage
-----
  python 05_basin_analysis.py            # all steps
  python 05_basin_analysis.py --step 1   # extraction only
  python 05_basin_analysis.py --step 2   # theoretical fluxes only

Optional overrides
  --db      path to SQLite database (simulations)
  --db-bed  path to BedMachine_GL SQLite database
  --nc      path to basin NetCDF
  --outdir  path for output CSVs
"""

import argparse
import copy
import glob
import os
import sqlite3
import time
import warnings

import numpy as np
import pandas as pd
import xarray as xr

import config

warnings.filterwarnings("ignore")


# DEFAULT PIPELINE CONFIGURATION
PIPELINE_CFG = dict(
    db_path     = config.get_db_path(16),
    db_bed      = os.path.join(config.SAVE_PATH, "BedMachine_GL_16km.db"),
    nc_basins   = config.NC_BASINS,
    csv_all     = os.path.join(config.SAVE_PATH, "data_16km_buttressing.csv"),
    simulations = config.SIMULATIONS,
    out_dir     = config.BASIN_CSV_DIR,
    rho_ice     = config.RHO_ICE,
    rho_water   = config.RHO_WATER,
    gravity     = config.GRAVITY,
    glen_n      = config.GLEN_N,
    coulomb_f   = config.COULOMB_F,
    gl_width_km = 16,
    year_in_sec = config.YEAR_S,
    fig_dpi     = 300,
    ref_year    = 2015,
)


# =============================================================================
# PHYSICAL FUNCTIONS
# =============================================================================

def friction_coefficient(basal_drag: np.ndarray, velocity: np.ndarray, m: int = 3) -> np.ndarray:
    v = copy.deepcopy(velocity).astype(float)
    v[v < 1e-12] = 1e-12
    return np.abs(basal_drag) / v ** (1.0 / m)


def weertman_flux(thickness, viscosity, friction, buttressing,
                  n=3, m=1, rho=917, g=9.81, rhow=1028) -> np.ndarray:
    year_s      = 365.25 * 24 * 3600
    exp_den     = 1 + 1.0 / m
    grav_factor = (
        (rho * g) ** ((n + 1) / exp_den)
        * (1 - rho / rhow) ** (n / exp_den)
        * 4 ** (-(n / exp_den))
    )
    fluidity_exp = ((1.0 / viscosity) ** n) ** (1.0 / exp_den)
    beta_si      = (friction * year_s ** (1.0 / m)) ** (-(1.0 / (m + 1)))
    flux         = (
        beta_si
        * fluidity_exp
        * thickness ** ((n + 3 + 1.0 / m) / exp_den)
        * buttressing ** (n / (1.0 / m + 1))
    )
    return grav_factor * flux


def coulomb_flux(thickness, viscosity, buttressing,
                 n=3, rho=917, g=9.81, rhow=1028, f=0.6) -> np.ndarray:
    Q0          = 0.61
    grav_factor = 8 * (rho * g) ** n * (1 - rho / rhow) ** (n - 1) * 4 ** (-n)
    fluidity    = (1.0 / viscosity) ** n
    return grav_factor * Q0 * fluidity * thickness ** (n + 2) * f**-1 * buttressing


def flux_scaling(gl_width_km: float, rho_ice: float, year_in_sec: float) -> float:
    return gl_width_km * 1e3 * rho_ice * 1e-12 * year_in_sec


# =============================================================================
# STEP 1 — EXTRACTION
# =============================================================================

def _load_basin_grid(nc_path: str) -> tuple:
    ds       = xr.open_dataset(nc_path)
    basin_id = ds["Basin_ID"].values
    x_coords = ds["x"].values
    y_coords = ds["y"].values
    ds.close()
    return basin_id, x_coords, y_coords


def _query_db(db_path: str, simulations: list) -> pd.DataFrame:
    sims_str = ", ".join(f"'{s}'" for s in simulations)
    query    = f"""
        SELECT x, y, flux, simulation, time
        FROM   last_grounded_point
        WHERE  simulation IN ({sims_str})
    """
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(query, conn)
    conn.close()
    # Round coordinates to match BedMachine lookup
    df["x"] = df["x"].round(0)
    df["y"] = df["y"].round(0)
    return df


def _load_flux_bed(db_bed: str) -> pd.DataFrame:
    """
    Loads (x, y, flux_bed) from the BedMachine_GL database produced by
    04_build_bedmachine_database.py.
    One unique row per pixel — rounds coordinates to match simulation grid.
    """
    conn = sqlite3.connect(db_bed)
    df   = pd.read_sql("SELECT x, y, flux_bed FROM bedmachine_gl", conn)
    conn.close()

    df["x"] = df["x"].round(0)
    df["y"] = df["y"].round(0)
    df = df.drop_duplicates(subset=["x", "y"])[["x", "y", "flux_bed"]]
    print(f"  {len(df):,} unique BedMachine GL pixels loaded.")
    return df


def _pivot_simulations(df_simu: pd.DataFrame, simulations: list) -> pd.DataFrame:
    df_agg = (
        df_simu
        .groupby(["x", "y", "simulation"], as_index=False)
        .agg(flux=("flux", "mean"), time=("time", "mean"))
    )
    flux_wide = df_agg.pivot_table(
        index=["x", "y"], columns="simulation", values="flux", aggfunc="mean"
    ).reset_index()
    time_wide = df_agg.pivot_table(
        index=["x", "y"], columns="simulation", values="time", aggfunc="mean"
    ).reset_index()

    flux_wide.columns = ["x", "y"] + [f"flux_{s}" for s in flux_wide.columns[2:]]
    time_wide.columns = ["x", "y"] + [f"time_{s}" for s in time_wide.columns[2:]]

    merged = flux_wide.merge(time_wide, on=["x", "y"])
    for s in simulations:
        for prefix in ("flux_", "time_"):
            col = f"{prefix}{s}"
            if col not in merged.columns:
                merged[col] = np.nan
    return merged


def _build_basin_lookup(basin_id, x_coords, y_coords) -> dict:
    basins    = {}
    basin_ids = np.unique(basin_id[~np.isnan(basin_id.astype(float))]).astype(int)
    print(basin_ids)
    for i in basin_ids:
        rows, cols = np.where(basin_id == i)
        basins[i]  = pd.DataFrame({
            "x": x_coords[cols].round(0),
            "y": y_coords[rows].round(0),
        })
    return basins


def extract_basins_to_csv(cfg: dict) -> None:
    t0 = time.perf_counter()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    print("Loading basin grid …")
    basin_id, x_coords, y_coords = _load_basin_grid(cfg["nc_basins"])

    print("Querying simulation database …")
    df_simu = _query_db(cfg["db_path"], cfg["simulations"])
    print(f"  {len(df_simu):,} rows retrieved.")

    df_wide = _pivot_simulations(df_simu, cfg["simulations"])

    # Load BedMachine flux_bed and join onto simulation pixels
    print("Loading BedMachine flux_bed …")
    df_bed  = _load_flux_bed(cfg["db_bed"])
    df_wide = df_wide.merge(df_bed, on=["x", "y"], how="left")
    n_matched = df_wide["flux_bed"].notna().sum()
    print(f"  {n_matched:,} simulation pixels matched with a BedMachine flux.")

    basins = _build_basin_lookup(basin_id, x_coords, y_coords)

    for i, df_basin in sorted(basins.items()):
        df_out = df_basin.merge(df_wide, on=["x", "y"], how="left")

        col_order = ["x", "y"]
        for s in cfg["simulations"]:
            col_order += [f"flux_{s}", f"time_{s}"]
        col_order.append("flux_bed")

        for c in col_order:
            if c not in df_out.columns:
                df_out[c] = np.nan
        df_out = df_out[col_order]

        out_path = os.path.join(cfg["out_dir"], f"basin_{i:02d}.csv")
        df_out.to_csv(out_path, index=False)
        n_bed = df_out["flux_bed"].notna().sum()
        print(f"  Basin {i:2d} → {len(df_out):4d} pts  ({n_bed} with flux_bed) → {out_path}")

    print(f"Step 1 done in {time.perf_counter() - t0:.1f}s\n")


# =============================================================================
# STEP 2 — THEORETICAL FLUXES
# =============================================================================

def compute_theoretical_fluxes(cfg: dict) -> None:
    t0 = time.perf_counter()
    print("Loading full CSV for theoretical fluxes …")
    data = pd.read_csv(cfg["csv_all"])

    thickness  = data["thickness"].values
    viscosity  = data["viscosity"].values
    basal_drag = data["drag"].values
    velocity_b = data["velocity_base"].values
    theta      = np.clip(data["buttressing_natural"].values, 0, 1)
    scaling    = flux_scaling(cfg["gl_width_km"], cfg["rho_ice"], cfg["year_in_sec"])

    c1 = np.clip(friction_coefficient(basal_drag, velocity_b, m=1), 1e-2, None)
    c3 = np.clip(friction_coefficient(basal_drag, velocity_b, m=3), 1e-2, None)

    print("Computing Weertman m=1 …")
    w1_theta = weertman_flux(thickness, viscosity, c1, theta, n=cfg["glen_n"], m=1,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling
    w1_no_b  = weertman_flux(thickness, viscosity, c1, 1,     n=cfg["glen_n"], m=1,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling

    print("Computing Weertman m=3 …")
    w3_theta = weertman_flux(thickness, viscosity, c3, theta, n=cfg["glen_n"], m=3,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling
    w3_no_b  = weertman_flux(thickness, viscosity, c3, 1,     n=cfg["glen_n"], m=3,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling

    print("Computing Coulomb/Tsai …")
    c_theta  = coulomb_flux(thickness, viscosity, theta, n=cfg["glen_n"],
                            rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"],
                            f=cfg["coulomb_f"]) * scaling
    c_no_b   = coulomb_flux(thickness, viscosity, 1,     n=cfg["glen_n"],
                            rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"],
                            f=cfg["coulomb_f"]) * scaling

    for arr in [w1_theta, w1_no_b, w3_theta, w3_no_b, c_theta, c_no_b]:
        arr[arr < 1e-6] = 1e-6
        arr = np.nan_to_num(arr)

    df_theo = pd.DataFrame({
        "x"             : data["x"].values.round(0),
        "y"             : data["y"].values.round(0),
        "w1_theta"      : w1_theta,
        "w1_no_butt"    : w1_no_b,
        "w3_theta"      : w3_theta,
        "w3_no_butt"    : w3_no_b,
        "coulomb_theta" : c_theta,
        "coulomb_no_b"  : c_no_b,
        "buttressing"   : theta,
        "thickness"     : thickness,
        "viscosity"     : viscosity,
    })

    csv_files = sorted(
        glob.glob(os.path.join(cfg["out_dir"], "basin_*.csv")),
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df["x"] = df["x"].round(0)
        df["y"] = df["y"].round(0)
        df.merge(df_theo, on=["x", "y"], how="left").to_csv(csv_path, index=False)
        print(f"  {os.path.basename(csv_path)} enriched.")

    print(f"Step 2 done in {time.perf_counter() - t0:.1f}s\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step",   type=int, choices=[1, 2], default=0)
    parser.add_argument("--db",     default=PIPELINE_CFG["db_path"])
    parser.add_argument("--db-bed", default=PIPELINE_CFG["db_bed"])
    parser.add_argument("--nc",     default=PIPELINE_CFG["nc_basins"])
    parser.add_argument("--outdir", default=PIPELINE_CFG["out_dir"])
    args = parser.parse_args()

    cfg = {
        **PIPELINE_CFG,
        "db_path"  : args.db,
        "db_bed"   : args.db_bed,
        "nc_basins": args.nc,
        "out_dir"  : args.outdir,
    }

    if args.step in (0, 1):
        print("=" * 60)
        print("STEP 1 — Basin extraction")
        print("=" * 60)
        extract_basins_to_csv(cfg)

    if args.step in (0, 2):
        print("=" * 60)
        print("STEP 2 — Theoretical fluxes")
        print("=" * 60)
        conn = sqlite3.connect(config.get_db_path(config.RESOLUTIONS[0]))
        query = (
            "SELECT simulation, experiment, time, x, y, thickness, velocity, "
            "viscosity, drag, flux, velocity_base, buttressing, "
            "buttressing_natural, velocity_normal FROM last_grounded_point"
        )
        df_R_flux = pd.read_sql(query, conn)
        conn.close()
        df_R_flux.to_csv(cfg["csv_all"])
        compute_theoretical_fluxes(cfg)


main()
