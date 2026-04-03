"""
05_basin_analysis.py
====================
Three-step pipeline for analysing grounding-line flux per Zwally basin.

  Step 1 — extract_basins_to_csv()
      Reads last_grounded_point from the main SQLite database, assigns each
      (x, y) point to its Zwally basin, and writes one CSV per basin.

  Step 2 — compute_theoretical_fluxes()
      For each basin CSV, computes Weertman (m=1, m=3) and Coulomb/Tsai
      theoretical fluxes and appends the results to the CSV.

  Step 3 — plot_all_basins()
      Generates a multi-panel figure comparing BedMachine, simulated
      and theoretical fluxes for every basin.

Usage
-----
  python 05_basin_analysis.py            # all three steps
  python 05_basin_analysis.py --step 1   # extraction only
  python 05_basin_analysis.py --step 2   # theoretical fluxes only
  python 05_basin_analysis.py --step 3   # figures only

Optional overrides
  --db     path to SQLite database
  --nc     path to basin NetCDF
  --outdir path for output CSVs
"""

import argparse
import copy
import glob
import math
import os
import sqlite3
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress, pearsonr

import config

warnings.filterwarnings("ignore")


# DEFAULT PIPELINE CONFIGURATION  (can be overridden from the command line)
PIPELINE_CFG = dict(
    db_path = config.get_db_path(16),
    nc_basins = config.NC_BASINS,
    csv_all = os.path.join(config.SAVE_PATH, "data_16km_buttressing.csv"),
    simulations = config.PLOT_SIMULATIONS,
    out_dir = config.BASIN_CSV_DIR,
    rho_ice = config.RHO_ICE,
    rho_water = config.RHO_WATER,
    gravity = config.GRAVITY,
    glen_n = config.GLEN_N,
    coulomb_f = config.COULOMB_F,
    gl_width_km = 16,
    year_in_sec = config.YEAR_S,
    fig_out = os.path.join(config.SAVE_PATH, "Flux_par_bassin.png"),
    fig_dpi = 300,
    ref_year = 2015,
)


# PHYSICAL FUNCTIONS
def friction_coefficient(basal_drag: np.ndarray,velocity: np.ndarray,m: int = 3) -> np.ndarray:
    """β = |τ_b| / u^(1/m)   (Pa m^(1/m) yr^(-1/m))"""
    v = copy.deepcopy(velocity).astype(float)
    v[v < 1e-12] = 1e-12
    return np.abs(basal_drag) / v ** (1.0 / m)


def weertman_flux(thickness, viscosity, friction, buttressing,n=3, m=1, rho=917, g=9.81, rhow=1028) -> np.ndarray:
    """Grounding-line flux from Schoof (2007) — Weertman sliding law."""
    year_s     = 365.25 * 24 * 3600
    exp_num    = n + 1
    exp_den    = 1 + 1.0 / m
    grav_factor = (
        (rho * g) ** (exp_num / exp_den)
        * (1 - rho / rhow) ** (n / exp_den)
        * 4 ** (-(n / exp_den))
    )
    fluidity      = (1.0 / viscosity) ** n
    fluidity_exp  = fluidity ** (1.0 / exp_den)
    beta_si       = (friction * year_s ** (1.0 / m)) ** (-(1.0 / (m + 1)))
    flux          = (
        beta_si
        * fluidity_exp
        * thickness ** ((n + 3 + 1.0 / m) / exp_den)
        * buttressing ** (n / (1.0 / m + 1))
    )
    return grav_factor * flux


def coulomb_flux(thickness, viscosity, buttressing,n=3, rho=917, g=9.81, rhow=1028, f=0.6) -> np.ndarray:
    """Grounding-line flux from Tsai et al. (2015) — Coulomb sliding law."""
    Q0          = 0.61
    grav_factor = 8 * (rho * g) ** n * (1 - rho / rhow) ** (n - 1) * 4 ** (-n)
    fluidity    = (1.0 / viscosity) ** n
    flux        = fluidity * thickness ** (n + 2)
    return grav_factor * Q0 * flux * f**-1 * buttressing


def flux_scaling(gl_width_km: float, rho_ice: float, year_in_sec: float) -> float:
    """Conversion factor: m² s-1 → Gt yr-1 for a given grid cell width."""
    return gl_width_km * 1e3 * rho_ice * 1e-12 * year_in_sec


# STEP 1 — EXTRACTION
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
    return df


def _pivot_simulations(df_simu: pd.DataFrame, simulations: list) -> pd.DataFrame:
    df_agg = (df_simu.groupby(["x", "y", "simulation"], as_index=False).agg(flux=("flux", "mean"), time=("time", "mean")))
    flux_wide = df_agg.pivot_table(index=["x", "y"], columns="simulation", values="flux", aggfunc="mean").reset_index()
    time_wide = df_agg.pivot_table(index=["x", "y"], columns="simulation", values="time", aggfunc="mean").reset_index()

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
    basins   = {}
    basin_ids = np.unique(basin_id[~np.isnan(basin_id.astype(float))]).astype(int)
    for i in basin_ids:
        rows, cols = np.where(basin_id == i)
        basins[i] = pd.DataFrame({"x": x_coords[cols], "y": y_coords[rows]})
    return basins


def extract_basins_to_csv(cfg: dict) -> None:
    t0 = time.perf_counter()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    print("Loading basin grid …")
    basin_id, x_coords, y_coords = _load_basin_grid(cfg["nc_basins"])

    print("Querying database …")
    df_simu = _query_db(cfg["db_path"], cfg["simulations"])
    print(f"  {len(df_simu):,} rows retrieved.")

    df_wide = _pivot_simulations(df_simu, cfg["simulations"])
    basins = _build_basin_lookup(basin_id, x_coords, y_coords)

    for i, df_basin in sorted(basins.items()):
        df_out = df_basin.merge(df_wide, on=["x", "y"], how="left")

        col_order = ["x", "y"]
        for s in cfg["simulations"]:
            col_order += [f"flux_{s}", f"time_{s}"]
        for c in col_order:
            if c not in df_out.columns:
                df_out[c] = np.nan
        df_out = df_out[col_order]

        out_path = os.path.join(cfg["out_dir"], f"basin_{i:02d}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"  Basin {i:2d} → {len(df_out):4d} pts → {out_path}")

    print(f"Step 1 done in {time.perf_counter() - t0:.1f}s\n")


# STEP 2 — THEORETICAL FLUXES
def compute_theoretical_fluxes(cfg: dict) -> None:
    t0 = time.perf_counter()
    print("Loading full CSV for theoretical fluxes …")
    data = pd.read_csv(cfg["csv_all"])

    thickness = data["thickness"].values
    viscosity = data["viscosity"].values
    basal_drag = data["drag"].values
    velocity_b = data["velocity_base"].values
    theta = np.clip(data["buttressing_natural"].values, 0, 1)
    scaling = flux_scaling(cfg["gl_width_km"], cfg["rho_ice"], cfg["year_in_sec"])

    c1 = np.clip(friction_coefficient(basal_drag, velocity_b, m=1), 1e-2, None)
    c3 = np.clip(friction_coefficient(basal_drag, velocity_b, m=3), 1e-2, None)

    print("Computing Weertman m=1 …")
    w1_theta = weertman_flux(thickness, viscosity, c1, theta, n=cfg["glen_n"], m=1, rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling
    w1_no_b = weertman_flux(thickness, viscosity, c1, 1, n=cfg["glen_n"], m=1, rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling

    print("Computing Weertman m=3 …")
    w3_theta = weertman_flux(thickness, viscosity, c3, theta, n=cfg["glen_n"], m=3, rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling
    w3_no_b = weertman_flux(thickness, viscosity, c3, 1, n=cfg["glen_n"], m=3, rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"]) * scaling

    print("Computing Coulomb/Tsai …")
    c_theta = coulomb_flux(thickness, viscosity, theta, n=cfg["glen_n"], rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"], f=cfg["coulomb_f"]) * scaling
    c_no_b = coulomb_flux(thickness, viscosity, 1, n=cfg["glen_n"], rho=cfg["rho_ice"], rhow=cfg["rho_water"], g=cfg["gravity"], f=cfg["coulomb_f"]) * scaling

    for arr in [w1_theta, w1_no_b, w3_theta, w3_no_b, c_theta, c_no_b]:
        arr[arr < 1e-6] = 1e-6
        arr = np.nan_to_num(arr)

    df_theo = pd.DataFrame({
        "x" : data["x"].values,
        "y" : data["y"].values,
        "w1_theta" : w1_theta,
        "w1_no_butt" : w1_no_b,
        "w3_theta" : w3_theta,
        "w3_no_butt" : w3_no_b,
        "coulomb_theta" : c_theta,
        "coulomb_no_b" : c_no_b,
        "buttressing" : theta,
        "thickness" : thickness,
        "viscosity" : viscosity,
    })

    csv_files = sorted(
        glob.glob(os.path.join(cfg["out_dir"], "basin_*.csv")),
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df.merge(df_theo, on=["x", "y"], how="left").to_csv(csv_path, index=False)
        print(f"  {os.path.basename(csv_path)} enriched.")

    print(f"Step 2 done in {time.perf_counter() - t0:.1f}s\n")


# STEP 3 — VISUALISATION
def _plot_one_basin(ax, df: pd.DataFrame, basin_id: int, cfg: dict) -> None:
    idx = np.arange(len(df))

    if "flux_bed" in df.columns:
        flux_bed = np.abs(df["flux_bed"].values) * cfg["rho_ice"] / 1e12
        ax.scatter(idx, flux_bed, s=5, c="rebeccapurple", label="BedMachine", zorder=5)
    else:
        flux_bed = None

    for simu, color in config.MODEL_COLORS.items():
        col_flux = f"flux_{simu}"
        col_time = f"time_{simu}"
        if col_flux not in df.columns:
            continue
        flux_s = df[col_flux].values
        time_s = np.nanmean(df[col_time].values) + cfg["ref_year"] if col_time in df.columns else np.nan
        diff   = np.nanmean(np.abs(flux_bed - flux_s)) if flux_bed is not None else np.nan
        ax.scatter(idx, flux_s, s=2, c=color, alpha=0.5,
                   label=f"{simu} (Δ={diff:.3f}, {time_s:.0f})")

    for col, (lbl, c) in {
        "w1_theta"      : ("W m=1 +butt", "steelblue"),
        "w3_theta"      : ("W m=3 +butt", "firebrick"),
        "coulomb_theta" : ("Tsai +butt",  "goldenrod"),
    }.items():
        if col in df.columns:
            ax.scatter(idx, df[col].values, s=1, c=c, alpha=0.3, marker="^", label=lbl)

    ax.set_title(f"Basin {basin_id}", fontsize=9)
    ax.set_xlabel("Index (x, y)", fontsize=7)
    ax.set_ylabel("Flux (Gt yr-1)", fontsize=7)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=5, markerscale=2)


def plot_all_basins(cfg: dict) -> None:
    t0 = time.perf_counter()
    csv_files = sorted(
        glob.glob(os.path.join(cfg["out_dir"], "basin_*.csv")),
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    if not csv_files:
        raise FileNotFoundError(f"No basin CSVs found in {cfg['out_dir']}. Run step 1 first.")

    ncols = 5
    nrows = math.ceil(len(csv_files) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    axes = axes.flatten()

    for k, csv_path in enumerate(csv_files):
        basin_id = int(csv_path.split("_")[-1].split(".")[0])
        _plot_one_basin(axes[k], pd.read_csv(csv_path), basin_id, cfg)

    for i in range(len(csv_files), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Grounding-line flux per basin — BedMachine vs Models vs Theory",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(cfg["fig_out"], dpi=cfg["fig_dpi"])
    plt.close()
    print(f"Figure saved: {cfg['fig_out']}")
    print(f"Step 3 done in {time.perf_counter() - t0:.1f}s\n")


# STATISTICS UTILITY
def compute_stats(x: np.ndarray, y: np.ndarray) -> tuple:
    """Returns (r_pearson, residual_std, slope) ignoring NaN values."""
    mask = ~np.isnan(x) & ~np.isnan(y)
    xv, yv = x[mask], y[mask]
    if len(xv) < 2:
        return np.nan, np.nan, np.nan
    r, _ = pearsonr(xv, yv)
    slope, intcpt, *_ = linregress(xv, yv)
    return r, float(np.std(yv - (slope * xv + intcpt))), float(slope)

def main():
    parser = argparse.ArgumentParser(description="Basin-level grounding-line flux analysis pipeline.")
    parser.add_argument("--step",type=int, choices=[1, 2, 3], default=0,help="Step to run (1=extract, 2=theory, 3=plot). Default: all.")
    parser.add_argument("--db", default=PIPELINE_CFG["db_path"],  help="SQLite database")
    parser.add_argument("--nc", default=PIPELINE_CFG["nc_basins"], help="Basin NetCDF")
    parser.add_argument("--outdir", default=PIPELINE_CFG["out_dir"],   help="CSV output dir")
    args = parser.parse_args()

    cfg = {**PIPELINE_CFG,
           "db_path" : args.db,
           "nc_basins": args.nc,
           "out_dir" : args.outdir}

    step = args.step

    if step in (0, 1):
        print("=" * 60)
        print("STEP 1 — Basin extraction")
        print("=" * 60)
        extract_basins_to_csv(cfg)

    if step in (0, 2):
        print("=" * 60)
        print("STEP 2 — Theoretical fluxes")
        print("=" * 60)
        compute_theoretical_fluxes(cfg)

    if step in (0, 3):
        print("=" * 60)
        print("STEP 3 — Visualisation")
        print("=" * 60)
        plot_all_basins(cfg)


main()
