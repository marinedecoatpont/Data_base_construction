"""
=============================================================================
PIPELINE D'ANALYSE DES FLUX PAR BASSIN GLACIAIRE (Zwally 16 km)
=============================================================================

Ce script est organisé en 3 étapes indépendantes :

  ÉTAPE 1 — extract_basins_to_csv()
      Lit la base SQLite de simulations, intersecte chaque point de grille
      avec son bassin Zwally, et exporte un CSV par bassin.

  ÉTAPE 2 — compute_theoretical_fluxes()
      Pour chaque CSV de bassin, calcule les flux théoriques de Schoof/Weertman
      (m=1 et m=3) et de Tsai/Coulomb, et enrichit le CSV.

  ÉTAPE 3 — plot_all_basins()
      Trace côte à côte les flux BedMachine, les flux simulés et les flux
      théoriques pour chaque bassin, et sauvegarde une figure multi-panneaux.

Chaque étape peut être lancée indépendamment (voir le bloc __main__).

Usage rapide :
    python basin_analysis_pipeline.py            # toutes les étapes
    python basin_analysis_pipeline.py --step 1  # extraction seulement
    python basin_analysis_pipeline.py --step 2  # flux théoriques seulement
    python basin_analysis_pipeline.py --step 3  # figures seulement
=============================================================================
"""

# ── Imports standard ─────────────────────────────────────────────────────────
import argparse
import copy
import glob
import math
import os
import sqlite3
import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress, pearsonr

warnings.filterwarnings("ignore")

print('test')
# =============================================================================
# CONFIGURATION — adaptez ces chemins à votre environnement
# =============================================================================

CFG = dict(
    # ── Fichiers d'entrée ───────────────────────────────────────────────────
    nc_basins       = "Basins_Zwally_16km.nc",        # fichier bassins NetCDF
    db_path         = "/Users/lebescom/DataBase_16km_test.db",  # base SQLite
    csv_all         = "data_16km_buttressing.csv",    # CSV complet (pour étape 2)

    # ── Simulations à extraire ──────────────────────────────────────────────
    simulations     = ["IGE_ElmerIce", "LSCE_GRISLI2", "PIK_PISM", "DC_ISSM", "NCAR_CISM1"],

    # ── Dossier de sortie des CSVs par bassin ──────────────────────────────
    out_dir         = "basins_csv",

    # ── Physique ────────────────────────────────────────────────────────────
    rho_ice         = 917.0,    # kg/m³
    rho_water       = 1028.0,   # kg/m³
    gravity         = 9.81,     # m/s²
    glen_n          = 3,        # exposant de Glen
    coulomb_f       = 0.6,      # coefficient de Coulomb
    gl_width_km     = 16,       # largeur de grille en km
    year_in_sec     = 365.25 * 24 * 3600,

    # ── Visualisation ───────────────────────────────────────────────────────
    fig_out         = "Flux_par_bassin.png",
    fig_dpi         = 300,
    ref_year        = 2015,     # offset temporel pour affichage
)

# Couleurs par modèle (cohérentes entre les figures)
MODEL_COLORS = {
    "IGE_ElmerIce" : "royalblue",
    "LSCE_GRISLI2" : "limegreen",
    "PIK_PISM"     : "indianred",
    "DC_ISSM"      : "deeppink",
    "NCAR_CISM1"   : "darkred",
}

# =============================================================================
# SECTION 1 — FONCTIONS PHYSIQUES
# =============================================================================

def friction_coefficient(basal_drag: np.ndarray, velocity: np.ndarray, m: int = 3) -> np.ndarray:
    """
    Convertit le frottement basal en coefficient de friction β via la loi :
        τ_b = β · u^(1/m)   →   β = |τ_b| / u^(1/m)

    Paramètres
    ----------
    basal_drag : Pa
    velocity   : m/a
    m          : exposant de Weertman (1 = linéaire, 3 = non-linéaire)

    Retourne
    --------
    β en Pa·m/a  (ou Pa·m/a^(1/m) selon m)
    """
    v = copy.deepcopy(velocity).astype(float)
    v[v < 1e-12] = 1e-12          # évite la division par zéro
    return np.abs(basal_drag) / v ** (1.0 / m)


def weertman_flux(thickness, viscosity, friction, buttressing,
                  n=3, m=1, rho=917, g=9.81, rhow=1028) -> np.ndarray:
    """
    Flux de ligne d'échouage selon Schoof (2007) — loi de frottement de Weertman.

    Paramètres
    ----------
    thickness   : épaisseur de glace (m)
    viscosity   : viscosité en Pa·s^(1/3)
    friction    : coefficient β en Pa·m/a
    buttressing : facteur θ ∈ [0, 1]
    n           : exposant de Glen
    m           : exposant de Weertman (1 ou 3)
    rho / rhow  : densités glace / eau (kg/m³)

    Retourne
    --------
    Flux en m²/s
    """
    year_s = 365.25 * 24 * 3600

    # Facteur gravitationnel (dépend de n et m)
    exp_num = n + 1
    exp_den = 1 + 1.0 / m
    grav_factor = (
        (rho * g) ** (exp_num / exp_den)
        * (1 - rho / rhow) ** (n / exp_den)
        * 4 ** (-(n / exp_den))
    )

    fluidity      = (1.0 / viscosity) ** n                  # Pa^-n · s^-1
    fluidity_exp  = fluidity ** (1.0 / exp_den)

    # Conversion friction Pa·m/a → Pa·s^(1/m)
    beta_si = (friction * year_s ** (1.0 / m)) ** (-(1.0 / (m + 1)))

    flux = (
        beta_si
        * fluidity_exp
        * thickness ** ((n + 3 + 1.0 / m) / exp_den)
        * buttressing ** (n / (1.0 / m + 1))
    )
    return grav_factor * flux


def coulomb_flux(thickness, viscosity, buttressing,
                 n=3, rho=917, g=9.81, rhow=1028, f=0.6) -> np.ndarray:
    """
    Flux de ligne d'échouage selon Tsai et al. (2015) — loi de Coulomb.

    Retourne
    --------
    Flux en m²/s
    """
    Q0   = 0.61
    grav_factor = 8 * (rho * g) ** n * (1 - rho / rhow) ** (n - 1) * 4 ** (-n)
    fluidity    = (1.0 / viscosity) ** n
    flux        = fluidity * thickness ** (n + 2)
    return grav_factor * Q0 * flux * f ** -1 * buttressing


def flux_scaling(gl_width_km: float, rho_ice: float, year_in_sec: float) -> float:
    """
    Facteur de mise à l'échelle pour convertir m²/s → Gt/an
    en supposant une largeur de grille uniforme.
    """
    return gl_width_km * 1e3 * rho_ice * 1e-12 * year_in_sec


# =============================================================================
# SECTION 2 — ÉTAPE 1 : EXTRACTION DES BASSINS → CSVs
# =============================================================================

def _load_basin_grid(nc_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Charge le fichier NetCDF des bassins et retourne (Basin_ID, x, y)
    sur la grille 2D.
    """
    ds        = xr.open_dataset(nc_path)
    basin_id  = ds["Basin_ID"].values    # shape (ny, nx)
    x_coords  = ds["x"].values           # 1D
    y_coords  = ds["y"].values           # 1D
    ds.close()
    return basin_id, x_coords, y_coords


def _query_db(db_path: str, simulations: list[str]) -> pd.DataFrame:
    """
    Extrait depuis la base SQLite les colonnes utiles pour tous les modèles.
    Utilise un seul SELECT avec pivot manuel côté Python.
    """
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


def _pivot_simulations(df_simu: pd.DataFrame,
                       simulations: list[str]) -> pd.DataFrame:
    """
    Pivote le DataFrame long (x, y, simulation, flux, time)
    en DataFrame large avec une colonne flux_<simu> et time_<simu>
    par simulation. Beaucoup plus rapide que la boucle imbriquée originale.
    """
    # Agrégation : moyenne si plusieurs entrées pour (x, y, simulation)
    df_agg = (
        df_simu
        .groupby(["x", "y", "simulation"], as_index=False)
        .agg(flux=("flux", "mean"), time=("time", "mean"))
    )

    # Pivot en colonnes
    flux_wide = df_agg.pivot_table(
        index=["x", "y"], columns="simulation",
        values="flux", aggfunc="mean"
    ).reset_index()
    time_wide = df_agg.pivot_table(
        index=["x", "y"], columns="simulation",
        values="time", aggfunc="mean"
    ).reset_index()

    # Renommage propre
    flux_wide.columns = (
        ["x", "y"] + [f"flux_{s}" for s in flux_wide.columns[2:]]
    )
    time_wide.columns = (
        ["x", "y"] + [f"time_{s}" for s in time_wide.columns[2:]]
    )

    merged = flux_wide.merge(time_wide, on=["x", "y"])

    # S'assurer que toutes les colonnes attendues existent
    for s in simulations:
        for prefix in ("flux_", "time_"):
            col = f"{prefix}{s}"
            if col not in merged.columns:
                merged[col] = np.nan

    return merged


def _build_basin_xy_lookup(
    basin_id: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    bedmachine_flux: xr.DataArray | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Pour chaque bassin i ∈ [1..26], retourne un DataFrame
    avec les colonnes (x, y [, flux_bed]).
    """
    basins = {}
    basin_ids = np.unique(basin_id[~np.isnan(basin_id.astype(float))]).astype(int)

    for i in basin_ids:
        mask_2d = basin_id == i
        rows, cols = np.where(mask_2d)

        x_pts = x_coords[cols]
        y_pts = y_coords[rows]

        df = pd.DataFrame({"x": x_pts, "y": y_pts})

        if bedmachine_flux is not None:
            # Extraction du flux BedMachine aux points du bassin
            flux_vals = bedmachine_flux.values[rows, cols]
            df["flux_bed"] = flux_vals
        basins[i] = df

    return basins


def extract_basins_to_csv(cfg: dict) -> None:
    """
    ÉTAPE 1 — Extraction

    Pour chaque bassin Zwally (1–26) :
    1. Identifie les points de grille appartenant au bassin.
    2. Récupère les flux de tous les modèles en une seule jointure vectorisée.
    3. Exporte un CSV enrichi dans cfg['out_dir'].
    """
    t0 = time.perf_counter()
    os.makedirs(cfg["out_dir"], exist_ok=True)

    print("⟳  Chargement de la grille des bassins…")
    basin_id, x_coords, y_coords = _load_basin_grid(cfg["nc_basins"])

    print("⟳  Requête sur la base de données…")
    df_simu = _query_db(cfg["db_path"], cfg["simulations"])
    print(f"   {len(df_simu):,} lignes récupérées.")

    print("⟳  Pivot des simulations (vectorisé)…")
    df_wide = _pivot_simulations(df_simu, cfg["simulations"])

    print("⟳  Construction du lookup spatial…")
    basins = _build_basin_xy_lookup(basin_id, x_coords, y_coords)

    print("⟳  Jointure bassin × simulations et export CSV…")
    basin_ids = sorted(basins.keys())
    for i in basin_ids:
        df_basin = basins[i]

        # Jointure sur (x, y) — pas de boucle pixel à pixel
        df_out = df_basin.merge(df_wide, on=["x", "y"], how="left")

        # Colonnes dans un ordre cohérent
        col_order = ["x", "y"]
        for s in cfg["simulations"]:
            col_order += [f"flux_{s}", f"time_{s}"]
        # Ajouter les colonnes manquantes éventuelles
        for c in col_order:
            if c not in df_out.columns:
                df_out[c] = np.nan
        df_out = df_out[col_order]

        out_path = os.path.join(cfg["out_dir"], f"basin_{i:02d}.csv")
        df_out.to_csv(out_path, index=False)
        print(f"   Bassin {i:2d} → {len(df_out):4d} points → {out_path}")

    print(f"✓  Étape 1 terminée en {time.perf_counter() - t0:.1f}s\n")


# =============================================================================
# SECTION 3 — ÉTAPE 2 : CALCUL DES FLUX THÉORIQUES
# =============================================================================

def compute_theoretical_fluxes(cfg: dict) -> None:
    """
    ÉTAPE 2 — Flux théoriques

    Lit le CSV global (data_16km_buttressing.csv) pour avoir viscosity,
    basal_drag, thickness, etc., calcule les flux Weertman (m=1, m=3)
    et Coulomb, puis enrichit chaque CSV de bassin avec ces colonnes.
    """
    t0 = time.perf_counter()

    print("⟳  Chargement du CSV complet pour les flux théoriques…")
    data = pd.read_csv(cfg["csv_all"])

    # ── Préparation des variables ────────────────────────────────────────────
    thickness   = data["thickness"].values
    viscosity   = data["viscosity"].values
    basal_drag  = data["drag"].values
    velocity_b  = data["velocity_base"].values
    velocity_m  = data["velocity"].values

    # Buttressing clampé entre 0 et 1
    theta = np.clip(data["buttressing_natural"].values, 0, 1)

    scaling = flux_scaling(cfg["gl_width_km"], cfg["rho_ice"], cfg["year_in_sec"])

    # ── Coefficients de friction ─────────────────────────────────────────────
    c1 = friction_coefficient(basal_drag, velocity_b, m=1)
    c3 = friction_coefficient(basal_drag, velocity_b, m=3)
    c1 = np.clip(c1, 1e-2, None)
    c3 = np.clip(c3, 1e-2, None)

    print("⟳  Calcul des flux théoriques Weertman m=1…")
    w1_theta = weertman_flux(thickness, viscosity, c1, theta,
                             n=cfg["glen_n"], m=1,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"],
                             g=cfg["gravity"]) * scaling

    w1_no_b  = weertman_flux(thickness, viscosity, c1, 1,
                             n=cfg["glen_n"], m=1,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"],
                             g=cfg["gravity"]) * scaling

    print("⟳  Calcul des flux théoriques Weertman m=3…")
    w3_theta = weertman_flux(thickness, viscosity, c3, theta,
                             n=cfg["glen_n"], m=3,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"],
                             g=cfg["gravity"]) * scaling

    w3_no_b  = weertman_flux(thickness, viscosity, c3, 1,
                             n=cfg["glen_n"], m=3,
                             rho=cfg["rho_ice"], rhow=cfg["rho_water"],
                             g=cfg["gravity"]) * scaling

    print("⟳  Calcul des flux théoriques Coulomb (Tsai)…")
    c_theta  = coulomb_flux(thickness, viscosity, theta,
                            n=cfg["glen_n"],
                            rho=cfg["rho_ice"], rhow=cfg["rho_water"],
                            g=cfg["gravity"], f=cfg["coulomb_f"]) * scaling

    c_no_b   = coulomb_flux(thickness, viscosity, 1,
                            n=cfg["glen_n"],
                            rho=cfg["rho_ice"], rhow=cfg["rho_water"],
                            g=cfg["gravity"], f=cfg["coulomb_f"]) * scaling

    # Clipping pour éviter les valeurs irréalistes
    for arr in [w1_theta, w1_no_b, w3_theta, w3_no_b, c_theta, c_no_b]:
        arr[arr < 1e-6] = 1e-6
        arr = np.nan_to_num(arr)

    # ── Construction du DataFrame de référence théorique (x, y comme clé) ───
    df_theo = pd.DataFrame({
        "x"            : data["x"].values,
        "y"            : data["y"].values,
        "w1_theta"     : w1_theta,
        "w1_no_butt"   : w1_no_b,
        "w3_theta"     : w3_theta,
        "w3_no_butt"   : w3_no_b,
        "coulomb_theta": c_theta,
        "coulomb_no_b" : c_no_b,
        "buttressing"  : theta,
        "thickness"    : thickness,
        "viscosity"    : viscosity,
    })

    print("⟳  Enrichissement des CSVs de bassin…")
    csv_files = sorted(
        glob.glob(os.path.join(cfg["out_dir"], "basin_*.csv")),
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        df_enriched = df.merge(df_theo, on=["x", "y"], how="left")
        df_enriched.to_csv(csv_path, index=False)
        print(f"   {os.path.basename(csv_path)} enrichi "
              f"({len(df_enriched)} points)")

    print(f"✓  Étape 2 terminée en {time.perf_counter() - t0:.1f}s\n")


# =============================================================================
# SECTION 4 — ÉTAPE 3 : VISUALISATION
# =============================================================================

def _plot_one_basin(ax: plt.Axes, df: pd.DataFrame,
                    basin_id: int, cfg: dict) -> None:
    """
    Trace sur `ax` les flux BedMachine, simulés et théoriques
    pour un bassin donné.
    """
    idx = np.arange(len(df))

    # ── Flux BedMachine (référence observationnelle) ──────────────────────
    if "flux_bed" in df.columns:
        flux_bed = np.abs(df["flux_bed"].values) * cfg["rho_ice"] / 1e12
        ax.scatter(idx, flux_bed, s=5, c="rebeccapurple",
                   label="BedMachine", zorder=5)

    # ── Flux simulés ──────────────────────────────────────────────────────
    for simu, color in MODEL_COLORS.items():
        col_flux = f"flux_{simu}"
        col_time = f"time_{simu}"
        if col_flux not in df.columns:
            continue
        flux_s = df[col_flux].values
        time_s = np.nanmean(df[col_time].values) + cfg["ref_year"] \
                 if col_time in df.columns else np.nan
        diff   = np.nanmean(np.abs(flux_bed - flux_s)) if "flux_bed" in df.columns \
                 else np.nan
        label  = f"{simu} (Δ={diff:.3f}, {time_s:.0f})"
        ax.scatter(idx, flux_s, s=2, c=color, alpha=0.5, label=label)

    # ── Flux théoriques ───────────────────────────────────────────────────
    theo_cols = {
        "w1_theta"     : ("W m=1 +butt", "--", "steelblue"),
        "w3_theta"     : ("W m=3 +butt", "--", "firebrick"),
        "coulomb_theta": ("Tsai +butt",  "--", "goldenrod"),
    }
    for col, (lbl, ls, c) in theo_cols.items():
        if col in df.columns:
            ax.scatter(idx, df[col].values, s=1, c=c, alpha=0.3,
                       marker="^", label=lbl)

    ax.set_title(f"Bassin {basin_id}", fontsize=9)
    ax.set_xlabel("Index (x, y)", fontsize=7)
    ax.set_ylabel("Flux (Gt/an)", fontsize=7)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=5, markerscale=2)


def plot_all_basins(cfg: dict) -> None:
    """
    ÉTAPE 3 — Visualisation

    Charge chaque CSV de bassin et génère une figure multi-panneaux
    comparant BedMachine, les modèles et les flux théoriques.
    """
    t0 = time.perf_counter()

    csv_files = sorted(
        glob.glob(os.path.join(cfg["out_dir"], "basin_*.csv")),
        key=lambda f: int(f.split("_")[-1].split(".")[0])
    )
    if not csv_files:
        raise FileNotFoundError(
            f"Aucun fichier CSV trouvé dans {cfg['out_dir']}. "
            "Lancez d'abord l'étape 1."
        )

    ncols    = 5
    nrows    = math.ceil(len(csv_files) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    axes      = axes.flatten()

    for k, csv_path in enumerate(csv_files):
        basin_id = int(csv_path.split("_")[-1].split(".")[0])
        df       = pd.read_csv(csv_path)
        _plot_one_basin(axes[k], df, basin_id, cfg)

    # Supprime les axes vides
    for i in range(len(csv_files), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Flux par bassin glaciaire — BedMachine vs Modèles vs Théorie",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(cfg["fig_out"], dpi=cfg["fig_dpi"])
    plt.close()

    print(f"✓  Figure sauvegardée : {cfg['fig_out']}")
    print(f"✓  Étape 3 terminée en {time.perf_counter() - t0:.1f}s\n")


# =============================================================================
# SECTION 5 — STATISTIQUES (utilitaire optionnel)
# =============================================================================

def compute_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Calcule la corrélation de Pearson, la dispersion (std des résidus)
    et la pente d'une régression linéaire entre x et y (sans NaN).

    Retourne (r_pearson, dispersion, slope)
    """
    mask    = ~np.isnan(x) & ~np.isnan(y)
    x_v, y_v = x[mask], y[mask]
    if len(x_v) < 2:
        return np.nan, np.nan, np.nan

    r, _              = pearsonr(x_v, y_v)
    slope, intercept, *_ = linregress(x_v, y_v)
    residuals         = y_v - (slope * x_v + intercept)
    return r, float(np.std(residuals)), float(slope)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline analyse des flux glaciaires par bassin"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3], default=0,
        help="Étape à lancer (1=extraction, 2=théorique, 3=figures). "
             "Sans argument : toutes les étapes."
    )
    # Possibilité de surcharger les chemins depuis la ligne de commande
    parser.add_argument("--db",     default=CFG["db_path"],   help="Chemin SQLite")
    parser.add_argument("--nc",     default=CFG["nc_basins"], help="Fichier NetCDF bassins")
    parser.add_argument("--csv",    default=CFG["csv_all"],   help="CSV global")
    parser.add_argument("--outdir", default=CFG["out_dir"],   help="Dossier CSVs sortie")
    args = parser.parse_args()

    cfg = {**CFG,
           "db_path"  : args.db,
           "nc_basins": args.nc,
           "csv_all"  : args.csv,
           "out_dir"  : args.outdir}

    step = args.step

    if step in (0, 1):
        print("=" * 60)
        print("ÉTAPE 1 — Extraction des bassins")
        print("=" * 60)
        extract_basins_to_csv(cfg)

    if step in (0, 2):
        print("=" * 60)
        print("ÉTAPE 2 — Flux théoriques")
        print("=" * 60)
        compute_theoretical_fluxes(cfg)

    if step in (0, 3):
        print("=" * 60)
        print("ÉTAPE 3 — Visualisation")
        print("=" * 60)
        plot_all_basins(cfg)


if __name__ == "__main__":
    main()