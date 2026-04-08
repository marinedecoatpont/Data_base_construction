"""
config.py — Shared configuration for the grounding line flux pipeline.
All paths, physical constants and simulation lists are defined here.
"""

import os

# =============================================================================
# PATHS
# =============================================================================

# Root directory where NetCDF simulation files are stored
PATH_IF = "/Users/lebescom"

# Output directory for databases and results
SAVE_PATH = "/Users/lebescom/Documents/Code/Data_base_construction"

# Reference grid files (one per resolution)
GRID_FILES = {
    4:  "grid4x4.nc",
    16: "grid16x16.nc",
}

# Basins grid file (Zwally)
NC_BASINS = "Basins_Zwally_16km.nc"

# BedMachine NetCDF file
BEDMACHINE_FILE = "BedMachineAntarctica.nc"

# Output directories
BASIN_DB_DIR  = os.path.join(SAVE_PATH, "basin_databases")
BASIN_CSV_DIR = os.path.join(SAVE_PATH, "basins_csv")

# =============================================================================
# DATABASE PATHS  (one per resolution)
# =============================================================================

def get_db_path(reso: int) -> str:
    return os.path.join(SAVE_PATH, f"DataBase_{reso}km.db")


# =============================================================================
# RESOLUTIONS TO PROCESS
# =============================================================================

RESOLUTIONS = [16]

# =============================================================================
# SIMULATIONS  —  name : number_of_experiments
# =============================================================================

SIMULATIONS = {
    "DC_ISSM"                     : 6,
    "IGE_ElmerIce"                : 6,
    "ILTS_SICOPOLIS"              : 14,
    "LSCE_GRISLI"                 : 14,
    "LSCE_GRISLI2"                : 14,
    "NCAR_CISM1"                  : 6,
    "NORCE_CISM2-MAR364-ERA-t1"   : 6,
    "PIK_PISM"                    : 6,
    "UCM_Yelmo"                   : 14,
    "ULB_fETISh-KoriBU1"          : 14,
    "ULB_fETISh-KoriBU2"          : 14,
    "UTAS_ElmerIce"               : 6,
}

# Simulations used in basin analysis / plotting
PLOT_SIMULATIONS = [
    "IGE_ElmerIce",
    "LSCE_GRISLI2",
    "PIK_PISM",
    "DC_ISSM",
    "NCAR_CISM1",
]

# Colors per model (consistent across all figures)
MODEL_COLORS = {
    "IGE_ElmerIce" : "royalblue",
    "LSCE_GRISLI2" : "limegreen",
    "PIK_PISM"     : "indianred",
    "DC_ISSM"      : "deeppink",
    "NCAR_CISM1"   : "darkred",
}

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

RHO_ICE    = 917.0        # kg m-3
RHO_WATER  = 1027.0       # kg m-3
GRAVITY    = 9.81         # m s-2
GLEN_N     = 3            # Glen flow law exponent
COULOMB_F  = 0.6          # Coulomb friction coefficient
YEAR_S     = 365.25 * 24 * 3600   # seconds per year

# Effective gravitational driving factor  ρ_i g (1 - ρ_i/ρ_w)
GRAVITY_EFF = GRAVITY * (1.0 - RHO_ICE / RHO_WATER)

# Viscosity exponent  -1/n
EXPOS = -1.0 / GLEN_N

# =============================================================================
# TOTAL NUMBER OF TIME STEPS (per model)
# =============================================================================

TIMES_TOTAL        = 286
TIMES_TOTAL_GRISLI = 285   # LSCE_GRISLI has one fewer time step
