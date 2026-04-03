# Grounding-Line Ice Flux Pipeline

This repository builds and analyses a multi-model database of Antarctic grounding-line ice flux derived from the ISMIP6 AIS ensemble.

The pipeline reads NetCDF simulation outputs, computes physical quantities at the grounding line, stores them in SQLite databases, and enables comparison across models, basins and theoretical predictions.

---

## Repository structure

```
grounding_line_flux/
├── config.py                         # Shared paths, constants, simulation list
├── scripts/
│   ├── 01_build_flux_database.py     # Build the main flux_data table
│   ├── 02_build_derived_tables.py    # Build last_grounded_point and time_1_10
│   ├── 03_build_basin_databases.py   # Split last_grounded_point by Zwally basin
│   ├── 04_build_bedmachine_database.py  # Filter on BedMachine grounding line
│   └── 05_basin_analysis.py          # Theoretical fluxes + visualisation
└── docs/
    └── metadata.md                   # Database schema and column descriptions
```

Run the scripts in numerical order (01 → 02 → 03 / 04 → 05).

---

## Scripts

---

### `config.py`

#### General description
Centralises every path, physical constant and simulation metadata used by the rest of the pipeline. No computation is performed here.

#### Step-by-step
- Defines file paths for NetCDF inputs, grid files, BedMachine, basin grid, and database outputs.
- Provides a helper `get_db_path(reso)` that returns the database path for a given resolution.
- Stores the complete simulation dictionary (name → number of experiments).
- Defines all physical constants (ice density, water density, gravity, Glen exponent, …).

#### How to run
This file is imported by every other script; it is not run directly.

#### Inputs
None.

#### Outputs
None (configuration module only).

---

### `01_build_flux_database.py`

#### General description
Builds the main SQLite database for all resolutions. For every simulation / experiment / time step it identifies grounding-line pixels, computes physical quantities (flux, thickness, velocity, viscosity, buttressing, slope, …) and inserts the results into the `flux_data` table.

#### Step-by-step
1. Opens the reference grid file for the target resolution and reads the grid spacing `dx`.
2. Connects to (or creates) the SQLite database and creates the `flux_data` table if it does not exist.
3. Loops over simulations and experiments, loading the grounding-line flux field (`ligroundf`), ice thickness, velocity, basal drag, surface, base and bed topography.
4. Applies masks to keep only valid, non-zero grounding-line pixels.
5. Computes effective viscosity from the strain-rate tensor and the deviatoric-stress formulation.
6. Computes surface slope, driving stress, buoyancy coefficient and the drag ratio R.
7. Regrids all fields to the reference grid.
8. Computes the buttressing factor for each GL pixel from the stress tensor projected onto the outward-normal direction.
9. Batch-inserts all records and commits after each time step.

#### How to run
```bash
python scripts/01_build_flux_database.py           # full run
python scripts/01_build_flux_database.py --test    # small test (5 time steps, one simulation)
```

#### Inputs
- NetCDF grounding-line flux files: `{PATH_IF}/ligroundf_{simu}_{exp}.nc`
- NetCDF fields per variable: `{PATH_IF}/{var}_{simu}_{exp}.nc`
- Reference grid: `grid{reso}x{reso}.nc`

#### Outputs
- `{SAVE_PATH}/DataBase_{reso}km.db`  — SQLite database containing the `flux_data` table.

---

### `02_build_derived_tables.py`

#### General description
Creates two summary tables (`last_grounded_point` and `time_1_10`) from the existing `flux_data` table, and adds `residence`, `n_simulations` and `n_timesteps` statistics.

#### Step-by-step
1. Adds a `residence` column to `flux_data` counting how many time steps each (simulation, experiment, x, y) combination appears.
2. Creates `last_grounded_point` by extracting, for each (simulation, experiment, x, y), only the row with the latest time step.
3. Creates `time_1_10` by keeping only rows where `time % 10 == 0`.
4. For both new tables, computes per-(x, y) statistics: number of distinct simulations (`n_simulations`) and total number of rows (`n_timesteps`).

#### How to run
```bash
python scripts/02_build_derived_tables.py
```
Requires that `01_build_flux_database.py` has been run first.

#### Inputs
- `{SAVE_PATH}/DataBase_{reso}km.db`  (must contain `flux_data`)

#### Outputs
- Same database, with new tables `last_grounded_point` and `time_1_10`.

---

### `03_build_basin_databases.py`

#### General description
Splits `last_grounded_point` into one SQLite database per Zwally basin. Each output database contains a single table (`basin_data`) with only the rows whose (x, y) coordinates fall within that basin.

#### Step-by-step
1. Loads the Zwally basin grid from the NetCDF file and builds a spatial lookup: basin index → set of (x, y) coordinate pairs.
2. Reads `last_grounded_point` from the main database into a DataFrame.
3. For each basin, filters the DataFrame and writes a new `.db` file in `BASIN_DB_DIR`.

#### How to run
```bash
python scripts/03_build_basin_databases.py
```
Requires that `02_build_derived_tables.py` has been run first.

#### Inputs
- `{SAVE_PATH}/DataBase_{reso}km.db`  (must contain `last_grounded_point`)
- `Basins_Zwally_16km.nc`

#### Outputs
- `{BASIN_DB_DIR}/basin_{id:02d}_{reso}km.db`  — one database per basin (1–26).

---

### `04_build_bedmachine_database.py`

#### General description
Creates a database containing only the rows from `last_grounded_point` whose (x, y) coordinates match non-zero pixels in the BedMachine grounding-line flux file (`ligroundf_bedmachine_all_test.nc`). The BedMachine flux value is attached to each matched row as `flux_bed`, providing a direct observational reference at each simulated GL pixel.

#### Step-by-step
1. Opens `ligroundf_bedmachine_all_test.nc` and interpolates the flux field onto the simulation grid.
2. Masks zero values to NaN, then extracts (x, y, flux) for all non-zero pixels — these define the observed grounding line.
3. Builds a lookup dictionary `(x, y) → flux_bed`.
4. Reads `last_grounded_point` from the main database and keeps only rows whose coordinates appear in the lookup.
5. Adds a `flux_bed` column to the filtered rows and writes everything to a new SQLite database.

#### How to run
```bash
python scripts/04_build_bedmachine_database.py
```
Requires that `02_build_derived_tables.py` has been run first.

#### Inputs
- `{SAVE_PATH}/DataBase_{reso}km.db`  (must contain `last_grounded_point`)
- `{SAVE_PATH}/ligroundf_bedmachine_all_test.nc`
- `grid{reso}x{reso}.nc`

#### Outputs
- `{SAVE_PATH}/BedMachine_GL_{reso}km.db`  — SQLite database with `bedmachine_gl` table (same schema as `last_grounded_point` plus `flux_bed`).

---

### `05_basin_analysis.py`

#### General description
Three-step analysis pipeline: extracts per-basin CSVs, computes Weertman and Coulomb/Tsai theoretical fluxes, and generates a multi-panel comparison figure.

#### Step-by-step
**Step 1 — extract_basins_to_csv**
1. Loads the Zwally basin grid.
2. Queries `last_grounded_point` from the database for the selected simulations.
3. Pivots the data so each (x, y) row has one flux column per simulation.
4. Merges with the spatial basin lookup and writes one CSV per basin.

**Step 2 — compute_theoretical_fluxes**
1. Reads the full CSV (containing viscosity, thickness, drag, buttressing).
2. Derives friction coefficients β for Weertman m=1 and m=3.
3. Computes Weertman (m=1, m=3) and Coulomb/Tsai theoretical fluxes with and without buttressing.
4. Merges the theoretical columns into each per-basin CSV.

**Step 3 — plot_all_basins**
1. Loads each basin CSV.
2. Generates a log-scale scatter plot comparing BedMachine flux, simulated flux and theoretical fluxes for each basin.
3. Saves a multi-panel figure.

#### How to run
```bash
python scripts/05_basin_analysis.py               # all steps
python scripts/05_basin_analysis.py --step 1      # extraction only
python scripts/05_basin_analysis.py --step 2      # theoretical fluxes only
python scripts/05_basin_analysis.py --step 3      # figure only

# Override default paths:
python scripts/05_basin_analysis.py --db /path/to/db --nc /path/to/basins.nc --outdir /path/to/csvs
```

#### Inputs
- `{SAVE_PATH}/DataBase_{reso}km.db`  (must contain `last_grounded_point`)
- `Basins_Zwally_16km.nc`
- `{SAVE_PATH}/data_16km_buttressing.csv`  (for step 2)

#### Outputs
- `{BASIN_CSV_DIR}/basin_{id:02d}.csv`  — one CSV per basin (steps 1 & 2).
- `{SAVE_PATH}/Flux_par_bassin.png`  — multi-panel figure (step 3).

---

## Dependencies

```
numpy
xarray
pandas
scipy
matplotlib
sqlite3  (standard library)
```

Custom function module: `ISMIP_function` (located at `/Users/lebescom/Documents/Code/Function/`).
