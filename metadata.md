# Database Metadata

This document describes the structure and column definitions of all SQLite databases produced by the grounding-line flux pipeline.

---

## Main database  —  `DataBase_{reso}km.db`

### Table: `flux_data`

One row per grounding-line pixel, per simulation, per experiment, per time step.

| Column | Long name | Unit |
|---|---|---|
| `x` | Easting (polar stereographic) | m |
| `y` | Northing (polar stereographic) | m |
| `simulation` | Model name (e.g. IGE_ElmerIce) | — |
| `experiment` | Experiment identifier (e.g. expAE01) | — |
| `time` | Time index (0-based integer) | — |
| `flux` | Ice mass flux at the grounding line | Gt yr⁻¹ |
| `thickness` | Ice thickness | m |
| `velocity` | Ice velocity norm | m s⁻¹ |
| `drag` | Basal shear stress magnitude | Pa |
| `surface` | Ice surface elevation | m |
| `base` | Ice base elevation | m |
| `bed` | Bedrock topography | m |
| `flotaison` | Buoyancy anomaly  ρ_i · H − ρ_w · |base| | kg m⁻² |
| `R_drag` | Drag ratio  τ_b / τ_d | — |
| `driving_stress` | Gravitational driving stress  ρ_i g H (dS/dx · n̂) | Pa |
| `slope_flux` | Surface slope in the flow direction | m m⁻¹ |
| `slope_max` | Maximum surface slope magnitude | m m⁻¹ |
| `viscosity` | Effective ice viscosity  μ from strain-rate tensor | Pa s^(1/n) |
| `buttressing` | Buttressing factor (clipped to [0, 1]) | — |
| `buttressing_natural` | Raw buttressing factor (unclipped) | — |
| `residence` | Number of time steps the pixel appears in flux_data for this (simu, exp) | — |

---

### Table: `last_grounded_point`

Same columns as `flux_data`, plus:

| Column | Long name | Unit |
|---|---|---|
| `n_simulations` | Number of distinct simulations at this (x, y) | — |
| `n_timesteps` | Total number of rows at this (x, y) across all simulations | — |

Each row is the **last available time step** for a given (simulation, experiment, x, y).

---

### Table: `time_1_10`

Same columns and extra statistics as `last_grounded_point`.

Contains only rows from `flux_data` where `time % 10 == 0` (every 10th time step).

---

## Per-basin databases  —  `basin_databases/basin_{id:02d}_{reso}km.db`

### Table: `basin_data`

Same schema as `last_grounded_point` (all columns including `n_simulations` and `n_timesteps`).

Rows are filtered to the spatial extent of Zwally basin `{id}` (1–26).

---

## BedMachine grounding-line database  —  `BedMachine_GL_{reso}km.db`

### Table: `bedmachine_gl`

Same schema as `last_grounded_point`, plus:

| Column | Long name | Unit |
|---|---|---|
| `flux_bed` | BedMachine observed grounding-line flux (from `ligroundf_bedmachine_all_test.nc`) | Same unit as `ligroundf` field |

Rows are filtered to (x, y) pixels that have a non-zero value in the BedMachine `ligroundf` field after interpolation onto the simulation grid.

---

## Notes on key derived quantities

**Viscosity**  
Computed from the vertically integrated strain-rate tensor:

```
ε̇_e = sqrt( 0.5 (ε_xx² + ε_yy²) + ε_xy² )
τ_d  = 0.25 · ρ_ice · g_eff · H          (deviatoric stress scale)
μ    = (ε̇_e / τ_d^n)^(-1/n)
```

**Buttressing**  
Normal stress projected onto the grounding-line outward normal n̂:

```
N    = 2μ [(2ε_xx + ε_yy) nx² + (ε_xy)(nx ny) + (2ε_yy + ε_xx) ny²]
Θ    = N / (G · H · 0.5)      where  G = ρ_i (1 - ρ_i/ρ_w) g
```

Θ = 0 → fully buttressed; Θ = 1 → no buttressing.

**Buoyancy anomaly (flotaison)**  
```
flotaison = ρ_ice · H - ρ_water · |base|
```
Positive values indicate the ice is grounded above hydrostatic equilibrium.

---

## Physical constants used

| Symbol | Value | Unit |
|---|---|---|
| ρ_ice | 917 | kg m⁻³ |
| ρ_water | 1027 | kg m⁻³ |
| g | 9.81 | m s⁻² |
| n (Glen) | 3 | — |
| 1 yr | 365.25 × 24 × 3600 | s |
