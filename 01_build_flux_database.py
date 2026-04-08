"""
01_build_flux_database.py
=========================
Builds the main SQLite database (flux_data table) for each resolution.

For every simulation / experiment / time step:
  1. Identifies grounding-line pixels from the ligroundf field.
  2. Computes per-pixel physical quantities (flux, thickness, velocity,drag, viscosity, surface slope, buoyancy, buttressing).
  3. Inserts all records into flux_data.

The viscosity is derived from the local strain-rate tensor and the deviatoric-stress formulation (see buttressing computation below).
"""

import sys
import sqlite3
import argparse
import numpy as np
import xarray as xr
import config



# =============================================================================
# HELPERS
# =============================================================================

def _open(simu: str, exp: str, var: str) -> xr.Dataset:
    data = xr.open_dataset(f'{config.PATH_IF}/{simu}/{exp}/{var}_AIS_{simu}_{exp}.nc', decode_times = False)
    return data


def _grid(mask,reso):
    if reso == 4:
        ref_grid_data = xr.open_dataset(f'{config.SAVE_PATH}/grid4x4.nc')
        ref_grid = ref_grid_data.grounding_mask

        mask_interp = mask.interp(x = ref_grid.x, y = ref_grid.y)
    elif reso == 16:
        ref_grid_data = xr.open_dataset(f'{config.SAVE_PATH}/grid16x16.nc')
        mask_interp = mask.interp(x = ref_grid_data.x, y = ref_grid_data.y)

    return mask_interp

def get_resolution(data):
    """Give the resolution of a netCDF grid in meters

    Parameters
    ----------
        data : dataset, (e.g ice flux at the grounding line)

    Returns
    -------
    (float)
        Grid resolution
    """
    x = data.x
    x0 = x.isel(x = 0)
    x1 = x.isel(x = 1)

    reso = abs(x1 - x0)
    return reso.values.item()


def _compute_viscosity(vx: xr.DataArray, vy: xr.DataArray,thick: xr.DataArray, dx: float) -> xr.DataArray:
    """
    Effective viscosity from the strain-rate tensor and deviatoric stress.
    mu = (strain_rate / deviatoric_stress^n)^(-1/n)
    """
    exx = np.gradient(vx.values, dx, axis=1)
    eyy = np.gradient(vy.values, dx, axis=0)
    dvx_dy = np.gradient(vx.values, dx, axis=0)
    dvy_dx = np.gradient(vy.values, dx, axis=1)
    exy = 0.5 * (dvx_dy + dvy_dx)

    strain_rate = np.sqrt(0.5 * (exx**2 + eyy**2 + 2 * exy**2))
    deviatoric  = 0.25 * config.RHO_ICE * config.GRAVITY_EFF * thick.values

    mask_valid = (strain_rate > 0) & (deviatoric > 0) & np.isfinite(strain_rate) & np.isfinite(deviatoric)
    mu = np.where(mask_valid,
                  (strain_rate / deviatoric**config.GLEN_N) ** config.EXPOS,
                  np.nan)
    return xr.DataArray(mu, coords=vx.coords, dims=vx.dims)


def _compute_buttressing(vx: xr.DataArray, vy: xr.DataArray,thick: np.ndarray, mu: np.ndarray,coords: np.ndarray) -> tuple[list, list, list]:
    """
    Computes the buttressing factor for each grounding-line point.

    Returns
    -------
    nx_list, ny_list : unit outward normal vectors along the GL
    b_list           : buttressing values clipped to [0, 1]
    b_natural_list   : raw (unclipped) buttressing values
    vn_list          : velocity normal to the GL (m yr-1)
    """
    nx_list, ny_list = [], []

    for i in range(len(coords)):
        if i == 0:
            dx_c = coords[i+1, 0] - coords[i, 0]
            dy_c = coords[i+1, 1] - coords[i, 1]
        elif i == len(coords) - 1:
            dx_c = coords[i, 0] - coords[i-1, 0]
            dy_c = coords[i, 1] - coords[i-1, 1]
        else:
            dx_c = coords[i+1, 0] - coords[i-1, 0]
            dy_c = coords[i+1, 1] - coords[i-1, 1]

        # outward normal (perpendicular to tangent)
        norm = np.sqrt(dx_c**2 + dy_c**2)
        nx_list.append(-dy_c / norm)
        ny_list.append( dx_c / norm)

    G = config.RHO_ICE * (1.0 - config.RHO_ICE / config.RHO_WATER) * config.GRAVITY

    dudx = vx.differentiate("x")
    dvdy = vy.differentiate("y")
    dudy = vx.differentiate("y")
    dvdx = vy.differentiate("x")

    b_list, b_natural_list, vn_list = [], [], []

    for idx, (x_pt, y_pt) in enumerate(coords):
        nx = nx_list[idx]
        ny = ny_list[idx]
        h  = thick[idx]
        m  = mu[idx]

        dudx_v = dudx.sel(x=x_pt, y=y_pt, method="nearest").values
        dvdy_v = dvdy.sel(x=x_pt, y=y_pt, method="nearest").values
        dudy_v = dudy.sel(x=x_pt, y=y_pt, method="nearest").values
        dvdx_v = dvdx.sel(x=x_pt, y=y_pt, method="nearest").values
        vx_v   = vx.sel(x=x_pt, y=y_pt,   method="nearest").values
        vy_v   = vy.sel(x=x_pt, y=y_pt,   method="nearest").values

        exx = dudx_v
        eyy = dvdy_v
        exy = 0.5 * (dudy_v + dvdx_v)
        strain_rate  = np.sqrt(0.5 * (exx**2 + eyy**2) + exy**2)
        deviatoric   = 0.25 * config.RHO_ICE * config.GRAVITY_EFF * h

        if strain_rate > 0 and deviatoric > 0:
            mu_local = (strain_rate / deviatoric**config.GLEN_N) ** config.EXPOS
        else:
            mu_local = float("nan")

        visco = 0.5 * mu_local * strain_rate ** ((1 - config.GLEN_N) / config.GLEN_N) if not np.isnan(mu_local) else float("nan")

        N   = 2 * visco * ((2*exx + eyy)*nx**2 + (dudy_v + dvdx_v)*nx*ny + (2*eyy + exx)*ny**2)
        b_n = N / (G * h * 0.5) if (h > 0 and not np.isnan(N)) else float("nan")
        b   = float(np.clip(b_n, 0.0, 1.0)) if not np.isnan(b_n) else float("nan")

        vn = (vx_v * nx + vy_v * ny) * config.YEAR_S

        b_list.append(b)
        b_natural_list.append(float(b_n) if not np.isnan(b_n) else float("nan"))
        vn_list.append(float(vn))

    return nx_list, ny_list, b_list, b_natural_list, vn_list


# =============================================================================
# MAIN
# =============================================================================

def build_flux_database(reso: int, simulations: dict, db_path: str, test: bool = False) -> None:
    """
    Builds (or appends to) the flux_data table in the SQLite database at db_path.
    """
    ds_ref = xr.open_dataset(config.GRID_FILES[reso])
    dx     = get_resolution(ds_ref)

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flux_data (
        x                REAL,
        y                REAL,
        simulation       TEXT,
        experiment       TEXT,
        time             INTEGER,
        flux             REAL,
        thickness        REAL,
        velocity         REAL,
        drag             REAL,
        surface          REAL,
        base             REAL,
        bed              REAL,
        flotaison        REAL,
        R_drag           REAL,
        driving_stress   REAL,
        slope_flux       REAL,
        slope_max        REAL,
        viscosity        REAL,
        buttressing      REAL,
        buttressing_natural REAL,
        velocity_normal REAL
    );
    """)
    conn.commit()

    records = []

    for simu, nb_exp in simulations.items():
        print(f"Processing {simu} …", flush=True)
        exp_list = [f"expAE{str(i).zfill(2)}" for i in range(1, nb_exp + 1)]

        for exp in exp_list:
            print(f"  {exp}", flush=True)
            file_path = f"{config.PATH_IF}/ligroundf_{simu}_{exp}.nc"

            try:
                ligroundf_ds = xr.open_dataset(file_path, decode_times=False)
            except FileNotFoundError:
                print(f"  skip, file not found: {file_path}", flush=True)
                continue

            orog_ds = _open(simu, exp, "lithk")
            vx_ds = _open(simu, exp, "xvelmean")
            vy_ds = _open(simu, exp, "yvelmean")
            drag_ds = _open(simu, exp, "strbasemag")
            bed_ds = _open(simu, exp, "topg")
            surf_ds = _open(simu, exp, "orog")
            base_ds = _open(simu, exp, "base")

            n_times = (config.TIMES_TOTAL_GRISLI
                       if simu == "LSCE_GRISLI"
                       else config.TIMES_TOTAL)
            if test:
                n_times = min(n_times, 5)

            for t in range(n_times):
                ligroundf = ligroundf_ds.ligroundf.isel(time=t)
                orog = orog_ds.lithk.isel(time=t)
                vx = vx_ds.xvelmean.isel(time=t)
                vy = vy_ds.yvelmean.isel(time=t)
                drag = drag_ds.strbasemag.isel(time=t)
                bed = bed_ds.topg.isel(time=t)
                surf = surf_ds.orog.isel(time=t)
                base = base_ds.base.isel(time=t)

                flux = np.abs(ligroundf * config.RHO_ICE) / 1e12

                # --- masks ---
                mask = (flux != 0) & np.isfinite(flux)
                vx_m = xr.where(mask & np.isfinite(vx),vx,0)
                vy_m = xr.where(mask & np.isfinite(vy),vy, 0)
                thick = xr.where(mask & np.isfinite(orog),orog,0)
                drag_m = xr.where(mask & np.isfinite(drag),drag,0)
                surf_m = xr.where(mask & np.isfinite(surf),surf,0)
                base_m = xr.where(mask & np.isfinite(base),base,0)
                bed_m = xr.where(mask & np.isfinite(bed), bed,0)

                velocity  = np.sqrt(vx_m**2 + vy_m**2)
                flotaison = config.RHO_ICE * thick - config.RHO_WATER * (-base_m)

                #viscosity 
                mu = _compute_viscosity(vx_m, vy_m, thick, dx)

                # surface slope & driving stress 
                dsdx = surf_m.differentiate("x")
                dsdy = surf_m.differentiate("y")
                vel = velocity.where(velocity > 0)
                nx_v = vx_m / vel
                ny_v = vy_m / vel
                slope_flux = dsdx * nx_v + dsdy * ny_v
                slope_max = np.sqrt(dsdx**2 + dsdy**2)
                tau_d = config.RHO_ICE * config.GRAVITY * thick * slope_flux
                R_drag = drag_m / tau_d.where(tau_d != 0)

                # regrid
                flux_g = _grid(flux, reso)
                vel_g = _grid(velocity,reso)
                thick_g = _grid(thick, reso)
                drag_g = _grid(drag_m,reso)
                bed_g = _grid(bed_m,reso)
                surf_g = _grid(surf_m,reso)
                base_g = _grid(base_m,reso)
                mu_g = _grid(mu,reso)
                flot_g = _grid(flotaison,reso)
                sf_g = _grid(slope_flux,reso)
                sm_g = _grid(slope_max, reso)
                tau_g = _grid(tau_d,reso)
                R_g = _grid(R_drag,reso)
                vx_g = _grid(vx_m, reso)
                vy_g = _grid(vy_m,reso)

                # extract GL pixels
                flux_arr = flux_g.values
                ii, jj = np.where((flux_arr != 0) & ~np.isnan(flux_arr))
                if len(ii) == 0:
                    continue

                x_coords = flux_g.x.values[jj]
                y_coords = flux_g.y.values[ii]
                coords = np.column_stack([x_coords, y_coords])

                thick_pts = thick_g.values[ii, jj]
                mu_pts = mu_g.values[ii, jj]

                # buttressing 
                vx_g_t = vx_g
                vy_g_t = vy_g
                _, _, b_list, b_nat_list, vn_list = _compute_buttressing(vx_g_t, vy_g_t, thick_pts, mu_pts, coords)

                for k in range(len(ii)):
                    records.append((
                        float(x_coords[k]),
                        float(y_coords[k]),
                        str(simu),
                        str(exp),
                        int(t),
                        float(flux_arr[ii[k], jj[k]]),
                        float(thick_g.values[ii[k], jj[k]]),
                        float(vel_g.values[ii[k], jj[k]]),
                        float(drag_g.values[ii[k], jj[k]]),
                        float(surf_g.values[ii[k], jj[k]]),
                        float(base_g.values[ii[k], jj[k]]),
                        float(bed_g.values[ii[k], jj[k]]),
                        float(flot_g.values[ii[k], jj[k]]),
                        float(R_g.values[ii[k], jj[k]]),
                        float(tau_g.values[ii[k], jj[k]]),
                        float(sf_g.values[ii[k], jj[k]]),
                        float(sm_g.values[ii[k], jj[k]]),
                        float(mu_pts[k]),
                        float(b_list[k]),
                        float(b_nat_list[k]),
                        float(vn_list[k]),
                    ))

                cursor.executemany("""
                INSERT INTO flux_data (
                    x, y, simulation, experiment, time,
                    flux, thickness, velocity, drag,
                    surface, base, bed, flotaison,
                    R_drag, driving_stress, slope_flux, slope_max,
                    viscosity, buttressing, buttressing_natural, velocity_normal
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, records)
                conn.commit()
                records = []

            ligroundf_ds.close()

    conn.close()
    print("flux_data table complete.", flush=True)

parser = argparse.ArgumentParser(description="Build the main flux_data SQLite table.")
parser.add_argument("--test", action="store_true", help="Run on a small subset.")
args = parser.parse_args()

simulations = ({"IGE_ElmerIce": 2} if args.test else config.SIMULATIONS)

for reso in config.RESOLUTIONS:
    db_path = config.get_db_path(reso)
    print(f"\n=== Resolution {reso} km  DB: {db_path} ===", flush=True)
    build_flux_database(reso, simulations, db_path, test=args.test)

print("Done.", flush=True)
