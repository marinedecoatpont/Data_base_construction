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


def _compute_viscosity_buttressing(vx, vy,thick, x, y, k):
    """
    Effective viscosity from the strain-rate tensor and deviatoric stress.
    mu = (strain_rate / deviatoric_stress^n)^(-1/n)
    """
    dudx = vx.differentiate("x")
    dvdy = vy.differentiate("y")
    dudy = vx.differentiate("y")
    dvdx = vy.differentiate("x")

    dudx = dudx.sel(x=x[k], y=y[k], method="nearest").values
    dvdy = dvdy.sel(x=x[k], y=y[k], method="nearest").values
    dudy = dudy.sel(x=x[k], y=y[k], method="nearest").values
    dvdx = dvdx.sel(x=x[k], y=y[k], method="nearest").values

    # strain rate tensor
    exx = dudx
    eyy = dvdy
    exy = 0.5 * (dudy + dvdx)
    strain_rate = np.sqrt(0.5*(exx**2 + eyy**2) + exy**2)
    deviatic_stress = 0.25*config.RHO_ICE*config.GRAVITY*thick
    mu = (strain_rate / deviatic_stress**config.GLEN_N)**config.EXPOS

    G = config.RHO_ICE * (1.0 - config.RHO_ICE / config.RHO_WATER) * config.GRAVITY
    visco = 0.5 * mu * strain_rate**((1-config.GLEN_N)/config.GLEN_N)

    #buttresing
    velocity  = np.sqrt(vx**2 + vy**2)
    vel = velocity.where(velocity > 0)
    nx = vx / vel
    ny = vy / vel
    nx = nx.sel(x=x[k], y=y[k], method="nearest").values
    ny = ny.sel(x=x[k], y=y[k], method="nearest").values
    N = 2 * visco * ((2*dudx + dvdy)*nx**2 + (dudy+dvdx)*nx*ny + (2*dvdy + dudx)*ny**2)
    b_n = N/(G*thick*0.5)
    b= max(0, min(1,b_n))
    vn = (vx.sel(x=x[k], y=y[k], method="nearest").values * nx + vy.sel(x=x[k], y=y[k], method="nearest").values * ny) * config.YEAR_S
    return mu, b_n, b, vn



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
                flot_g = _grid(flotaison,reso)
                sf_g = _grid(slope_flux,reso)
                sm_g = _grid(slope_max, reso)
                tau_g = _grid(tau_d,reso)
                R_g = _grid(R_drag,reso)
                vx_g = _grid(vx_m, reso)
                vy_g = _grid(vy_m,reso)
                nx_g = _grid(nx_v, reso)
                ny_g = _grid(ny_v, reso)

                # extract GL pixels
                flux_arr = flux_g.values
                ii, jj = np.where((flux_arr != 0) & ~np.isnan(flux_arr))
                if len(ii) == 0:
                    continue

                x_coords = flux_g.x.values[jj]
                y_coords = flux_g.y.values[ii]
                coords = np.column_stack([x_coords, y_coords])

                thick_pts = thick_g.values[ii, jj]

                for k in range(len(ii)):
                    mu, b_n, b, vel_n = _compute_viscosity_buttressing(vx_m, vy_m, thick_g.values[ii[k], jj[k]], x_coords, y_coords, k)


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
                        float(mu),
                        float(b),
                        float(b_n),
                        float(vel_n),
                    ))

                cursor.executemany("""
                INSERT INTO flux_data (
                    x, y, simulation, experiment, time,
                    flux, thickness, velocity, drag,
                    surface, base, bed, flotaison,
                    R_drag, driving_stress, slope_flux, slope_max,
                    viscosity, buttressing, buttressing_natural, velocity_normal
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
