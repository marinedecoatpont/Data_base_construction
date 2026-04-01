import numpy as np
import xarray as xr
import sys
import importlib
import config
import sqlite3

#load personal function
sys.path.append('/Users/lebescom/Documents/Code/Function')
import ISMIP_function as ismip
importlib.reload(ismip)

#---------------------------------- COMPARISON OF THE FLUX A THE GROUNDING LINE  -----------------------------------------------
#AUTHOR: Marine de Coaptont
#November 12, 2025
#IGE / PhD 
#
#This script save the positions the grounding line of all the simulation expriment in order to compute the mean grounding line flux for each grid step
#
#------------------------------------------------------------------------------------------------------------------------------------------
print('START OF COMPARISON PROGRAM', flush=True)

test = True

"""
simulation = {
    'DC_ISSM' : 10 ,
    'IGE_ElmerIce' : 6,
    'ILTS_SICOPOLIS' : 14,
    'LSCE_GRISLI2' : 14,
    'NORCE_CISM2-MAR364-ERA-t1' : 6,
    'PIK_PISM' : 10,
    'UCM_Yelmo' : 14,
    'ULB_fETISh-KoriBU2' : 14,
    'UNN_Ua' : 30,
    'UTAS_ElmerIce' : 10,
    'VUB_AISMPALEO' : 10,
    'VUW_PRISM1' : 10,
    'VUW_PRISM2' : 10,
}

vuw = ['VUW_PRISM1', 'VUW_PRISM2']
"""

#parameters
density_ice = 917
convert_vel = 365.25 * 24 * 3600 #to get m/yr
gravity = 9.81*(1-(density_ice/1027))
n = 3
g = 9.81
expos = -1/n
rho_ice = 917
rho_water = 1027
resolutions = [16]
records = []


for reso in resolutions:
    #templet for netCDF file
    file = f'grid{reso}x{reso}.nc'
    ds_ref = xr.open_dataset(file)
    dx = ismip.get_resolution(ds_ref)

    if test == True:
        simulation = {
            'IGE_ElmerIce' : 6,
        }
        times= 25
        db_path = f"Flux_DataBase_{reso}km_elmerice.db"
    else:
        simulation = {
            'DC_ISSM' : 6,
            'IGE_ElmerIce' : 6,
            'ILTS_SICOPOLIS' : 14,
            'LSCE_GRISLI' : 14,
            'LSCE_GRISLI2' : 14,
            'NCAR_CISM1':6,
            'NORCE_CISM2-MAR364-ERA-t1' : 6,
            'PIK_PISM' : 6,
            'UCM_Yelmo' : 14,
            'ULB_fETISh-KoriBU1' : 14,
            'ULB_fETISh-KoriBU2' : 14,
            'UTAS_ElmerIce' : 6,
        }
        times_tot= 286
        db_path = f"{config.SAVE_PATH}/DataBase_{reso}km_all.db"

    print("Creating database", flush=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE flux_data (
        x REAL,
        y REAL,
        simulation TEXT,
        experiment TEXT,
        time INTEGER,
        flux REAL,
        thickness REAL,
        velocity REAL,
        drag REAL,
        viscosity REAL,
        surface REAL,
        base REAL,
        bed REAL,
        flotaison REAL,
        R_drag REAL,
        driving_stress REAL,
        slope_flux REAL,
        slope_max REAL
    );
    """)
    conn.commit()

    print("Starting computation", flush=True)
    for simu, nb_exp in simulation.items():
        # -------- SIMULATION AND EXPERIMENT LOOP --------
        print(f'Computation for {simu}:', flush=True)
        exp_list = [f'expAE{str(i).zfill(2)}' for i in range(1, nb_exp + 1)]

        for exp in exp_list:
            print(f'Computation for {exp}', flush=True)
            file_path = f'{config.PATH_IF}/ligroundf_{simu}_{exp}.nc'
            
            try:
                ligroundf_ds = xr.open_dataset(file_path, decode_times=False)
            except FileNotFoundError:
                print(f"file: {file_path} does not exist", flush = True)
                continue
            data_orog = ismip.open_file(simu, exp, 'lithk')
            vx_data = ismip.open_file(simu, exp, 'xvelmean')
            vy_data = ismip.open_file(simu, exp, 'yvelmean')
            drag_data = ismip.open_file(simu, exp, 'strbasemag')
            mass_balance_data = ismip.open_file(simu, exp, 'acabf')
            basal_mass_balance_data = ismip.open_file(simu, exp, 'libmassbffl')
            bed_data = ismip.open_file(simu, exp, 'topg')
            surf_data = ismip.open_file(simu, exp, 'orog')
            base_data = ismip.open_file(simu, exp, 'base')

            if simu == 'LSCE_GRISLI':
                times = times_tot - 1
            else:
                time = times_tot
            
            for time in range(times):
                ligroundf = ligroundf_ds.ligroundf.isel(time=time)#ice flux at the grounding line
                orog = data_orog.lithk.isel(time=time)#ice thickness
                vx = vx_data.xvelmean.isel(time=time)#velocity
                vy = vy_data.yvelmean.isel(time=time)#velocity
                drag = drag_data.strbasemag.isel(time=time)#basal drag
                bed = bed_data.topg.isel(time=time)#altitude of bed
                surf = surf_data.orog.isel(time=time)#altitude of surface
                base = base_data.base.isel(time=time)#altitude of the ice sheet base

                flux = np.abs(ligroundf * density_ice)/1e12

                mask_flux = (flux != 0) & np.isfinite(flux)
                mask_vx   = mask_flux & np.isfinite(vx)
                mask_vy   = mask_flux & np.isfinite(vy)
                mask_thick = mask_flux & np.isfinite(orog)
                mask_drag  = mask_flux & np.isfinite(drag)
                mask_surf  = mask_flux & np.isfinite(surf)
                mask_base  = mask_flux & np.isfinite(base)
                mask_bed   = mask_flux & np.isfinite(bed)
                mask_vel = mask_vx & mask_vy

                vx_gl   = xr.where(mask_vx, vx, 0)
                vy_gl   = xr.where(mask_vy, vy, 0)
                thick_gl = xr.where(mask_thick, orog, 0)
                drag_gl  = xr.where(mask_drag, drag, 0)
                surf_gl  = xr.where(mask_surf, surf, 0)
                base_gl  = xr.where(mask_base, base, 0)
                bed_gl   = xr.where(mask_bed, bed, 0)

                velocity = np.sqrt(vx_gl**2 + vy_gl**2)#velocity norm in m/s

                #viscosity
                exx = np.gradient(vx_gl, dx, axis=1)
                dvx_dy = np.gradient(vx_gl, dx, axis=0)
                dvy_dx = np.gradient(vy_gl, dx, axis=1)
                eyy = np.gradient(vy_gl, dx, axis=0)            
                exy = 0.5*(dvx_dy + dvy_dx)
                strain_rate = np.sqrt(0.5 * (exx**2 + eyy**2 + 2*(exy**2)))
                deviatic_stress = 0.25*density_ice*gravity*thick_gl
                mask_mu = (mask_flux & np.isfinite(strain_rate)& np.isfinite(deviatic_stress)& (strain_rate > 0)& (deviatic_stress > 0))
                mu = xr.where(mask_mu,(strain_rate / deviatic_stress**n) ** expos,np.nan)
                
                #buoyancy coefficient
                flotaison = rho_ice*thick_gl - rho_water*(-base_gl)

                #surface slope
                dsdx = surf_gl.differentiate('x')
                dsdy =surf_gl.differentiate('y')
                nx = vx_gl / velocity
                ny = vy_gl / velocity

                slope_flux = dsdx*nx + dsdy*ny
                slope_max  = np.sqrt(dsdx**2 + dsdy**2)

                #driving stress
                tau_d = rho_ice * g * thick_gl * slope_flux

                #R drag
                R = drag / tau_d

                flux = ismip.grid(flux, reso)
                velocity = ismip.grid(velocity, reso)
                thick_gl = ismip.grid(thick_gl, reso)
                drag_gl = ismip.grid(drag_gl, reso)
                bed_gl = ismip.grid(bed_gl, reso)
                surf_gl = ismip.grid(surf_gl, reso)
                base_gl = ismip.grid(base_gl, reso)
                mu = ismip.grid(mu, reso)
                flotaison = ismip.grid(flotaison, reso)
                slope_flux = ismip.grid(slope_flux, reso)
                slope_max = ismip.grid(slope_max, reso)
                driving_stress = ismip.grid(tau_d, reso)
                R = ismip.grid(R, reso)
                
                
                flux = flux.values
                velo = velocity.values
                thick_gl = thick_gl.values
                drag_gl =drag_gl.values
                bed_gl = bed_gl.values
                surf_gl = surf_gl.values
                base_gl = base_gl.values
                mu = mu.values
                flotaison = flotaison.values
                slope_max = slope_max.values
                slope_flux = slope_flux.values
                driving_stress = driving_stress.values
                R = R.values

                ii, jj = np.where((flux != 0) & ~np.isnan(flux))

                flux_values = flux[ii, jj]
                orog = thick_gl[ii, jj]
                velocity = velo[ii, jj]
                drag = drag_gl[ii, jj]
                bed = bed_gl [ii,jj]
                surf = surf_gl[ii,jj]
                base = base_gl[ii,jj]
                mu = mu[ii, jj]
                flotaison = flotaison[ii,jj]
                slope_max = slope_max[ii,jj]
                slope_flux = slope_flux[ii,jj]
                driving_stress = driving_stress[ii,jj]
                R = R[ii,jj]

                #coordinate grounding line
                x = flux.x.values
                y = flux.y.values
                xf = x[jj]
                yf = y[ii]

                for x, y, flux_val, orog_val, vel_val, drag_val, bed_val, surf_val, base_val, mu_val, float_val, R_val, driving_val, slope_flux_val, slope_max_val in zip(xf,yf,flux_values, orog,velocity, drag, bed, surf, base, mu, flotaison, R, driving_stress, slope_flux, slope_max):
                    records.append((
                            float(x), float(y), str(simu), str(exp), int(time),
                            float(flux_val),
                            float(orog_val),
                            float(vel_val),
                            float(drag_val),
                            float(mu_val),
                            float(surf_val), 
                            float(base_val),
                            float(bed_val),
                            float(float_val),
                            float(R_val),
                            float(driving_val),
                            float(slope_flux_val),
                            float(slope_max_val)
                    ))
                cursor.executemany("""INSERT INTO flux_data (
                        x, y, simulation, experiment, time,
                        flux, thickness, velocity, drag, viscosity,
                        surface, base, bed, flotaison,R_drag,driving_stress,
                        slope_flux,slope_max
                        )VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?,?)""", records)
                conn.commit()
                records = [] 
            ligroundf_ds.close()

    conn.close()

print('Everything seems okay ~(°w°~)', flush=True)
print('END OF COMPARISON PROGRAM', flush=True)