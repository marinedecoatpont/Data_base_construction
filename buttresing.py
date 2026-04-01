import config
import sqlite3
import sys
import pandas as pd
import xarray as xr
import numpy as np


# ---------------------------------- COMPARISON OF THE FLUX AT THE GROUNDING LINE -----------------------------------------------
# AUTHOR: Marine de Coatpont
# November 12, 2025
# IGE / PhD
#
# This script extracts, for each (simulation, experiment, x, y),
# the last available time step from flux_data_2 and stores it in
# a new table: last_grounded_point
# -------------------------------------------------------------------------------------------------------------------------------
def grid(mask,reso):
    if reso == 4:
        ref_grid_data = xr.open_dataset(f'{config.SAVE_PATH}/grid4x4.nc')
        ref_grid = ref_grid_data.grounding_mask

        mask_interp = mask.interp(x = ref_grid.x, y = ref_grid.y)
    elif reso == 16:
        ref_grid_data = xr.open_dataset(f'grid16x16.nc')
        mask_interp = mask.interp(x = ref_grid_data.x, y = ref_grid_data.y)

    return mask_interp

print('START OF COMPARISON PROGRAM', flush=True)

#parameters
test = False

if test :
    resolutions = [16]
    table = ['last_grounded_point']
else:
    resolutions = [16, 4]
    table = ['last_grounded_point','time_1_10','flux_data']


g = 9.81
rho_ice = 917
rho_water = 1027
G = rho_ice*(1-(rho_ice/rho_water))*g
n=3
expos = -1/n
gravity = 9.81*(1-(rho_ice/1027))
r_tot = []

for reso in resolutions:
    for data in table:
        if test:
            db_path = "/Users/lebescom/DataBase_16km_test.db"
            quevery = f"SELECT rowid, x, y, time, simulation, experiment, viscosity, thickness FROM '{data}' WHERE simulation IN ('DC_ISSM','ILTS_SICOPOLIS', 'LSCE_GRISLI','NCAR_CISM1','NORCE_CISM2-MAR364-ERA-t1', 'UCM_Yelmo','ULB_fETISh-KoriBU1','ULB_fETISh-KoriBU2', 'UTAS_ElmerIce')"
        else:
            db_path = "/Users/lebescom/DataBase_16km_test.db"
            quevery = f"SELECT rowid, x, y, vx, thickness, vy, surface, drag, time, simulation, experiment, velocity,viscosity FROM '{data}' WHERE simulation IN ('LSCE_GRISLI') "
        
        conn = sqlite3.connect(db_path)
        
        df = pd.read_sql(quevery, conn)
        print(df)
        cursor = conn.cursor()

        queveryr = f"""ALTER TABLE '{data}' ADD COLUMN buttressing REAL"""
        queveryr_b = f"""ALTER TABLE '{data}' ADD COLUMN buttressing_natural REAL"""
        queveryvelonormal = f"""ALTER TABLE '{data}' ADD COLUMN velocity_normal REAL"""

        try:
            cursor.execute(queveryvelonormal)
            cursor.execute(queveryr_b)
            cursor.execute(queveryr)
            
        except sqlite3.OperationalError:
            pass

        simulations = df["simulation"].unique()


        for simu in simulations:
            sub = df[df['simulation'] == simu]

            for exp in sub["experiment"].unique():
                print(f'Computation for {reso} {data} {simu} {exp}', flush = True)
                df_exp = sub[sub["experiment"] == exp]

                #normal a la GL
                coords = df_exp[['x','y']].values
                nx_list = []
                ny_list = []

                for i in range(len(coords)):

                    if i == 0:
                        dx = coords[i+1,0] - coords[i,0]
                        dy = coords[i+1,1] - coords[i,1]

                    elif i == len(coords)-1:
                        dx = coords[i,0] - coords[i-1,0]
                        dy = coords[i,1] - coords[i-1,1]

                    else:
                        dx = coords[i+1,0] - coords[i-1,0]
                        dy = coords[i+1,1] - coords[i-1,1]

                    # tangente
                    tx = dx
                    ty = dy

                    # normale
                    nx = -ty
                    ny = tx

                    norm = np.sqrt(nx**2 + ny**2)

                    nx_list.append(nx/norm)
                    ny_list.append(ny/norm)

                df_exp["nx"] = nx_list
                df_exp["ny"] = ny_list

                vx = xr.open_dataset(f'/Users/lebescom/{simu}/{exp}/xvelmean_AIS_{simu}_{exp}.nc', decode_times = False)
                vx = vx.xvelmean
                vy = xr.open_dataset(f'/Users/lebescom/{simu}/{exp}/yvelmean_AIS_{simu}_{exp}.nc', decode_times = False)
                vy = vy.yvelmean
                print(df_exp)

                for _, row in df_exp.iterrows():
                    vx_data = vx.isel(time=int(row["time"]))
                    vy_data = vy.isel(time=int(row["time"]))
                    thick = row["thickness"]
                    mu = row["viscosity"]
                    print('Visco',mu)
                    dudx = vx_data.differentiate("x")
                    dvdy = vy_data.differentiate("y")
                    dudy = vx_data.differentiate("y")
                    dvdx = vy_data.differentiate("x")
                    nx = row['nx']
                    ny = row['ny']
                    

                    #interpolation
                    dudx = grid(dudx, reso)
                    dvdy = grid(dvdy, reso)
                    dudy = grid(dudy, reso)
                    dvdx = grid(dvdx, reso)
                    vx_data = grid(vx_data, reso)
                    vy_data = grid(vy_data, reso)

                    dudx = dudx.sel(x=row["x"], y=row["y"], method="nearest").values
                    dvdy = dvdy.sel(x=row["x"], y=row["y"], method="nearest").values
                    dudy = dudy.sel(x=row["x"], y=row["y"], method="nearest").values
                    dvdx = dvdx.sel(x=row["x"], y=row["y"], method="nearest").values

                    vy_val = vy_data.sel(x=row["x"], y=row["y"], method="nearest").values
                    vx_val = vx_data.sel(x=row["x"], y=row["y"], method="nearest").values

                    # strain rate tensor
                    exx = dudx
                    eyy = dvdy
                    exy = 0.5 * (dudy + dvdx)
                    strain_rate = np.sqrt(0.5*(exx**2 + eyy**2) + exy**2)
                    deviatic_stress = 0.25*rho_ice*gravity*thick
                    mu = (strain_rate / deviatic_stress**n)**expos
                    print('mu', mu)
                    visco = 0.5 * mu * strain_rate**((1-n)/n)
                    print('Visco',visco)
                    print('dudy', dudy)
                    print('dudx', dudx)
                    print('dvdy', dvdy)
                    print('dvdx', dvdx)
                    print('ny',ny)
                    print('nx',nx)

                    #buttresing

                    N = 2 * visco * ((2*dudx + dvdy)*nx**2 + (dudy+dvdx)*nx*ny + (2*dvdy + dudx)*ny**2)
                    b_n = N/(G*thick*0.5)
                    b = max(0, min(1,b_n))
                    velo_normal = (vx_val*nx + vy_val*ny)* 365.25 * 24 * 3600
                    print('buttressing',b)


                    cursor.execute("""UPDATE last_grounded_point SET buttressing = ? WHERE rowid = ?""",(float(b), int(row["rowid"])))
                    cursor.execute("""UPDATE last_grounded_point SET viscosity = ? WHERE rowid = ?""",(float(mu), int(row["rowid"])))
                    cursor.execute("""UPDATE last_grounded_point SET velocity_normal = ? WHERE rowid = ?""",(float(velo_normal), int(row["rowid"])))
                    cursor.execute("""UPDATE last_grounded_point SET buttressing_natural = ? WHERE rowid = ?""",(float(b_n), int(row["rowid"])))
                    conn.commit()

conn.commit()
conn.close()


print('Everything seems okay ~(°w°~)', flush=True)
print('END OF COMPARISON PROGRAM', flush=True)