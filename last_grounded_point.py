import config
import sqlite3

# ---------------------------------- COMPARISON OF THE FLUX AT THE GROUNDING LINE -----------------------------------------------
# AUTHOR: Marine de Coatpont
# November 12, 2025
# IGE / PhD
#
# This script extracts, for each (simulation, experiment, x, y),
# the last available time step from flux_data_2 and stores it in
# a new table: last_grounded_point
# -------------------------------------------------------------------------------------------------------------------------------

print('START OF COMPARISON PROGRAM', flush=True)

test = False
resolutions = [16]


for reso in resolutions:
    if test:
        db_path = f"{config.SAVE_PATH}/Flux_DataBase_test_{reso}km.db"
    else:
        db_path = f"/Users/lebescom/DataBase_{reso}km.db"

    print("Connecting to database", flush=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    #cursor.execute("DROP TABLE IF EXISTS last_grounded_point")
    
    
    try:
        cursor.execute("""
        CREATE TABLE  last_grounded_point_ (
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
            flotaison REAL
        );
        """)
    except sqlite3.OperationalError:
        pass

    queveryr = f"""ALTER TABLE last_grounded_point_ ADD COLUMN R_drag REAL"""
    queveryr_m = f"""ALTER TABLE last_grounded_point_ ADD COLUMN driving_stress REAL"""
    queveryslope_m = f"""ALTER TABLE last_grounded_point_ ADD COLUMN slope_max REAL"""
    queveryslope_f = f"""ALTER TABLE last_grounded_point_ ADD COLUMN slope_flux REAL"""
    try:
        cursor.execute(queveryr)
        cursor.execute(queveryr_m)
        cursor.execute(queveryslope_m)
        cursor.execute(queveryslope_f)
    except sqlite3.OperationalError:
        pass

    conn.commit()
    insert_query = """
    INSERT INTO last_grounded_point_
    SELECT
        f.x,
        f.y,
        f.simulation,
        f.experiment,
        f.time,
        f.flux,
        f.thickness,
        f.velocity,
        f.drag,
        f.viscosity,
        f.surface,
        f.base,
        f.bed,
        f.flotaison,
        f.R_drag,
        f.driving_stress,
        f.slope_max,
        f.slope_flux
    FROM flux_data f
    JOIN (
        SELECT
            simulation,
            experiment,
            x,
            y,
            MAX(time) AS max_time
        FROM flux_data
        GROUP BY simulation, experiment, x, y
    ) m
    ON  f.simulation = m.simulation
    AND f.experiment = m.experiment
    AND f.x = m.x
    AND f.y = m.y
    AND f.time = m.max_time;
    """

    cursor.execute(insert_query)
    conn.commit()

    cursor.execute("""ALTER TABLE last_grounded_point_ ADD COLUMN n_simulations INTEGER""")
    cursor.execute("""ALTER TABLE last_grounded_point_ ADD COLUMN n_timesteps INTEGER""")
    conn.commit()

    cursor.execute("""
    CREATE TEMP TABLE xy_stats AS
    SELECT
        x,
        y,
        COUNT(DISTINCT simulation) AS n_simulations,
        COUNT(time) AS n_timesteps
    FROM last_grounded_point_
    GROUP BY x, y;
    """)
    conn.commit()

    cursor.execute("""
    UPDATE last_grounded_point_
    SET
        n_simulations = (
            SELECT n_simulations
            FROM xy_stats s
            WHERE s.x = last_grounded_point_.x AND s.y = last_grounded_point_.y
        ),
        n_timesteps = (
            SELECT n_timesteps
            FROM xy_stats s
            WHERE s.x = last_grounded_point_.x AND s.y = last_grounded_point_.y
        );
    """)
    conn.commit()
    conn.close()

    conn.close()

print('Everything seems okay ~(°w°~)', flush=True)
print('END OF COMPARISON PROGRAM', flush=True)