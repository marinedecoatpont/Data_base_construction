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
    #cursor.execute("DROP TABLE IF EXISTS time_1_10")

    try:
        cursor.execute("""
        CREATE TABLE time_1_10 (
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
    conn.commit()
    insert_query = """
    INSERT INTO time_1_10
    SELECT *
    FROM flux_data
    WHERE time % 10 = 0;
    """

    cursor.execute(insert_query)
    conn.commit()

    cursor.execute("""ALTER TABLE time_1_10 ADD COLUMN n_simulations INTEGER""")
    cursor.execute("""ALTER TABLE time_1_10 ADD COLUMN n_timesteps INTEGER""")
    conn.commit()

    cursor.execute("""
    CREATE TEMP TABLE xy_stats AS
    SELECT
        x,
        y,
        COUNT(DISTINCT simulation) AS n_simulations,
        COUNT(time) AS n_timesteps
    FROM time_1_10
    GROUP BY x, y;
    """)
    conn.commit()

    cursor.execute("""
    UPDATE time_1_10
    SET
        n_simulations = (
            SELECT n_simulations
            FROM xy_stats s
            WHERE s.x = time_1_10.x AND s.y = time_1_10.y
        ),
        n_timesteps = (
            SELECT n_timesteps
            FROM xy_stats s
            WHERE s.x = time_1_10.x AND s.y = time_1_10.y
        );
    """)
    conn.commit()
    conn.close()

print('Everything seems okay ~(°w°~)', flush=True)
print('END OF COMPARISON PROGRAM', flush=True)
