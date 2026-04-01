import config
import sqlite3

# ---------------------------------- COMPARISON OF THE FLUX AT THE GROUNDING LINE -----------------------------------------------
# AUTHOR: Marine de Coatpont
# November 12, 2025
# IGE / PhD
#
# This script extracts, for each (simulation, experiment, x, y),
# the last available time step from flux_data and stores it in
# a new table: last_grounded_point
# -------------------------------------------------------------------------------------------------------------------------------

print('START OF PROGRAM', flush=True)

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

    #CALCUL DU TEMPS DE RÉSIDENCE DANS FLUX_DATA 
    print("Computing residence time...", flush=True)

    try:
        cursor.execute("""ALTER TABLE flux_data ADD COLUMN residence INTEGER""")
    except sqlite3.OperationalError:
        print("Column 'residence' already exists", flush=True)

    cursor.execute("""
            CREATE TEMP TABLE residence_stats AS
            SELECT
                x,
                y,
                simulation,
                experiment,
                COUNT(time) AS residence
            FROM flux_data
            GROUP BY x, y, simulation, experiment
    """)

    cursor.execute("""
            UPDATE flux_data
            SET residence = (
                SELECT r.residence
                FROM residence_stats r
                WHERE r.x = flux_data.x
                AND r.y = flux_data.y
                AND r.simulation = flux_data.simulation
                AND r.experiment = flux_data.experiment
            )
    """)

    conn.commit()

    #CREATION DE LA TABLE LAST_GROUNDED_POINT
    try:
        cursor.execute("""
        CREATE TABLE last_grounded_point (
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

    # Ajout colonnes
    for q in [
        "ALTER TABLE last_grounded_point ADD COLUMN R_drag REAL",
        "ALTER TABLE last_grounded_point ADD COLUMN driving_stress REAL",
        "ALTER TABLE last_grounded_point ADD COLUMN slope_max REAL",
        "ALTER TABLE last_grounded_point ADD COLUMN slope_flux REAL",
        "ALTER TABLE last_grounded_point ADD COLUMN residence INTEGER"
    ]:
        try:
            cursor.execute(q)
        except sqlite3.OperationalError:
            pass

    conn.commit()

    insert_query = """
    INSERT INTO last_grounded_point
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
        f.slope_flux,
        f.residence
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

print('Everything seems okay ~(°w°~)', flush=True)
print('END OF COMPARISON PROGRAM', flush=True)