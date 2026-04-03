"""
02_build_derived_tables.py
==========================
Creates two derived tables from flux_data:

  last_grounded_point
      For each (simulation, experiment, x, y), keeps only the row with
      the latest time step.  Adds n_simulations and n_timesteps per (x, y).

  time_1_10
      Subset of flux_data keeping every 10th time step (time % 10 == 0).
      Also adds n_simulations and n_timesteps per (x, y).

Both tables share the same schema as flux_data plus the two count columns.
"""

import sqlite3
import config


# =============================================================================
# HELPERS
# =============================================================================

EXTRA_COLS = [
    "ALTER TABLE {table} ADD COLUMN n_simulations INTEGER",
    "ALTER TABLE {table} ADD COLUMN n_timesteps   INTEGER",
]

RESIDENCE_SQL = """
ALTER TABLE flux_data ADD COLUMN residence INTEGER
"""

XY_STATS_SQL = """
CREATE TEMP TABLE xy_stats AS
SELECT
    x,
    y,
    COUNT(DISTINCT simulation) AS n_simulations,
    COUNT(time) AS n_timesteps
FROM {table}
GROUP BY x, y;
"""

UPDATE_COUNTS_SQL = """
UPDATE {table}
SET
    n_simulations = (SELECT n_simulations FROM xy_stats s
                     WHERE s.x = {table}.x AND s.y = {table}.y),
    n_timesteps   = (SELECT n_timesteps   FROM xy_stats s
                     WHERE s.x = {table}.x AND s.y = {table}.y);
"""

FLUX_DATA_COLUMNS = """
    x, y, simulation, experiment, time,
    flux, thickness, velocity, drag,
    surface, base, bed, flotaison,
    R_drag, driving_stress, slope_flux, slope_max,
    viscosity, buttressing, buttressing_natural
"""

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    x                    REAL,
    y                    REAL,
    simulation           TEXT,
    experiment           TEXT,
    time                 INTEGER,
    flux                 REAL,
    thickness            REAL,
    velocity             REAL,
    drag                 REAL,
    surface              REAL,
    base                 REAL,
    bed                  REAL,
    flotaison            REAL,
    R_drag               REAL,
    driving_stress       REAL,
    slope_flux           REAL,
    slope_max            REAL,
    viscosity            REAL,
    buttressing          REAL,
    buttressing_natural  REAL,
    residence            INTEGER
);
"""


def _add_extra_columns(cursor, table: str) -> None:
    for stmt in EXTRA_COLS:
        try:
            cursor.execute(stmt.format(table=table))
        except Exception:
            pass


def _fill_counts(cursor, conn, table: str) -> None:
    cursor.execute(XY_STATS_SQL.format(table=table))
    cursor.execute(UPDATE_COUNTS_SQL.format(table=table))
    conn.commit()


# RESIDENCE TIME
def _add_residence(cursor, conn) -> None:
    """Adds how many time steps each (x, y, simu, exp) appears in flux_data."""
    try:
        cursor.execute(RESIDENCE_SQL)
    except Exception:
        pass

    cursor.execute("""
    CREATE TEMP TABLE res_stats AS
    SELECT x, y, simulation, experiment, COUNT(time) AS residence
    FROM flux_data
    GROUP BY x, y, simulation, experiment;
    """)

    cursor.execute("""
    UPDATE flux_data
    SET residence = (
        SELECT r.residence FROM res_stats r
        WHERE r.x = flux_data.x
          AND r.y = flux_data.y
          AND r.simulation = flux_data.simulation
          AND r.experiment = flux_data.experiment
    );
    """)
    conn.commit()


# LAST_GROUNDED_POINT
def build_last_grounded_point(cursor, conn) -> None:
    print("Building last_grounded_point …", flush=True)

    cursor.execute(CREATE_TABLE_SQL.format(table="last_grounded_point"))
    _add_extra_columns(cursor, "last_grounded_point")
    conn.commit()

    cursor.execute(f"""
    INSERT INTO last_grounded_point ({FLUX_DATA_COLUMNS}, residence)
    SELECT {FLUX_DATA_COLUMNS}, residence
    FROM flux_data f
    JOIN (
        SELECT simulation, experiment, x, y, MAX(time) AS max_time
        FROM flux_data
        GROUP BY simulation, experiment, x, y
    ) m
      ON  f.simulation = m.simulation
      AND f.experiment = m.experiment
      AND f.x = m.x
      AND f.y = m.y
      AND f.time = m.max_time;
    """)
    conn.commit()

    _fill_counts(cursor, conn, "last_grounded_point")
    print("  last_grounded_point done.", flush=True)


# TIME_1_10
def build_time_1_10(cursor, conn) -> None:
    print("Building time_1_10 …", flush=True)

    cursor.execute(CREATE_TABLE_SQL.format(table="time_1_10"))
    _add_extra_columns(cursor, "time_1_10")
    conn.commit()

    cursor.execute(f"""
    INSERT INTO time_1_10 ({FLUX_DATA_COLUMNS}, residence)
    SELECT {FLUX_DATA_COLUMNS}, residence
    FROM flux_data
    WHERE time % 10 = 0;
    """)
    conn.commit()

    _fill_counts(cursor, conn, "time_1_10")
    print("  time_1_10 done.", flush=True)


for reso in config.RESOLUTIONS:
    db_path = config.get_db_path(reso)
    print(f"\n=== Resolution {reso} km | DB: {db_path} ===", flush=True)

    conn   = sqlite3.connect(db_path)
    cursor = conn.cursor()

    _add_residence(cursor, conn)
    build_last_grounded_point(cursor, conn)
    build_time_1_10(cursor, conn)

    conn.close()

print("Done.", flush=True)
