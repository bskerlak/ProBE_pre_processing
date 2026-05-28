"""
Pre-processing MIDI: read Anriss layer from all GDBs, assign global IDs,
tag AWN sub-region membership, and expand to per-depth simulations.

Outputs (in input/midi_preprocessed/):
  prozessquelle.parquet  - Xurce canton-wide prozessgebiet polygons (id_pq)
  testgebiete.parquet     - AWN test sub-region polygons
  job_manifest.parquet    - One row per (anriss × h_type):
                            id_prozessquelle, id_anriss, id_anriss_original, x, y, area_m2, d_cm,
                            h_type, h_cm, AWN_teilgebiet
"""

import os
import duckdb

INPUT = "/home/bojan/probe_control_center/input"
OUTPUT = INPUT
os.makedirs(OUTPUT, exist_ok=True)

# Set True to keep only Anriss points inside AWN testgebiete
FILTER_TO_TESTGEBIETE = True

con = duckdb.connect()
con.execute("SET memory_limit = '8GB'")
con.execute(f"SET temp_directory = '{OUTPUT}/tmp'")
con.execute("INSTALL spatial; LOAD spatial;")

# ── 1. process_source ────────────────────────────────────────────────────────
print("Converting process_source …")
ps_path = os.path.join(INPUT, "Xurce - Prozessquellen", "process_source.shp")
out = os.path.join(OUTPUT, "prozessquelle.parquet")
con.execute(f"""
    COPY (
        SELECT id_pq, Shape_Area AS area_m2, geom
        FROM ST_Read('{ps_path}')
    ) TO '{out}'
""")
n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
print(f"  {n} features → prozessquelle.parquet")

# ── 2. AWN testgebiete ───────────────────────────────────────────────────────
print("Converting AWN testgebiete …")
awn_path = os.path.join(
    INPUT,
    "Geo7 - ProBE_MIDI_Gebiete_20260522",
    "ProBE_MIDI_Gebiete_20260522.shp",
)
con.execute(f"""
    CREATE TABLE testgebiete AS
    SELECT Testperime AS region, OID_Prozes AS oid_prozessquelle, geom
    FROM ST_Read('{awn_path}')
""")
out = os.path.join(OUTPUT, "testgebiete.parquet")
con.execute(f"COPY testgebiete TO '{out}'")
n, regions = con.execute("SELECT COUNT(*), LIST(region) FROM testgebiete").fetchone()
print(f"  {n} features → testgebiete.parquet")
print("  Regions:", regions)

# ── 3. Anrisse from all GDBs ─────────────────────────────────────────────────
print("\nReading Anriss layer from GDBs …")
gdb_base = os.path.join(INPUT, "Xurce - MINI Teil B")
con.execute("""
    CREATE TABLE anrisse_raw (
        id_anriss_original INTEGER,
        id_prozessquelle   INTEGER,
        x                  DOUBLE,
        y                  DOUBLE,
        d_cm               INTEGER,
        area_m2            DOUBLE
    )
""")

total = 0
for folder in sorted(os.listdir(gdb_base)):
    folder_path = os.path.join(gdb_base, folder)
    if not os.path.isdir(folder_path):
        continue
    gdb = os.path.join(folder_path, "modellinput.gdb")
    if not os.path.isdir(gdb):
        continue
    con.execute(f"""
        INSERT INTO anrisse_raw
        SELECT
            id_prozessquelle,
            id_anriss                         AS id_anriss_original,
            x,
            y,
            ROUND(bodengruendigkeit)::INTEGER AS d_cm,
            anrissflaeche                     AS area_m2
        FROM ST_Read('{gdb}', layer='Anriss')
    """)
    new_total = con.execute("SELECT COUNT(*) FROM anrisse_raw").fetchone()[0]
    print(f"  {folder}: {new_total - total} anrisse")
    total = new_total

# Assign globally unique id_anriss
con.execute("""
    CREATE TABLE anrisse AS
    SELECT
        id_prozessquelle,
        ROW_NUMBER() OVER () AS id_anriss,
        id_anriss_original,
        x,
        y,
        d_cm,
        area_m2
    FROM anrisse_raw
""")
print(f"  Total: {total} anrisse")
rows = con.execute("SELECT area_m2, COUNT(*) AS n FROM anrisse GROUP BY area_m2 ORDER BY area_m2").fetchall()
print("  By area_m2 (m²):")
for area, n in rows:
    print(f"    {area:>8.0f} m²: {n:>6} anrisse")
rows = con.execute("SELECT d_cm <= 30 AS shallow, COUNT(*) FROM anrisse GROUP BY 1 ORDER BY 1 DESC").fetchall()
for shallow, n in rows:
    label = "d ≤ 30 cm (2 sims)" if shallow else "d > 30 cm (3 sims)"
    print(f"  {label}: {n}")

# ── 4. Tag AWN sub-region membership ─────────────────────────────────────────
print("\nTagging AWN_teilgebiet …")
con.execute("""
    CREATE TABLE anrisse_flagged AS
    SELECT
        a.id_prozessquelle,
        a.id_anriss,
        a.id_anriss_original,
        a.x,
        a.y,
        a.area_m2,
        a.d_cm,
        COUNT(t.region) > 0 AS AWN_teilgebiet
    FROM anrisse a
    LEFT JOIN testgebiete t ON ST_Within(ST_Point(a.x, a.y), t.geom)
    GROUP BY a.id_prozessquelle, a.id_anriss, a.id_anriss_original, a.x, a.y, a.area_m2, a.d_cm
""")
n_awn = con.execute("SELECT COUNT(*) FROM anrisse_flagged WHERE AWN_teilgebiet").fetchone()[0]
print(f"  {n_awn} of {total} anrisse within AWN sub-regions ({100*n_awn/total:.1f}%)")

if FILTER_TO_TESTGEBIETE:
    print("\nFiltering to AWN testgebiete only …")
    con.execute("""
        CREATE TABLE anrisse_in_testgebiete AS
        SELECT
            id_prozessquelle,
            id_anriss,
            id_anriss_original,
            x,
            y,
            area_m2,
            d_cm
        FROM anrisse_flagged
        WHERE AWN_teilgebiet
    """)
    source_table = "anrisse_in_testgebiete"
else:
    source_table = "anrisse_flagged"

# ── 5. Expand to simulations ──────────────────────────────────────────────────
# d ≤ 30 cm → h_min, h_max          (2 rows)
# d > 30 cm → h_min, h_mean, h_max  (3 rows)
print("\nExpanding to job manifest …")
out_manifest = os.path.join(OUTPUT, "midi_job_manifest.parquet")
con.execute(f"""
    COPY (
        SELECT *, 'min'  AS h_type, ROUND(0.5 * d_cm)::INTEGER AS h_cm FROM {source_table}
        UNION ALL
        SELECT *, 'mean' AS h_type,       d_cm  AS h_cm FROM {source_table} WHERE d_cm > 30
        UNION ALL
        SELECT *, 'max'  AS h_type, ROUND(1.5 * d_cm)::INTEGER AS h_cm FROM {source_table}
        ORDER BY id_anriss, h_cm
    ) TO '{out_manifest}'
""")
n_sim = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_manifest}')").fetchone()[0]
print(f"  {n_sim} rows → job_manifest.parquet  (expected {total*2}–{total*3})")

# add points for easy visualization in QGIS
out_manifest_qgis = os.path.join(OUTPUT, "midi_job_manifest_qgis.parquet")
con.execute(f"""
    COPY (SELECT *, ST_Point(x,y) from read_parquet('{out_manifest}'))
    TO '{out_manifest_qgis}'
    """)

out_summary = os.path.join(OUTPUT, "summary.csv")
con.execute(f"""
    COPY (
        SELECT
            id_prozessquelle,
            id_anriss,
            area_m2,
            COUNT(*) FILTER (WHERE h_type = 'min')  AS n_min,
            COUNT(*) FILTER (WHERE h_type = 'mean') AS n_mean,
            COUNT(*) FILTER (WHERE h_type = 'max')  AS n_max,
            COUNT(*)                                 AS n_total
        FROM read_parquet('{out_manifest}')
        GROUP BY id_prozessquelle, id_anriss, area_m2
        ORDER BY id_prozessquelle, id_anriss, area_m2
    ) TO '{out_summary}' (HEADER, DELIMITER ',')
""")
rows = con.execute(f"SELECT * FROM read_csv('{out_summary}')").fetchall()
print(f"\n  {'prozessquelle':>14}  {'area_m2':>8}  {'n_min':>6}  {'n_mean':>6}  {'n_max':>6}  {'n_total':>7}")
for pq, area, n_min, n_mean, n_max, n_total in rows:
    print(f"  {pq:>14}  {area:>8.0f}  {n_min:>6}  {n_mean:>6}  {n_max:>6}  {n_total:>7}")

print("\nDone. Files written to:", OUTPUT)
for f in sorted(os.listdir(OUTPUT)):
    if f == "tmp":
        continue
    size_mb = os.path.getsize(os.path.join(OUTPUT, f)) / 1e6
    print(f"  {f}: {size_mb:.1f} MB")
