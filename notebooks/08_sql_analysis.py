# ============================================================
# NOTEBOOK 08 — Création base SQLite & Requêtes Analytiques
# Projet : Analyse Inclusion Financière ASS
# Auteur : Ronald Dossou-Kohi
# ============================================================

# %% CELLULE 1 — Imports
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

ROOT      = Path("..")
DATA_PROC = ROOT / "data" / "processed"
SQL_DIR   = ROOT / "sql"
SQL_DIR.mkdir(exist_ok=True)

print("✅ Imports OK")


# %% CELLULE 2 — Créer la base SQLite
print("⏳ Création de la base SQLite...")

conn = sqlite3.connect(DATA_PROC / "kiva_ssa.db")

# Charger les tables
loans   = pd.read_parquet(DATA_PROC / "loans_ssa_mpi.parquet")
country = pd.read_csv(DATA_PROC / "powerbi" / "dim_country.csv",  sep=";", decimal=",")
sector  = pd.read_csv(DATA_PROC / "powerbi" / "dim_sector.csv",   sep=";", decimal=",")
partner = pd.read_csv(DATA_PROC / "powerbi" / "dim_partner.csv",  sep=";", decimal=",")

# Convertir booléens en 0/1 pour SQLite
bool_cols = ["is_female","is_fully_funded","is_group","is_unfunded","region_missing"]
for col in bool_cols:
    if col in loans.columns:
        loans[col] = loans[col].map({True:1, False:0, "True":1, "False":0, 1:1, 0:0})

# Écrire les tables dans SQLite
loans.to_sql("fact_loans",   conn, if_exists="replace", index=False)
country.to_sql("dim_country", conn, if_exists="replace", index=False)
sector.to_sql("dim_sector",   conn, if_exists="replace", index=False)
partner.to_sql("dim_partner", conn, if_exists="replace", index=False)

conn.commit()
print("✅ Base SQLite créée → data/processed/kiva_ssa.db")

# Vérification
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print(f"\n   Tables créées :")
for t in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {t[0]}")
    count = cursor.fetchone()[0]
    print(f"   {t[0]:<20} {count:>8,} lignes")

conn.close()


# %% CELLULE 3 — Exécuter et afficher les requêtes SQL
# On relit le fichier SQL et on exécute chaque requête
conn = sqlite3.connect(DATA_PROC / "kiva_ssa.db")

with open(SQL_DIR / "kiva_analysis.sql", "r", encoding="utf-8") as f:
    sql_content = f.read()

# Séparer les requêtes par le délimiteur qu'on a défini
queries = [q.strip() for q in sql_content.split("-- @@") if q.strip()]

for query in queries:
    # Extraire le titre (première ligne commentaire)
    lines  = query.strip().split("\n")
    title  = lines[0].replace("--","").strip()
    sql    = "\n".join(lines[1:]).strip()

    if not sql:
        continue

    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    try:
        result = pd.read_sql_query(sql, conn)
        print(result.to_string(index=False))
    except Exception as e:
        print(f"   ❌ Erreur : {e}")

conn.close()
print(f"\n✅ Toutes les requêtes exécutées")