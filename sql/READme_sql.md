# Requêtes SQL — Analyse Kiva ASS

Base de données : SQLite (`data/processed/kiva_ssa.db`)  
Générée par : `notebooks/08_sql_analysis.py`

## Requêtes disponibles

| # | Question analytique | Finding |
|---|--------------------|---------| 
| Q1 | Adéquation sectorielle Kiva vs emploi ASS | Gap agriculture -21.5pts |
| Q2 | Top 10 Field Partners par impact | 14 Champions identifiés |
| Q3 | Gap de genre par secteur | Ratio H/F moyen = 1.36x |
| Q4 | Pauvreté vs couverture par pays | ρ = -0.11, non significatif |
| Q5 | Saisonnalité agricole | Pic en mars, χ² p < 0.001 |

## Exécution
```bash
# Option 1 : Via Python (recommandé)
# Exécuter notebooks/08_sql_analysis.py cellule par cellule

# Option 2 : Via SQLite CLI
sqlite3 data/processed/kiva_ssa.db
.read sql/kiva_analysis.sql

# Option 3 : Extension VS Code
# Installer "SQLite Viewer" → ouvrir kiva_ssa.db → exécuter les requêtes
```

## Modèle de données
```
fact_loans ──── dim_country  (country)
           ──── dim_sector   (sector)
           ──── dim_partner  (partner_id)
```