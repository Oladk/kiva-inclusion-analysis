# ============================================================
# NOTEBOOK 07 — Export des données pour Power BI
# Projet : Analyse Inclusion Financière ASS
# Auteur : Ronald Dossou-Kohi
# ============================================================

# %% CELLULE 1 — Imports
import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT      = Path("..")
DATA_PROC = ROOT / "data" / "processed"
DATA_PBI  = ROOT / "data" / "processed" / "powerbi"
DATA_PBI.mkdir(parents=True, exist_ok=True)

print("✅ Imports OK")
print(f"   Export vers : {DATA_PBI}")


# %% CELLULE 2 — Chargement
loans = pd.read_parquet(DATA_PROC / "loans_ssa_mpi.parquet")
print(f"✅ {len(loans):,} prêts ASS chargés")


# %% CELLULE 3 — Table FAITS : un prêt par ligne
# ─────────────────────────────────────────────────────────
# Power BI fonctionne avec un modèle en étoile :
#   - 1 table de FAITS (les prêts)
#   - N tables de DIMENSIONS (pays, secteur, partenaire...)
#
# On garde uniquement les colonnes utiles pour les visuels.
# Moins de colonnes = fichier plus léger = dashboard plus rapide.
# ─────────────────────────────────────────────────────────

fact_loans = loans[[
    "id",
    "loan_amount",
    "funded_amount",
    "funding_ratio",
    "is_fully_funded",
    "term_in_months",
    "country",
    "country_code",
    "sub_region",
    "sector",
    "activity",
    "gender_clean",
    "is_female",
    "partner_id",
    "posted_year",
    "posted_month",
    "days_to_fund",
    "MPI_final",
    "loan_size_category",
    "repayment_interval",
    "log_loan_amount",
]].copy()

# Convertir loan_size_category en string (Power BI gère mieux)
fact_loans["loan_size_category"] = fact_loans["loan_size_category"].astype(str)

# Nettoyer les valeurs infinies
fact_loans = fact_loans.replace([np.inf, -np.inf], np.nan)

print(f"✅ Table FAITS : {len(fact_loans):,} lignes × {len(fact_loans.columns)} colonnes")
print(f"   Taille mémoire : {fact_loans.memory_usage(deep=True).sum()/1e6:.1f} Mo")


# %% CELLULE 4 — Table DIMENSION : Pays
dim_country = (
    loans
    .groupby(["country","country_code","sub_region"])
    .agg(
        n_loans       = ("id",             "count"),
        total_volume  = ("loan_amount",    "sum"),
        median_amount = ("loan_amount",    "median"),
        pct_female    = ("is_female",      "mean"),
        avg_mpi       = ("MPI_final",      "mean"),
        n_partners    = ("partner_id",     "nunique"),
    )
    .reset_index()
)
dim_country["pct_female"] *= 100
dim_country["pct_of_ssa"]  = dim_country["n_loans"] / len(loans) * 100

# Population adulte (ONU 2020) pour la pénétration
POP_ADULTES = {
    "Kenya":27.0,"Uganda":18.0,"Tanzania":25.0,"Rwanda":7.3,
    "Ethiopia":62.0,"Mozambique":16.0,"Nigeria":108.0,"Ghana":18.0,
    "Senegal":9.0,"Mali":9.0,"Burkina Faso":9.0,"Togo":4.2,
    "Benin":5.5,"Niger":9.0,"Sierra Leone":3.5,"Liberia":2.5,
    "Guinea":7.0,"Cameroon":13.0,"Madagascar":14.0,"Malawi":10.0,
    "Zambia":10.0,"Zimbabwe":10.0,"South Africa":39.0,"Lesotho":1.5,
    "Democratic Republic Of The Congo":45.0,"Congo":2.5,
}
dim_country["pop_adultes_M"]       = dim_country["country"].map(POP_ADULTES)
dim_country["prets_p1000_adultes"] = (
    dim_country["n_loans"] / (dim_country["pop_adultes_M"] * 1000)
).round(2)

print(f"✅ Table PAYS : {len(dim_country)} pays")


# %% CELLULE 5 — Table DIMENSION : Secteur
EMPLOI_BENCHMARK = {
    "Agriculture":54.0,"Food":10.0,"Retail":8.0,"Arts":4.0,
    "Services":7.0,"Education":3.0,"Health":2.0,"Housing":2.0,
    "Transportation":3.0,"Manufacturing":3.0,"Clothing":2.0,
    "Personal Use":1.0,"Construction":1.0,"Entertainment":0.0,"Wholesale":0.0,
}

dim_sector = (
    loans
    .groupby("sector")
    .agg(
        n_loans       = ("id",          "count"),
        total_volume  = ("loan_amount", "sum"),
        median_amount = ("loan_amount", "median"),
        pct_female    = ("is_female",   "mean"),
        avg_mpi       = ("MPI_final",   "mean"),
    )
    .reset_index()
)
dim_sector["pct_female"]   *= 100
dim_sector["pct_kiva"]      = dim_sector["n_loans"] / len(loans) * 100
dim_sector["pct_emploi"]    = dim_sector["sector"].map(EMPLOI_BENCHMARK).fillna(0)
dim_sector["gap_allocation"] = dim_sector["pct_kiva"] - dim_sector["pct_emploi"]

print(f"✅ Table SECTEUR : {len(dim_sector)} secteurs")


# %% CELLULE 6 — Table DIMENSION : Field Partners
dim_partner = (
    loans[loans["partner_id"].notna()]
    .groupby("partner_id")
    .agg(
        n_loans          = ("id",              "count"),
        total_volume     = ("loan_amount",     "sum"),
        median_amount    = ("loan_amount",     "median"),
        pct_female       = ("is_female",       "mean"),
        pct_funded       = ("is_fully_funded", "mean"),
        avg_mpi          = ("MPI_final",       "mean"),
        avg_days_to_fund = ("days_to_fund",    "mean"),
        n_countries      = ("country",         "nunique"),
        n_sectors        = ("sector",          "nunique"),
    )
    .query("n_loans >= 100")
    .reset_index()
)
dim_partner["pct_female"] *= 100
dim_partner["pct_funded"] *= 100

# Profil du partner
mpi_med    = dim_partner["avg_mpi"].median()
female_med = dim_partner["pct_female"].median()

def profil_partner(row):
    if pd.isna(row["avg_mpi"]):
        return "Non classifié"
    if row["pct_female"] >= female_med and row["avg_mpi"] >= mpi_med:
        return "Champion"
    elif row["pct_female"] >= female_med:
        return "Pro-genre"
    elif row["avg_mpi"] >= mpi_med:
        return "Pro-pauvres"
    else:
        return "Standard"

dim_partner["profil"] = dim_partner.apply(profil_partner, axis=1)

print(f"✅ Table FIELD PARTNERS : {len(dim_partner)} partenaires")
print(f"\n   Répartition des profils :")
print(dim_partner["profil"].value_counts().to_string())


# %% CELLULE 7 — Table DIMENSION : Temporelle
dim_temps = (
    loans[loans["posted_year"].notna()]
    .groupby(["posted_year","posted_month"])
    .agg(
        n_loans      = ("id",             "count"),
        total_volume = ("loan_amount",    "sum"),
        avg_amount   = ("loan_amount",    "mean"),
        pct_female   = ("is_female",      "mean"),
        pct_funded   = ("is_fully_funded","mean"),
    )
    .reset_index()
)
dim_temps["pct_female"] *= 100
dim_temps["pct_funded"] *= 100
dim_temps["posted_year"] = dim_temps["posted_year"].astype(int)
dim_temps["posted_month"]= dim_temps["posted_month"].astype(int)

# Ajouter noms des mois pour Power BI
mois_noms = {1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
             7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"}
dim_temps["mois_nom"] = dim_temps["posted_month"].map(mois_noms)

print(f"✅ Table TEMPORELLE : {len(dim_temps)} combinaisons année-mois")


# %% CELLULE 8 — Table KPI SYNTHÈSE (pour les cartes Power BI)
kpi_data = {
    "KPI"    : [
        "Total Prêts ASS",
        "Volume Total ($M)",
        "% Femmes",
        "Taux Financement Complet",
        "Montant Médian ($)",
        "Délai Médian Financement (jours)",
        "Pays Couverts",
        "Field Partners Actifs",
        "Gini Géographique",
        "Partners Champions",
        "Gap Agriculture (pts)",
        "AUC Modèle RF",
    ],
    "Valeur" : [
        len(loans),
        round(loans["loan_amount"].sum() / 1e6, 1),
        round(loans["is_female"].mean() * 100, 1),
        round(loans["is_fully_funded"].mean() * 100, 1),
        round(loans["loan_amount"].median(), 0),
        round(loans[loans["days_to_fund"].between(0,90)]["days_to_fund"].median(), 1),
        loans["country"].nunique(),
        72,
        0.704,
        14,
        -21.5,
        0.866,
    ],
    "Unité"  : [
        "prêts","M$","%","%","$","jours","pays","partenaires","indice","partenaires","pts",""
    ],
    "Source" : [
        "Kiva","Kiva","Kiva","Kiva","Kiva","Kiva","Kiva",
        "Notebook 06","Notebook 03","Notebook 06","Notebook 04","Notebook 06"
    ]
}
dim_kpi = pd.DataFrame(kpi_data)
print(f"✅ Table KPI : {len(dim_kpi)} indicateurs")
print(dim_kpi.to_string(index=False))


# %% CELLULE 9 — Export CSV pour Power BI
exports = {
    "fact_loans.csv"   : fact_loans,
    "dim_country.csv"  : dim_country,
    "dim_sector.csv"   : dim_sector,
    "dim_partner.csv"  : dim_partner,
    "dim_temps.csv"    : dim_temps,
    "dim_kpi.csv"      : dim_kpi,
}

print(f"\n💾 Export des tables Power BI :")
for filename, df in exports.items():
    path = DATA_PBI / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    size_kb = path.stat().st_size / 1024
    print(f"   ✅ {filename:<25} {len(df):>7,} lignes  {size_kb:>8.1f} Ko")

print(f"\n{'='*55}")
print(f" EXPORT TERMINÉ")
print(f"{'='*55}")
print(f"\n Fichiers dans : data/processed/powerbi/")
print(f"\n Ouvrir Power BI Desktop et importer :")
print(f"   1. fact_loans.csv   → Table principale")
print(f"   2. dim_country.csv  → Onglet géographie")
print(f"   3. dim_sector.csv   → Onglet sectoriel")
print(f"   4. dim_partner.csv  → Onglet field partners")
print(f"   5. dim_kpi.csv      → Cartes KPI (onglet exécutif)")
# %%
