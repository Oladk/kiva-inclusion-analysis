# ============================================================
# NOTEBOOK 01 : Première Exploration des Données Kiva
# Projet : Analyse Inclusion Financière ASS
# Auteur : Ronald Dossou-Kohi
# ============================================================
# Pour exécuter cellule par cellule dans VS Code :
# Installez l'extension "Jupyter" de Microsoft
# Chaque bloc "# %%" est une cellule exécutable (Ctrl+Enter)
# ============================================================

# %% CELLULE 1 : Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

# Chemins
ROOT     = Path("..") 
DATA_RAW = ROOT / "data" / "raw"

print("✅ Imports OK")
print(f"   pandas  : {pd.__version__}")
print(f"   numpy   : {np.__version__}")


# %% CELLULE 2 : Chargement (peut prendre 15-20 secondes)
print("⏳ Chargement de kiva_loans.csv...")

loans = pd.read_csv(DATA_RAW / "kiva_loans.csv", low_memory=False)

print(f"Chargé : {loans.shape[0]:,} lignes × {loans.shape[1]} colonnes")


# %% CELLULE 3 : Structure de base
print("=" * 55)
print(" STRUCTURE DU DATASET")
print("=" * 55)

print(f"\n Dimensions    : {loans.shape[0]:,} prêts × {loans.shape[1]} variables")
print(f" Mémoire       : {loans.memory_usage(deep=True).sum() / 1e6:.1f} Mo")
print(f"\n Colonnes :\n")
print(loans.dtypes.to_string())


# %% CELLULE 4 : Aperçu des premières lignes
loans.head(3)


# %% CELLULE 5 : Valeurs manquantes
print("=" * 55)
print(" VALEURS MANQUANTES")
print("=" * 55)

missing = pd.DataFrame({
    "n_manquants" : loans.isnull().sum(),
    "pct_%"       : (loans.isnull().sum() / len(loans) * 100).round(2)
}).sort_values("pct_%", ascending=False)

print(missing[missing["n_manquants"] > 0].to_string())


# %% CELLULE 6 : Statistiques descriptives (variables numériques)
print("=" * 55)
print(" STATISTIQUES DESCRIPTIVES")
print("=" * 55)

num_stats = loans.select_dtypes(include=[np.number]).describe().T
num_stats["cv_%"] = (num_stats["std"] / num_stats["mean"] * 100).round(1)
print(num_stats[["count","mean","std","min","25%","50%","75%","max","cv_%"]].round(2).to_string())


# %% CELLULE 7 : Distribution des montants de prêts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribution des Montants de Prêts — Kiva Global", 
             fontsize=13, fontweight="bold")

loan_data = loans["loan_amount"].dropna()

# Graphique 1 : Distribution brute
axes[0].hist(loan_data, bins=100, color="#1A6B5C", alpha=0.75, edgecolor="white")
axes[0].set_title("Distribution brute")
axes[0].set_xlabel("Montant ($)")
axes[0].set_ylabel("Fréquence")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# Graphique 2 : Sans les extrêmes (< P99)
p99 = loan_data.quantile(0.99)
axes[1].hist(loan_data[loan_data <= p99], bins=80, 
             color="#E8840B", alpha=0.75, edgecolor="white")
axes[1].set_title(f"Sans extrêmes (< P99 = ${p99:,.0f})")
axes[1].set_xlabel("Montant ($)")

# Graphique 3 : Log-transformé
axes[2].hist(np.log1p(loan_data), bins=80, 
             color="#2E86AB", alpha=0.75, edgecolor="white")
axes[2].set_title("Log-transformé [log(1 + montant)]")
axes[2].set_xlabel("log(1 + montant)")

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(ROOT / "reports" / "figures" / "01_loan_amount_distribution.png", 
            dpi=150, bbox_inches="tight")
plt.show()

print(f"\n Statistiques — loan_amount :")
print(f"   Médiane        : ${loan_data.median():>10,.0f}")
print(f"   Moyenne        : ${loan_data.mean():>10,.0f}")
print(f"   Ratio moy/méd  : {loan_data.mean()/loan_data.median():>10.2f}x")
print(f"   P25 / P75      :  ${loan_data.quantile(.25):,.0f}  /  ${loan_data.quantile(.75):,.0f}")
print(f"   P99            : ${loan_data.quantile(.99):>10,.0f}")


# %% CELLULE 8 : Top pays
print("=" * 55)
print(" COUVERTURE GÉOGRAPHIQUE")
print("=" * 55)

country_counts = loans["country"].value_counts()
total          = len(loans)

print(f"\n Nombre de pays distincts : {loans['country'].nunique()}")
print(f"\nTop 20 pays par nombre de prêts :\n")

top20 = pd.DataFrame({
    "n_prêts"    : country_counts.head(20),
    "pct_%"      : (country_counts.head(20) / total * 100).round(2),
    "cumulé_%"   : (country_counts.head(20) / total * 100).cumsum().round(2)
})
print(top20.to_string())


# %% CELLULE 9 : Secteurs d'activité
print("=" * 55)
print(" SECTEURS D'ACTIVITÉ")
print("=" * 55)

sector_counts = loans["sector"].value_counts()
sector_pct    = (sector_counts / total * 100).round(2)

print(f"\n {loans['sector'].nunique()} secteurs distincts\n")
for sector, pct in sector_pct.items():
    bar = "█" * int(pct / 1.5)
    print(f"  {sector:<22} {bar:<30} {pct:>5.1f}%")


# %% CELLULE 10 : Variable genre (aperçu brut)
print("=" * 55)
print(" VARIABLE GENRE (brut)")
print("=" * 55)

print(f"\nValeurs uniques (Top 10) :")
print(loans["borrower_genders"].value_counts().head(10).to_string())

print(f"\nValeurs manquantes : {loans['borrower_genders'].isnull().sum():,}")
print(f"Taux manquants     : {loans['borrower_genders'].isnull().mean()*100:.1f}%")


# %% CELLULE 11 : Temporalité
print("=" * 55)
print(" COUVERTURE TEMPORELLE")
print("=" * 55)

loans["posted_time"] = pd.to_datetime(loans["posted_time"], errors="coerce", utc=True)
loans["posted_year"] = loans["posted_time"].dt.year

annual = loans.groupby("posted_year").agg(
    n_prêts      = ("id",          "count"),
    volume_total = ("loan_amount", "sum"),
    montant_moy  = ("loan_amount", "mean")
).dropna()

annual["croissance_%"] = annual["n_prêts"].pct_change() * 100

print(f"\n Première publication : {loans['posted_time'].min()}")
print(f"   Dernière publication  : {loans['posted_time'].max()}")
print(f"\n Volume par année :\n")
print(annual.round(1).to_string())


# %% CELLULE 12 : Sauvegarde des stats descriptives
output_path = ROOT / "data" / "processed" / "01_stats_descriptives.csv"
loans.describe(include="all").T.to_csv(output_path)

print(f" Stats descriptives sauvegardées → {output_path}")
print(f"\n Notebook 01 terminé.")
