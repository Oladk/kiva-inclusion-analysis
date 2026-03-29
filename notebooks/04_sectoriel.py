# ============================================================
# NOTEBOOK 04 — Analyse Sectorielle
# Projet : Analyse Inclusion Financière ASS
# Auteur : Ronald Dossou-Kohi
# ============================================================

# %% CELLULE 1 — Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

ROOT      = Path("..")
DATA_PROC = ROOT / "data" / "processed"
FIGURES   = ROOT / "reports" / "figures"

print("✅ Imports OK")


# %% CELLULE 2 — Chargement
loans = pd.read_parquet(DATA_PROC / "loans_ssa_mpi.parquet")
print(f"✅ {len(loans):,} prêts ASS chargés")
print(f"   Secteurs distincts : {loans['sector'].nunique()}")


# %% CELLULE 3 — Distribution sectorielle
# ─────────────────────────────────────────────────────────
# On calcule deux métriques complémentaires :
# pct_loans  → part en NOMBRE (combien de prêts ?)
# pct_volume → part en VALEUR (combien de dollars ?)
#
# Si pct_volume > pct_loans → les prêts de ce secteur
#   sont plus grands que la moyenne (secteur "premium")
# Si pct_volume < pct_loans → prêts plus petits que la
#   moyenne (secteur de subsistance, micro-entrepreneurs)
# ─────────────────────────────────────────────────────────
sector_stats = (
    loans
    .groupby("sector")
    .agg(
        n_loans       = ("id",             "count"),
        total_volume  = ("loan_amount",    "sum"),
        median_amount = ("loan_amount",    "median"),
        mean_amount   = ("loan_amount",    "mean"),
        pct_female    = ("is_female",      "mean"),
        avg_mpi       = ("MPI_final",      "mean"),
        avg_term      = ("term_in_months", "mean"),
    )
    .sort_values("n_loans", ascending=False)
    .reset_index()
)

sector_stats["pct_loans"]   = sector_stats["n_loans"]       / len(loans)          * 100
sector_stats["pct_volume"]  = sector_stats["total_volume"]  / loans["loan_amount"].sum() * 100
sector_stats["pct_female"]  = sector_stats["pct_female"]    * 100
sector_stats["vol_per_loan"]= sector_stats["total_volume"]  / sector_stats["n_loans"]

print("📊 Distribution sectorielle — ASS :\n")
cols = ["sector","n_loans","pct_loans","pct_volume","median_amount","pct_female","avg_mpi"]
print(sector_stats[cols].round(2).to_string(index=False))


# %% CELLULE 4 — Visualisation : distribution sectorielle
COLORS_SECTOR = [
    "#1A6B5C","#E8840B","#2E86AB","#C73E1D",
    "#F7B731","#6C5CE7","#00B894","#D63031",
    "#74B9FF","#A29BFE","#FD79A8","#FDCB6E",
    "#55EFC4","#636E72","#B2BEC3"
]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Analyse Sectorielle — Kiva Afrique Subsaharienne",
             fontsize=13, fontweight="bold")

# ── Graphique gauche : % prêts vs % volume ───────────────
x      = np.arange(len(sector_stats))
width  = 0.38
colors = COLORS_SECTOR[:len(sector_stats)]

bars1 = axes[0].barh(
    sector_stats["sector"][::-1],
    sector_stats["pct_loans"][::-1],
    height=width, label="% Nombre de prêts",
    color="#1A6B5C", alpha=0.85
)
bars2 = axes[0].barh(
    [s + " " for s in sector_stats["sector"][::-1]],
    sector_stats["pct_volume"][::-1],
    height=width, label="% Volume ($)",
    color="#E8840B", alpha=0.85
)

# Version simplifiée : une seule métrique (nombre) pour la lisibilité
axes[0].cla()
bars = axes[0].barh(
    sector_stats["sector"][::-1],
    sector_stats["pct_loans"][::-1],
    color=colors[::-1], alpha=0.85
)
for bar, val in zip(bars, sector_stats["pct_loans"][::-1]):
    axes[0].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)

axes[0].set_xlabel("% des prêts ASS")
axes[0].set_title("Part de chaque secteur (nombre de prêts)")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# ── Graphique droit : montant médian par secteur ─────────
sorted_median = sector_stats.sort_values("median_amount")
bar_colors2   = COLORS_SECTOR[:len(sorted_median)]

bars2 = axes[1].barh(
    sorted_median["sector"],
    sorted_median["median_amount"],
    color=bar_colors2, alpha=0.85
)
for bar, val in zip(bars2, sorted_median["median_amount"]):
    axes[1].text(val + 5, bar.get_y() + bar.get_height()/2,
                 f"${val:,.0f}", va="center", fontsize=9)

axes[1].set_xlabel("Montant médian du prêt ($)")
axes[1].set_title("Taille médiane des prêts par secteur")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES / "04_sector_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure sauvegardée → 04_sector_distribution.png")


# %% CELLULE 5 — Expected vs Actual (graphique clé pour Kintésens)
# ─────────────────────────────────────────────────────────
# CONCEPT :
# Si le microcrédit était parfaitement aligné sur les besoins
# économiques réels, sa répartition sectorielle refléterait
# la structure de l'emploi en ASS.
#
# BENCHMARKS EMPLOI ASS (source : Banque Mondiale / OIT 2022) :
# Agriculture  : 54% de l'emploi
# Commerce     : 18%
# Services     : 13%
# Artisanat    : 7%
# Éducation    : 3%
# Santé        : 2%
# Autres       : 3%
#
# L'écart Expected - Actual = le GAP d'allocation sectorielle.
# C'est le finding le plus actionnable pour une IMF.
# ─────────────────────────────────────────────────────────

EMPLOI_BENCHMARK = {
    "Agriculture"    : 54.0,
    "Food"           : 10.0,
    "Retail"         : 8.0,
    "Arts"           : 4.0,
    "Services"       : 7.0,
    "Education"      : 3.0,
    "Health"         : 2.0,
    "Housing"        : 2.0,
    "Transportation" : 3.0,
    "Manufacturing"  : 3.0,
    "Clothing"       : 2.0,
    "Personal Use"   : 1.0,
    "Construction"   : 1.0,
    "Entertainment"  : 0.0,
    "Wholesale"      : 0.0,
}

benchmark_df = pd.DataFrame({
    "sector"    : list(EMPLOI_BENCHMARK.keys()),
    "expected"  : list(EMPLOI_BENCHMARK.values())
})

comparison = sector_stats.merge(benchmark_df, on="sector", how="left")
comparison["expected"]  = comparison["expected"].fillna(0)
comparison["gap"]       = comparison["pct_loans"] - comparison["expected"]
comparison["gap_label"] = comparison["gap"].apply(
    lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
)
comparison = comparison.sort_values("gap", ascending=False)

print("📊 EXPECTED vs ACTUAL — Allocation sectorielle :")
print(f"{'Secteur':<20} {'Kiva%':>8} {'Emploi%':>9} {'Écart':>8}  Signal")
print("-" * 60)
for _, row in comparison.iterrows():
    signal = "↑ Sur-financé" if row["gap"] > 3 else "↓ Sous-financé" if row["gap"] < -3 else "~ Équilibré"
    print(f"  {row['sector']:<18} {row['pct_loans']:>7.1f}% {row['expected']:>8.1f}% {row['gap_label']:>8}  {signal}")


# %% CELLULE 6 — Visualisation Expected vs Actual
fig, ax = plt.subplots(figsize=(12, 8))

sectors_sorted = comparison.sort_values("gap")
gap_colors = ["#C73E1D" if g < 0 else "#1A6B5C" for g in sectors_sorted["gap"]]

bars = ax.barh(
    sectors_sorted["sector"],
    sectors_sorted["gap"],
    color=gap_colors, alpha=0.85
)

for bar, val, label in zip(bars, sectors_sorted["gap"], sectors_sorted["gap_label"]):
    x_pos = val + 0.3 if val >= 0 else val - 0.3
    ha    = "left"    if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height()/2,
            label, va="center", ha=ha, fontsize=9, fontweight="bold")

ax.axvline(x=0, color="black", linewidth=1.2)
ax.set_xlabel("Écart = % Kiva − % Emploi ASS (points de pourcentage)")
ax.set_title(
    "Adéquation Sectorielle : Kiva vs Structure de l'Emploi ASS\n"
    "Vert = sur-financé | Rouge = sous-financé (vs benchmark emploi Banque Mondiale/OIT)",
    fontsize=12, fontweight="bold"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.text(0.01, -0.09,
    "⚠️  Benchmark emploi : Banque Mondiale / OIT 2022 (estimations régionales ASS).",
    transform=ax.transAxes, fontsize=8, style="italic", color="#666"
)

plt.tight_layout()
plt.savefig(FIGURES / "04_expected_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure sauvegardée → 04_expected_vs_actual.png")


# %% CELLULE 7 — Heatmap : Top 10 pays × secteurs
# ─────────────────────────────────────────────────────────
# Ce graphique répond à : "Y a-t-il des spécialisations
# sectorielles par pays ?"
# Kenya vs Nigeria vs Sénégal — même profil sectoriel ?
# ─────────────────────────────────────────────────────────

top10_pays = (
    loans.groupby("country")["id"]
    .count()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

heatmap_data = (
    loans[loans["country"].isin(top10_pays)]
    .groupby(["country","sector"])["id"]
    .count()
    .unstack(fill_value=0)
)

# Normaliser : % des prêts de ce pays dans ce secteur
heatmap_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 7))
sns.heatmap(
    heatmap_pct.round(1),
    annot=True, fmt=".1f",
    cmap="YlOrRd",
    ax=ax,
    linewidths=0.5,
    cbar_kws={"label": "% des prêts du pays dans ce secteur"}
)
ax.set_title(
    "Spécialisation Sectorielle par Pays (Top 10 pays ASS)\n"
    "Valeurs = % des prêts du pays dans ce secteur",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Secteur")
ax.set_ylabel("Pays")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(FIGURES / "04_heatmap_pays_secteur.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure sauvegardée → 04_heatmap_pays_secteur.png")


# %% CELLULE 8 — Analyse Agriculture (secteur dominant)
agri = loans[loans["sector"].str.lower() == "agriculture"].copy()

print(f"🌾 Focus Agriculture :")
print(f"   Prêts     : {len(agri):,} ({len(agri)/len(loans)*100:.1f}% des prêts ASS)")
print(f"   Volume    : ${agri['loan_amount'].sum()/1e6:.1f}M")
print(f"   Médiane   : ${agri['loan_amount'].median():,.0f}")
print(f"   % Femmes  : {agri['is_female'].mean()*100:.1f}%")
print(f"   IMP moyen : {agri['MPI_final'].mean():.3f}")

print(f"\n   Top 8 activités agricoles :")
agri_act = agri["activity"].value_counts().head(8)
for act, count in agri_act.items():
    print(f"   {act:<35} {count:>6,} ({count/len(agri)*100:.1f}%)")

# Saisonnalité : distribution par mois
print(f"\n📅 Distribution mensuelle des prêts agricoles :")
monthly_agri = agri["posted_month"].value_counts().sort_index()
mois = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
for m, count in monthly_agri.items():
    if pd.notna(m):
        bar = "█" * int(count / monthly_agri.max() * 25)
        pct = count / len(agri) * 100
        print(f"   {mois[int(m)-1]} {bar:<27} {pct:.1f}%")


# %% CELLULE 9 — Test de saisonnalité (Chi²)
# ─────────────────────────────────────────────────────────
# H0 : Les prêts agricoles sont uniformément distribués
#      sur les 12 mois (distribution uniforme)
# H1 : La distribution est non-uniforme (saisonnalité)
#
# Si H0 rejetée → signal saisonnier réel
# Si H0 non rejetée → pas de saisonnalité détectable
# ─────────────────────────────────────────────────────────
monthly_counts = agri["posted_month"].value_counts().sort_index().dropna()
expected_uniform = [len(agri) / 12] * len(monthly_counts)

chi2_stat, p_chi2 = stats.chisquare(monthly_counts.values, f_exp=expected_uniform)

print(f"📊 TEST DE SAISONNALITÉ AGRICOLE (χ²) :")
print(f"   H0 : distribution mensuelle uniforme")
print(f"   χ² = {chi2_stat:.2f}, p = {p_chi2:.6f}")
print(f"   Conclusion : {'H0 REJETÉE → saisonnalité significative ✅' if p_chi2 < 0.05 else 'H0 non rejetée → pas de saisonnalité détectée'}")

if p_chi2 < 0.05:
    pic_mois   = mois[int(monthly_counts.idxmax()) - 1]
    creux_mois = mois[int(monthly_counts.idxmin()) - 1]
    print(f"\n   Pic     : {pic_mois} ({monthly_counts.max():,} prêts)")
    print(f"   Creux   : {creux_mois} ({monthly_counts.min():,} prêts)")
    print(f"   Ratio pic/creux : {monthly_counts.max()/monthly_counts.min():.2f}x")


# %% CELLULE 10 — Synthèse sectorielle
print(f"""
{'='*60}
 SYNTHÈSE — NOTEBOOK 04
{'='*60}

 FINDING #3 — Adéquation sectorielle :
""")

# Calculer les principaux écarts
over_financed  = comparison[comparison["gap"] >  5].sort_values("gap", ascending=False)
under_financed = comparison[comparison["gap"] < -5].sort_values("gap")

print("   Secteurs SUR-financés (Kiva >> Emploi) :")
for _, row in over_financed.iterrows():
    print(f"     {row['sector']:<20} Kiva={row['pct_loans']:.1f}% vs Emploi={row['expected']:.1f}% ({row['gap_label']})")

print("\n   Secteurs SOUS-financés (Kiva << Emploi) :")
for _, row in under_financed.iterrows():
    print(f"     {row['sector']:<20} Kiva={row['pct_loans']:.1f}% vs Emploi={row['expected']:.1f}% ({row['gap_label']})")

print(f"""
 Figures produites :
   → 04_sector_distribution.png
   → 04_expected_vs_actual.png
   → 04_heatmap_pays_secteur.png

""")
# %%
