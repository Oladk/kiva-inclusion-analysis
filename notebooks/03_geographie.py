# ============================================================
# NOTEBOOK 03 — Analyse Géographique
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
loans = pd.read_parquet(DATA_PROC / "loans_ssa.parquet")
print(f"✅ {len(loans):,} prêts ASS chargés")
print(f"   Pays    : {loans['country'].nunique()}")
print(f"   Secteurs: {loans['sector'].nunique()}")
print(f"   Période : {loans['posted_year'].min():.0f} – {loans['posted_year'].max():.0f}")


# %% CELLULE 3 — Volume et valeur par pays
# ─────────────────────────────────────────────────────────
# On construit la table principale de l'analyse géographique.
# Chaque métrique a une interprétation spécifique :
#
# n_loans       → présence absolue (volume)
# total_volume  → poids financier
# median_amount → taille typique des prêts dans ce pays
# pct_female    → inclusivité genre
# n_partners    → profondeur de la présence institutionnelle
# ─────────────────────────────────────────────────────────
country_stats = (
    loans
    .groupby(["country", "country_code", "sub_region"])
    .agg(
        n_loans       = ("id",             "count"),
        total_volume  = ("loan_amount",    "sum"),
        median_amount = ("loan_amount",    "median"),
        pct_female    = ("is_female",      "mean"),
        n_partners    = ("partner_id",     "nunique"),
    )
    .sort_values("n_loans", ascending=False)
    .reset_index()
)
country_stats["pct_of_ssa"]  = country_stats["n_loans"]  / len(loans) * 100
country_stats["pct_volume"]  = country_stats["total_volume"] / loans["loan_amount"].sum() * 100
country_stats["pct_female"]  = country_stats["pct_female"] * 100

print("📊 Top 15 pays ASS par nombre de prêts :\n")
cols_display = ["country","sub_region","n_loans","pct_of_ssa","total_volume","median_amount","pct_female","n_partners"]
print(country_stats[cols_display].head(15).round(2).to_string(index=False))


# %% CELLULE 4 — Concentration géographique (Indice de Gini)
# ─────────────────────────────────────────────────────────
# POURQUOI calculer le Gini géographique ?
# Le Gini mesure l'inégalité de distribution.
# Ici : est-ce que les prêts sont équitablement répartis
# entre les 28 pays, ou concentrés sur 3-4 pays ?
#
# Gini = 0  → distribution parfaitement égale
# Gini = 1  → tout concentré sur un seul pays
#
# Référence : Gini des revenus en ASS ≈ 0.43-0.50
# Si notre Gini géographique > 0.50 → plus inégal que les revenus
# ─────────────────────────────────────────────────────────

def gini(array):
    array = np.sort(np.array(array, dtype=float))
    n     = len(array)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * array) - (n + 1) * np.sum(array)) / (n * np.sum(array))

gini_loans  = gini(country_stats["n_loans"].values)
gini_volume = gini(country_stats["total_volume"].values)

# Concentration : combien de pays pour 80% des prêts ?
country_stats_sorted = country_stats.sort_values("n_loans", ascending=False)
country_stats_sorted["cumul_pct"] = country_stats_sorted["pct_of_ssa"].cumsum()
n_for_80pct = (country_stats_sorted["cumul_pct"] <= 80).sum() + 1

print(f"📊 CONCENTRATION GÉOGRAPHIQUE :")
print(f"   Gini (nombre de prêts) : {gini_loans:.3f}")
print(f"   Gini (volume $)        : {gini_volume:.3f}")
print(f"   Pays pour atteindre 80%: {n_for_80pct} pays sur {len(country_stats)}")
print(f"\n   → Les {n_for_80pct} premiers pays représentent 80% des prêts ASS :")
print(country_stats_sorted[["country","pct_of_ssa","cumul_pct"]].head(n_for_80pct).round(2).to_string(index=False))


# %% CELLULE 5 — Visualisation : Top pays + sous-régions
COLORS = {
    "Afrique de l'Ouest" : "#1A6B5C",
    "Afrique de l'Est"   : "#E8840B",
    "Afrique Centrale"   : "#2E86AB",
    "Afrique Australe"   : "#C73E1D",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Distribution Géographique des Prêts Kiva — Afrique Subsaharienne",
             fontsize=13, fontweight="bold")

# ── Graphique gauche : Top 15 pays ──────────────────────
top15 = country_stats.head(15)
bar_colors = [COLORS.get(r, "#999") for r in top15["sub_region"]]

bars = axes[0].barh(
    top15["country"][::-1],
    top15["pct_of_ssa"][::-1],
    color=bar_colors[::-1],
    alpha=0.85
)
for bar, val in zip(bars, top15["pct_of_ssa"][::-1]):
    axes[0].text(val + 0.1, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)

axes[0].set_xlabel("% des prêts ASS")
axes[0].set_title("Top 15 pays par volume de prêts")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

legend_patches = [mpatches.Patch(color=c, label=r) for r, c in COLORS.items()]
axes[0].legend(handles=legend_patches, fontsize=8, loc="lower right")

# ── Graphique droit : Comparaison sous-régions ──────────
subregion_stats = (
    loans
    .groupby("sub_region")
    .agg(
        n_loans      = ("id",          "count"),
        total_volume = ("loan_amount", "sum"),
        pct_female   = ("is_female",   "mean"),
        n_countries  = ("country",     "nunique"),
    )
    .reset_index()
)
subregion_stats["pct_of_ssa"] = subregion_stats["n_loans"] / len(loans) * 100
subregion_stats["pct_female"] *= 100
subregion_stats = subregion_stats.sort_values("n_loans", ascending=False)

sr_colors = [COLORS.get(r, "#999") for r in subregion_stats["sub_region"]]
bars2 = axes[1].bar(
    subregion_stats["sub_region"],
    subregion_stats["pct_of_ssa"],
    color=sr_colors,
    alpha=0.85,
    width=0.6
)
for bar, val in zip(bars2, subregion_stats["pct_of_ssa"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

axes[1].set_ylabel("% des prêts ASS")
axes[1].set_title("Poids de chaque sous-région")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=15, ha="right")

plt.tight_layout()
plt.savefig(FIGURES / "03_geographic_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print("✅ Figure sauvegardée → reports/figures/03_geographic_distribution.png")


# %% CELLULE 6 — Normalisation par population adulte
# ─────────────────────────────────────────────────────────
# POURQUOI normaliser ?
# Le Kenya a plus de prêts que le Bénin. Mais le Kenya a aussi
# 5x plus d'habitants. Sans normalisation, on mesure la taille
# du pays, pas la pénétration réelle du microcrédit.
#
# Métrique : prêts pour 1000 adultes
# Source population : ONU 2020 (approximations)
# ─────────────────────────────────────────────────────────

POP_ADULTES = {
    "Kenya":27.0,"Uganda":18.0,"Tanzania":25.0,"Rwanda":7.3,
    "Ethiopia":62.0,"Mozambique":16.0,"Nigeria":108.0,"Ghana":18.0,
    "Senegal":9.0,"Mali":9.0,"Burkina Faso":9.0,"Togo":4.2,
    "Benin":5.5,"Niger":9.0,"Sierra Leone":3.5,"Liberia":2.5,
    "Guinea":7.0,"Cameroon":13.0,"Madagascar":14.0,"Malawi":10.0,
    "Zambia":10.0,"Zimbabwe":10.0,"South Africa":39.0,"Lesotho":1.5,
    "Democratic Republic Of The Congo":45.0,"Congo":2.5,
    "Tajikistan":5.0,"Pakistan":120.0
}

country_stats["pop_adultes_M"]    = country_stats["country"].map(POP_ADULTES)
country_stats["prêts_p1000_adultes"] = (
    country_stats["n_loans"] / (country_stats["pop_adultes_M"] * 1000)
).round(2)

penetration = (
    country_stats[country_stats["pop_adultes_M"].notna()]
    .sort_values("prêts_p1000_adultes", ascending=False)
    [["country","sub_region","n_loans","prêts_p1000_adultes","pct_female"]]
)

print("📊 Pénétration Kiva (prêts pour 1 000 adultes) :\n")
print(penetration.head(15).round(2).to_string(index=False))
print(f"\n⚠️  Population source : ONU 2020 — chiffres approximatifs")
print(f"   Ces taux sont indicatifs, pas des mesures précises")


# %% CELLULE 7 — Visualisation : Pénétration vs % Femmes 
fig, ax = plt.subplots(figsize=(12, 7))

# Fix : on merge uniquement total_volume depuis country_stats
# sub_region est déjà dans penetration — pas besoin de le re-merger
plot_data = penetration.merge(
    country_stats[["country", "total_volume"]],   # ← sub_region retiré ici
    on="country",
    how="left"
).dropna(subset=["prêts_p1000_adultes"])

for _, row in plot_data.iterrows():
    color = COLORS.get(row["sub_region"], "#999")
    ax.scatter(
        row["pct_female"],
        row["prêts_p1000_adultes"],
        s=max(row["total_volume"] / plot_data["total_volume"].max() * 800, 50),
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidth=0.8
    )
    ax.annotate(
        row["country"][:4].upper(),
        (row["pct_female"], row["prêts_p1000_adultes"]),
        fontsize=7.5, alpha=0.85,
        xytext=(4, 3), textcoords="offset points"
    )

ax.axvline(x=50, color="red", linestyle="--", alpha=0.5, linewidth=1, label="50% femmes")
ax.axhline(
    y=plot_data["prêts_p1000_adultes"].median(),
    color="gray", linestyle="--", alpha=0.5, linewidth=1,
    label=f"Médiane pénétration ({plot_data['prêts_p1000_adultes'].median():.1f})"
)

ax.set_xlabel("% Emprunteuses féminines", fontsize=11)
ax.set_ylabel("Prêts pour 1 000 adultes", fontsize=11)
ax.set_title(
    "Pénétration vs Inclusivité Genre par Pays ASS\n"
    "(Taille des points = volume total en $)",
    fontsize=12, fontweight="bold"
)

legend_patches = [mpatches.Patch(color=c, label=r) for r, c in COLORS.items()]
ax.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color="red",  linestyle="--", label="50% femmes"),
    plt.Line2D([0],[0], color="gray", linestyle="--", label="Médiane pénétration"),
], fontsize=8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.01, -0.1,
    "⚠️  Population ONU 2020 (approximations). Taux indicatifs.",
    transform=ax.transAxes, fontsize=8, style="italic", color="#666"
)

plt.tight_layout()
plt.savefig(FIGURES / "03_penetration_vs_genre.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure sauvegardée")


# %% CELLULE 8 — Intégration données MPI
# ─────────────────────────────────────────────────────────
# L'IMP (Indice de Pauvreté Multidimensionnelle) mesure
# la pauvreté au-delà du revenu : santé, éducation, niveau de vie.
#
# Question analytique : Kiva finance-t-il les régions les plus pauvres ?
# Si non → le capital de microfinancement va aux moins défavorisés,
# ce qui contredit la mission d'inclusion.
#
# STRATÉGIE DE JOINTURE :
# On joint loans_ssa sur (country + region) avec le dataset MPI.
# Pour les régions sans match exact → médiane du pays.
# Biais documenté : atténuation des disparités intra-pays.
# ─────────────────────────────────────────────────────────

mpi = pd.read_csv(DATA_PROC.parent / "raw" / "kiva_mpi_region_locations.csv")
print(f"📂 MPI dataset : {mpi.shape}")
print(mpi.head(3).to_string())


# %% CELLULE 9 — Nettoyage des clés de jointure MPI
mpi["country_j"] = mpi["country"].str.strip().str.title()
mpi["region_j"]  = mpi["region"].str.strip().str.title()
loans["country_j"] = loans["country"].str.strip().str.title()
loans["region_j"]  = loans["region"].str.strip().str.title()

# Jointure sur pays + région
loans_mpi = loans.merge(
    mpi[["country_j","region_j","MPI","lat","lon"]],
    on=["country_j","region_j"],
    how="left"
)

# Compléter les NaN par la médiane du pays
country_mpi_med = (
    mpi.groupby("country_j")["MPI"]
    .median()
    .rename("MPI_country_med")
)
loans_mpi = loans_mpi.merge(country_mpi_med, on="country_j", how="left")
loans_mpi["MPI_final"]  = loans_mpi["MPI"].fillna(loans_mpi["MPI_country_med"])
loans_mpi["mpi_source"] = np.where(loans_mpi["MPI"].notna(), "region", "country_median")

match_exact = (loans_mpi["mpi_source"] == "region").mean() * 100
match_total = loans_mpi["MPI_final"].notna().mean() * 100

print(f"\n✅ Jointure MPI terminée :")
print(f"   Match exact (région)    : {match_exact:.1f}%")
print(f"   Match total (+ médiane) : {match_total:.1f}%")
print(f"   Sans MPI                : {100-match_total:.1f}%")
print(f"\n⚠️  {100-match_exact:.1f}% utilisent la médiane pays → disparités intra-pays atténuées")


# %% CELLULE 10 — Corrélation MPI ↔ Volume de financement
# ─────────────────────────────────────────────────────────
# C'est LA question centrale de ce notebook :
# Le financement Kiva atteint-il les régions les plus pauvres ?
#
# On utilise Spearman (non-paramétrique) car :
#   1. La distribution de n_loans par pays est très asymétrique
#   2. On cherche une relation monotone, pas linéaire
#   3. Moins sensible aux outliers (Kenya, Nigeria)
# ─────────────────────────────────────────────────────────

country_mpi_agg = (
    loans_mpi[loans_mpi["MPI_final"].notna()]
    .groupby("country")
    .agg(
        n_loans      = ("id",          "count"),
        total_volume = ("loan_amount", "sum"),
        avg_mpi      = ("MPI_final",   "mean"),
        sub_region   = ("sub_region",  "first"),
    )
    .reset_index()
)

rho, p_val = stats.spearmanr(
    country_mpi_agg["avg_mpi"],
    country_mpi_agg["n_loans"]
)

print(f"📊 TEST : Corrélation IMP ↔ Volume de prêts")
print(f"   H0 : aucune relation monotone entre pauvreté et financement")
print(f"\n   Corrélation Spearman : ρ = {rho:.3f}")
print(f"   p-value              : {p_val:.4f}")
print(f"   Significatif (α=0.05): {'OUI ✅' if p_val < 0.05 else 'NON ❌'}")

if rho > 0.1 and p_val < 0.05:
    interpretation = "POSITIVE et significative → + un pays est pauvre, + il est financé ✅"
elif rho < -0.1 and p_val < 0.05:
    interpretation = "NÉGATIVE et significative → + un pays est pauvre, - il est financé ⚠️"
else:
    interpretation = "Pas de relation claire → le financement n'est pas corrélé à la pauvreté"

print(f"\n   → Interprétation : {interpretation}")


# %% CELLULE 11 — Visualisation : MPI vs Financement
fig, ax = plt.subplots(figsize=(11, 7))

for _, row in country_mpi_agg.iterrows():
    color = COLORS.get(row["sub_region"], "#999")
    ax.scatter(
        row["avg_mpi"],
        row["n_loans"],
        s=max(row["total_volume"] / country_mpi_agg["total_volume"].max() * 600, 40),
        color=color, alpha=0.75,
        edgecolors="white", linewidth=0.8
    )
    ax.annotate(
        row["country"][:3].upper(),
        (row["avg_mpi"], row["n_loans"]),
        fontsize=7.5, alpha=0.85,
        xytext=(4, 3), textcoords="offset points"
    )

# Ligne de tendance
x = country_mpi_agg["avg_mpi"].values
y = country_mpi_agg["n_loans"].values
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, p(x_line), "r--", alpha=0.6, linewidth=1.5,
        label=f"Tendance (ρ={rho:.2f}, p={p_val:.3f})")

ax.set_xlabel("IMP moyen (0 = moins pauvre → 1 = plus pauvre)", fontsize=11)
ax.set_ylabel("Nombre de prêts Kiva", fontsize=11)
ax.set_title(
    "Pauvreté Multidimensionnelle vs Volume de Financement Kiva\n"
    "Afrique Subsaharienne — par pays (taille = volume $)",
    fontsize=12, fontweight="bold"
)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

legend_patches = [mpatches.Patch(color=c, label=r) for r, c in COLORS.items()]
ax.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color="red", linestyle="--", label=f"Tendance ρ={rho:.2f}")
], fontsize=8)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(0.01, -0.1,
    "⚠️  IMP source : Kiva MPI dataset. Jointure par région (médiane pays si non-matchée).",
    transform=ax.transAxes, fontsize=8, style="italic", color="#666"
)

plt.tight_layout()
plt.savefig(FIGURES / "03_mpi_vs_financement.png", dpi=150, bbox_inches="tight")
plt.show()


# %% CELLULE 12 — Export final
loans_mpi.to_parquet(DATA_PROC / "loans_ssa_mpi.parquet", index=False)

print(f"✅ loans_ssa_mpi.parquet sauvegardé")
print(f"   {len(loans_mpi):,} prêts ASS avec MPI")
print(f"""
{'='*55}
 NOTEBOOK 03 TERMINÉ
{'='*55}

 Figures produites :
   → 03_geographic_distribution.png
   → 03_penetration_vs_genre.png
   → 03_mpi_vs_financement.png

 Fichier produit :
   → data/processed/loans_ssa_mpi.parquet

""")
# %%
