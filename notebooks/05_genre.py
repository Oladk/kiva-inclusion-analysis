# ============================================================
# NOTEBOOK 05 : Analyse du Genre
# Projet : Analyse Inclusion Financière ASS
# Auteur : Ronald Dossou-Kohi
# ============================================================

# %% CELLULE 1 : Imports
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

PALETTE = {
    "female"  : "#E84393",
    "male"    : "#3498DB",
    "mixed"   : "#F39C12",
    "unknown" : "#95A5A6",
}

print(" Imports OK")


# %% CELLULE 2 : Chargement
loans = pd.read_parquet(DATA_PROC / "loans_ssa_mpi.parquet")
print(f" {len(loans):,} prêts ASS chargés")

# Distribution genre
print(f"\n Distribution gender_clean :")
dist = loans["gender_clean"].value_counts()
for cat, count in dist.items():
    pct = count / len(loans) * 100
    print(f"   {cat:<12} {count:>7,}  ({pct:.1f}%)")


# %% CELLULE 3 : Filtrage pour l'analyse de genre
# ─────────────────────────────────────────────────────────
# On garde uniquement female et male pour les comparaisons.
# Les "mixed" (groupes mixtes) et "unknown" sont exclus car :
#   mixed   → impossible d'attribuer le prêt à un genre
#   unknown → information absente
#
# BIAIS DOCUMENTÉ : Si les prêts "unknown" ont des caractéristiques
# systématiques (ex. concentrés dans certains pays ou secteurs),
# leur exclusion introduit un biais de sélection.
# On vérifie ça avant d'exclure.
# ─────────────────────────────────────────────────────────

# Vérification : les "unknown" sont-ils systématiques ?
print(" Profil des prêts 'unknown' vs reste :\n")
unknown_profile = pd.DataFrame({
    "unknown" : loans[loans["gender_clean"] == "unknown"][
        ["loan_amount","is_fully_funded"]
    ].mean(),
    "connu"   : loans[loans["gender_clean"] != "unknown"][
        ["loan_amount","is_fully_funded"]
    ].mean(),
})
print(unknown_profile.round(2).to_string())
print("\n   → Si les valeurs sont proches : exclusion peu biaisante")
print("   → Si elles divergent : mentionner le biais dans le rapport")

# Filtrage
loans_g = loans[loans["gender_clean"].isin(["female","male"])].copy()
pct_conserve = len(loans_g) / len(loans) * 100
print(f"\n Dataset genre : {len(loans_g):,} prêts ({pct_conserve:.1f}% du total ASS)")
print(f"   Exclus : {len(loans) - len(loans_g):,} prêts (mixed + unknown)")


# %% CELLULE 4 : Représentation : Nombre vs Volume
# ─────────────────────────────────────────────────────────
# C'est la distinction analytique centrale de ce notebook.
#
# Un biais courant : voir 72% de femmes et conclure que
# les femmes sont bien servies. Mais si elles reçoivent
# des prêts 40% plus petits, leur part du VOLUME est
# bien inférieure à leur part en NOMBRE.
#
# → Part en NOMBRE = représentation
# → Part en VOLUME = allocation réelle du capital
# ─────────────────────────────────────────────────────────

gender_overview = (
    loans_g
    .groupby("gender_clean")
    .agg(
        n_loans        = ("id",             "count"),
        total_volume   = ("loan_amount",    "sum"),
        median_amount  = ("loan_amount",    "median"),
        mean_amount    = ("loan_amount",    "mean"),
        median_term    = ("term_in_months", "median"),
        avg_mpi        = ("MPI_final",      "mean"),
    )
)
gender_overview["pct_loans"]  = gender_overview["n_loans"] / len(loans_g) * 100
gender_overview["pct_volume"] = gender_overview["total_volume"] / loans_g["loan_amount"].sum() * 100
gender_overview["ratio_vol_count"] = gender_overview["pct_volume"] / gender_overview["pct_loans"]

print(" Vue d'ensemble : Genre :\n")
print(gender_overview.round(3).to_string())

# Calcul de l'écart
f_count  = gender_overview.loc["female","pct_loans"]
f_volume = gender_overview.loc["female","pct_volume"]
ecart    = f_count - f_volume

print(f"\n INSIGHT CLÉ :")
print(f"   Femmes : {f_count:.1f}% des prêts en NOMBRE")
print(f"   Femmes : {f_volume:.1f}% du VOLUME total")
print(f"   Écart  : {ecart:.1f} points de pourcentage")
print(f"   → Les femmes sont sur-représentées en nombre mais")
print(f"     sous-représentées en volume de capital alloué")


# %% CELLULE 5 : Visualisation : Nombre vs Volume
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Genre : Représentation en Nombre vs en Volume de Capital\n"
             "Afrique Subsaharienne — Kiva",
             fontsize=13, fontweight="bold")

genders = gender_overview.index.tolist()
colors  = [PALETTE[g] for g in genders]

# ── Donut 1 : Part en nombre ──────────────────────────────
wedges1, _ = axes[0].pie(
    gender_overview["pct_loans"],
    colors=colors,
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2)
)
axes[0].set_title("Part en NOMBRE\nde prêts", fontsize=11, fontweight="bold")
for i, (g, pct) in enumerate(zip(genders, gender_overview["pct_loans"])):
    axes[0].text(0, 0.1 - i*0.25,
                 f"{g.title()}\n{pct:.1f}%",
                 ha="center", fontsize=10, color=colors[i], fontweight="bold")

# ── Donut 2 : Part en volume ──────────────────────────────
axes[1].pie(
    gender_overview["pct_volume"],
    colors=colors,
    startangle=90,
    wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2)
)
axes[1].set_title("Part en VOLUME ($)\nalloué", fontsize=11, fontweight="bold")
for i, (g, pct) in enumerate(zip(genders, gender_overview["pct_volume"])):
    axes[1].text(0, 0.1 - i*0.25,
                 f"{g.title()}\n{pct:.1f}%",
                 ha="center", fontsize=10, color=colors[i], fontweight="bold")

# ── Graphique 3 : Montant médian ──────────────────────────
medians      = gender_overview["median_amount"]
bar_colors   = [PALETTE[g] for g in medians.index]
bars         = axes[2].bar(
    [g.title() for g in medians.index],
    medians.values,
    color=bar_colors, alpha=0.85, width=0.5
)
for bar, val in zip(bars, medians.values):
    axes[2].text(
        bar.get_x() + bar.get_width()/2,
        val + 5, f"${val:,.0f}",
        ha="center", fontsize=11, fontweight="bold"
    )
axes[2].set_ylabel("Montant médian ($)")
axes[2].set_title("Montant MÉDIAN\npar genre", fontsize=11, fontweight="bold")
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)

# Note méthodologique
fig.text(0.5, -0.03,
    " L'écart de montant peut refléter les pratiques des IMF (secteurs ciblés, "
    "garanties exigées) plutôt qu'une discrimination directe.",
    ha="center", fontsize=8.5, style="italic", color="#555"
)

plt.tight_layout()
plt.savefig(FIGURES / "05_genre_nombre_vs_volume.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Figure sauvegardée → 05_genre_nombre_vs_volume.png")


# %% CELLULE 6 : Test statistique : Mann-Whitney U
# ─────────────────────────────────────────────────────────
# POURQUOI Mann-Whitney et pas un t-test ?
#
# Le t-test suppose que les distributions comparées sont
# approximativement normales. Or notre ratio moy/méd = 1.68
# indique une forte asymétrie → t-test non approprié.
#
# Mann-Whitney U teste si une valeur tirée aléatoirement
# dans le groupe F est systématiquement plus petite/grande
# qu'une valeur tirée dans le groupe M.
# C'est un test sur les rangs, robuste à l'asymétrie.
# ─────────────────────────────────────────────────────────

female_amounts = loans_g[loans_g["gender_clean"] == "female"]["loan_amount"]
male_amounts   = loans_g[loans_g["gender_clean"] == "male"]["loan_amount"]

stat, p_mw = stats.mannwhitneyu(
    female_amounts, male_amounts,
    alternative="two-sided"
)

# Effect size : r = Z / sqrt(N)
n_total = len(female_amounts) + len(male_amounts)
z_score = stats.norm.ppf(p_mw / 2)
effect_r = abs(z_score) / np.sqrt(n_total)

print(f" TEST DE MANN-WHITNEY U : Montants F vs M")
print(f"   H0 : Les distributions de montants sont identiques")
print(f"\n   Médiane Femmes : ${female_amounts.median():,.0f}")
print(f"   Médiane Hommes : ${male_amounts.median():,.0f}")
print(f"   Ratio H/F      : {male_amounts.median()/female_amounts.median():.2f}x")
print(f"\n   U = {stat:.0f}")
print(f"   p-value = {p_mw:.6f}")
print(f"   Effect size r = {effect_r:.4f}")

sig = "SIGNIFICATIF" if p_mw < 0.05 else "Non significatif"
eff = "négligeable (<0.1)" if effect_r < 0.1 else \
      "petit (0.1-0.3)"    if effect_r < 0.3 else \
      "modéré (0.3-0.5)"   if effect_r < 0.5 else "large (>0.5)"

print(f"\n   → Test {sig} | Effect size : {eff}")
print(f"\n ATTENTION : Significatif ≠ important.")
print(f"   Avec {n_total:,} observations, même un écart minuscule")
print(f"   peut être statistiquement significatif.")
print(f"   C'est pourquoi l'effect size est indispensable.")


# %% CELLULE 7 — Contrôle par secteur
# ─────────────────────────────────────────────────────────
# QUESTION CRITIQUE : L'écart de montant entre genres
# est-il dû au genre lui-même, ou à la composition
# sectorielle différente ?
#
# Exemple : Si les femmes font surtout du commerce
# (petits montants) et les hommes surtout de l'agriculture
# (grands équipements), l'écart de montant s'explique
# par le secteur ; pas par le genre.
#
# On calcule le ratio H/F DANS CHAQUE SECTEUR.
# Si le ratio est similaire partout → gap réel indépendant du secteur
# Si le ratio varie → la composition sectorielle explique une partie
# ─────────────────────────────────────────────────────────

sector_gender = (
    loans_g
    .groupby(["sector","gender_clean"])["loan_amount"]
    .median()
    .unstack()
    .dropna()
)
sector_gender.columns.name = None
sector_gender["ratio_H_F"] = sector_gender["male"] / sector_gender["female"]
sector_gender["gap_abs"]   = sector_gender["male"] - sector_gender["female"]
sector_gender = sector_gender.sort_values("ratio_H_F", ascending=False)

print(" CONTRÔLE SECTORIEL : Ratio Médiane Hommes/Femmes :\n")
print(f"{'Secteur':<20} {'Médiane F':>10} {'Médiane M':>10} {'Ratio H/F':>10}  Signal")
print("-" * 65)
for sector, row in sector_gender.iterrows():
    signal = " Écart important" if row["ratio_H_F"] > 1.5 else \
             " Proche de 1"      if row["ratio_H_F"] < 1.1 else "~"
    print(f"  {sector:<20} ${row['female']:>8,.0f}  ${row['male']:>8,.0f}  "
          f"{row['ratio_H_F']:>9.2f}x  {signal}")

avg_ratio = sector_gender["ratio_H_F"].mean()
print(f"\n   Ratio moyen toutes secteurs : {avg_ratio:.2f}x")
print(f"   → Si ratio > 1 dans TOUS les secteurs : gap réel, pas seulement")
print(f"     expliqué par la composition sectorielle")


# %% CELLULE 8 : Visualisation : Gap de genre par secteur
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Gap de Genre par Secteur — Kiva Afrique Subsaharienne",
             fontsize=13, fontweight="bold")

# ── Graphique gauche : montants médians F vs M par secteur ─
x      = np.arange(len(sector_gender))
width  = 0.38

axes[0].barh(
    [s + " (F)" for s in sector_gender.index[::-1]],
    sector_gender["female"][::-1],
    height=width, color=PALETTE["female"], alpha=0.85, label="Femmes"
)
axes[0].barh(
    [s + " (M)" for s in sector_gender.index[::-1]],
    sector_gender["male"][::-1],
    height=width, color=PALETTE["male"], alpha=0.85, label="Hommes"
)
axes[0].set_xlabel("Montant médian ($)")
axes[0].set_title("Montant médian par secteur et genre")
axes[0].legend()
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# ── Graphique droit : ratio H/F par secteur ──────────────
ratio_colors = ["#C73E1D" if r > 1.5 else "#1A6B5C" if r < 1.1 else "#F7B731"
                for r in sector_gender["ratio_H_F"]]

bars = axes[1].barh(
    sector_gender.index[::-1],
    sector_gender["ratio_H_F"][::-1],
    color=ratio_colors[::-1], alpha=0.85
)
for bar, val in zip(bars, sector_gender["ratio_H_F"][::-1]):
    axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}x", va="center", fontsize=9)

axes[1].axvline(x=1.0, color="black", linewidth=1.2,
                linestyle="--", label="Parité (ratio = 1)")
axes[1].set_xlabel("Ratio Médiane Homme / Médiane Femme")
axes[1].set_title("Ratio H/F par secteur\n(>1 = hommes reçoivent plus)")
axes[1].legend(fontsize=9)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

# Légende couleurs
legend_patches = [
    mpatches.Patch(color="#C73E1D", label="Écart important (>1.5x)"),
    mpatches.Patch(color="#F7B731", label="Écart modéré (1.1-1.5x)"),
    mpatches.Patch(color="#1A6B5C", label="Proche parité (<1.1x)"),
]
axes[1].legend(handles=legend_patches, fontsize=8, loc="lower right")

fig.text(0.5, -0.03,
    " Ces écarts peuvent refléter les pratiques de sélection des IMF "
    "(montants accordés) plutôt qu'une discrimination directe.",
    ha="center", fontsize=8.5, style="italic", color="#555"
)

plt.tight_layout()
plt.savefig(FIGURES / "05_genre_par_secteur.png", dpi=150, bbox_inches="tight")
plt.show()
print(" Figure sauvegardée → 05_genre_par_secteur.png")


# %% CELLULE 9 : Gap de genre par pays
pays_genre = (
    loans_g
    .groupby(["country","gender_clean"])["loan_amount"]
    .agg(["median","count"])
    .unstack()
    .dropna()
)
pays_genre.columns = [f"{col[0]}_{col[1]}" for col in pays_genre.columns]
pays_genre["ratio_H_F"]  = pays_genre["median_male"] / pays_genre["median_female"]
pays_genre["pct_female"] = (
    loans_g[loans_g["gender_clean"] == "female"]
    .groupby("country")["id"].count()
    / loans_g.groupby("country")["id"].count()
    * 100
)

# Filtrer pays avec au moins 200 prêts F et 200 prêts M
pays_genre_f = pays_genre[
    (pays_genre["count_female"] >= 200) &
    (pays_genre["count_male"]   >= 200)
].dropna()

print(f" Gap de genre par pays ({len(pays_genre_f)} pays avec données suffisantes) :\n")
print(f"{'Pays':<25} {'Méd. F':>8} {'Méd. M':>8} {'Ratio H/F':>10} {'% Femmes':>10}")
print("-" * 65)
for country, row in pays_genre_f.sort_values("ratio_H_F", ascending=False).iterrows():
    print(f"  {country:<23} ${row['median_female']:>6,.0f}  ${row['median_male']:>6,.0f}"
          f"  {row['ratio_H_F']:>9.2f}x  {row['pct_female']:>8.1f}%")


# %% CELLULE 10 : Délai de financement par genre
# ─────────────────────────────────────────────────────────
# Hypothèse : les prêts féminins se financent plus vite
# car les prêteurs Kiva les perçoivent comme moins risqués
# et plus "narrativement attractifs".
#
# Si vrai → biais de plateforme en faveur des femmes
# (elles sont avantagées sur Kiva mais pas nécessairement
#  dans le secteur financier en général)
# ─────────────────────────────────────────────────────────

# Filtrer les valeurs aberrantes de days_to_fund
dtf = loans_g[
    (loans_g["days_to_fund"].notna()) &
    (loans_g["days_to_fund"] >= 0) &
    (loans_g["days_to_fund"] <= 90)
].copy()

delai_gender = dtf.groupby("gender_clean")["days_to_fund"].agg(["median","mean","count"])
print(f" DÉLAI DE FINANCEMENT par genre (prêts 0-90 jours) :\n")
print(delai_gender.round(2).to_string())

# Test Mann-Whitney sur les délais
f_delai = dtf[dtf["gender_clean"] == "female"]["days_to_fund"]
m_delai = dtf[dtf["gender_clean"] == "male"]["days_to_fund"]

stat_d, p_d = stats.mannwhitneyu(f_delai, m_delai, alternative="two-sided")
print(f"\n   Mann-Whitney : délai de financement :")
print(f"   p-value = {p_d:.6f}")
print(f"   → {'Différence significative ' if p_d < 0.05 else 'Pas de différence significative'}")

if p_d < 0.05:
    faster = "Femmes" if f_delai.median() < m_delai.median() else "Hommes"
    print(f"   → Les {faster} se financent plus vite "
          f"(médiane F={f_delai.median():.1f}j vs M={m_delai.median():.1f}j)")


# %% CELLULE 11 : Synthèse genre
print(f"""
{'='*60}
 SYNTHÈSE : NOTEBOOK 05
{'='*60}

 FINDING #4 : Représentation vs Capital :
   Femmes = {f_count:.1f}% des prêts | {f_volume:.1f}% du volume
   Écart : {ecart:.1f} points de pourcentage

 FINDING #5 : Gap de montant médian :
   Femmes : ${female_amounts.median():,.0f}
   Hommes : ${male_amounts.median():,.0f}
   Ratio  : {male_amounts.median()/female_amounts.median():.2f}x
   Test MW: p = {p_mw:.6f} → {'Significatif' if p_mw < 0.05 else 'Non significatif'}
   Effect size : {eff}

 FINDING #6 : Gap contrôlé par secteur :
   Ratio H/F moyen intra-secteur : {avg_ratio:.2f}x
   → Le gap persiste après contrôle sectoriel

 FINDING #7 : Délai de financement :
   Médiane F : {f_delai.median():.1f} jours
   Médiane M : {m_delai.median():.1f} jours

 Figures produites :
   → 05_genre_nombre_vs_volume.png
   → 05_genre_par_secteur.png

""")
