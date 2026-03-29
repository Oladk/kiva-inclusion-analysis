# ============================================================
# NOTEBOOK 06 — Field Partners & Déterminants du Financement
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
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
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
print(f"\n⚠️  RAPPEL MÉTHODOLOGIQUE CRITIQUE :")
print(f"   'is_fully_funded' ≠ taux de remboursement réel")
print(f"   On mesure si le prêt a atteint son objectif sur Kiva,")
print(f"   pas si l'emprunteur a remboursé l'IMF.")
print(f"\n   Taux de financement complet : {loans['is_fully_funded'].mean()*100:.1f}%")


# %% CELLULE 3 — Analyse des Field Partners
# ─────────────────────────────────────────────────────────
# Les Field Partners sont l'infrastructure invisible de Kiva.
# Leur performance détermine qui reçoit du capital et combien.
#
# On les évalue sur 4 dimensions :
# 1. Volume      → capacité à mobiliser du capital
# 2. Inclusivité → % femmes et IMP de leur clientèle
# 3. Efficacité  → taux de financement complet
# 4. Vitesse     → délai de financement moyen
# ─────────────────────────────────────────────────────────

partner_stats = (
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
    .sort_values("n_loans", ascending=False)
    .reset_index()
)

partner_stats["pct_female"] *= 100
partner_stats["pct_funded"] *= 100

print(f"📊 Field Partners actifs (≥100 prêts) : {len(partner_stats)}")
print(f"\n   Top 15 partenaires par volume :\n")
cols = ["partner_id","n_loans","total_volume","median_amount","pct_female","pct_funded","avg_mpi"]
print(partner_stats[cols].head(15).round(2).to_string(index=False))


# %% CELLULE 4 — Concentration : Gini des Field Partners
from numpy import sort

def gini(array):
    array = sort(np.array(array, dtype=float))
    n     = len(array)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * array) - (n + 1) * np.sum(array)) / (n * np.sum(array))

gini_partners = gini(partner_stats["n_loans"].values)

partner_stats_s  = partner_stats.sort_values("n_loans", ascending=False)
partner_stats_s["cumul_pct"] = (
    partner_stats_s["n_loans"].cumsum() / partner_stats_s["n_loans"].sum() * 100
)
n_for_80 = (partner_stats_s["cumul_pct"] <= 80).sum() + 1

print(f"📊 CONCENTRATION des Field Partners :")
print(f"   Gini (volume prêts) : {gini_partners:.3f}")
print(f"   Partners pour 80%  : {n_for_80} sur {len(partner_stats)}")
print(f"\n   Top {n_for_80} partners (80% du volume) :")
print(partner_stats_s[["partner_id","n_loans","cumul_pct"]].head(n_for_80).round(1).to_string(index=False))


# %% CELLULE 5 — Efficiency Frontier des Field Partners
# ─────────────────────────────────────────────────────────
# On cherche les partenaires qui combinent :
#   - Fort % de femmes (inclusivité genre)
#   - IMP élevé de leur clientèle (atteinte des plus pauvres)
#   - Volume significatif (impact à l'échelle)
#
# Ces partenaires sont dans le quadrant "Champion" :
# haut-droit du graphique (% femmes élevé ET IMP élevé)
#
# C'est le graphique le plus actionnable pour un bailleur
# (AFD, UNCDF) : il dit où allouer les lignes de crédit.
# ─────────────────────────────────────────────────────────

plot_partners = partner_stats[partner_stats["avg_mpi"].notna()].copy()

# Définir les quadrants
mpi_median    = plot_partners["avg_mpi"].median()
female_median = plot_partners["pct_female"].median()

def quadrant(row):
    if row["pct_female"] >= female_median and row["avg_mpi"] >= mpi_median:
        return "Champion\n(inclusif + pro-pauvres)"
    elif row["pct_female"] >= female_median and row["avg_mpi"] < mpi_median:
        return "Pro-genre\n(inclusif, moins pauvres)"
    elif row["pct_female"] < female_median and row["avg_mpi"] >= mpi_median:
        return "Pro-pauvres\n(moins féminin)"
    else:
        return "Standard"

plot_partners["profil"] = plot_partners.apply(quadrant, axis=1)

QUAD_COLORS = {
    "Champion\n(inclusif + pro-pauvres)"  : "#1A6B5C",
    "Pro-genre\n(inclusif, moins pauvres)": "#E8840B",
    "Pro-pauvres\n(moins féminin)"        : "#2E86AB",
    "Standard"                             : "#95A5A6",
}

fig, ax = plt.subplots(figsize=(12, 8))

for profil, group in plot_partners.groupby("profil"):
    ax.scatter(
        group["pct_female"],
        group["avg_mpi"],
        s=group["total_volume"] / plot_partners["total_volume"].max() * 600,
        color=QUAD_COLORS.get(profil, "#999"),
        alpha=0.75,
        edgecolors="white",
        linewidth=0.8,
        label=profil
    )

for _, row in plot_partners.nlargest(10, "n_loans").iterrows():
    ax.annotate(
        f"P{int(row['partner_id'])}",
        (row["pct_female"], row["avg_mpi"]),
        fontsize=7.5, alpha=0.85,
        xytext=(4, 3), textcoords="offset points"
    )

ax.axvline(x=female_median, color="gray", linestyle="--", alpha=0.5, linewidth=1)
ax.axhline(y=mpi_median,    color="gray", linestyle="--", alpha=0.5, linewidth=1)

ax.set_xlabel("% Emprunteuses féminines", fontsize=11)
ax.set_ylabel("IMP moyen de la clientèle (0=moins pauvre, 1=plus pauvre)", fontsize=11)
ax.set_title(
    "Efficiency Frontier des Field Partners Kiva — ASS\n"
    "Taille des points = Volume total mobilisé ($)",
    fontsize=12, fontweight="bold"
)
ax.legend(fontsize=8, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Annoter les quadrants
ax.text(female_median + 0.5, plot_partners["avg_mpi"].max() * 0.98,
        "CHAMPIONS ✅", fontsize=8, color="#1A6B5C", fontweight="bold")
ax.text(plot_partners["pct_female"].min(), plot_partners["avg_mpi"].max() * 0.98,
        "Pro-pauvres", fontsize=8, color="#2E86AB")

ax.text(0.5, -0.1,
    "⚠️  IMP = médiane pays pour 90.8% des prêts → disparités intra-pays atténuées.",
    transform=ax.transAxes, fontsize=8, style="italic", color="#666", ha="center"
)

plt.tight_layout()
plt.savefig(FIGURES / "06_efficiency_frontier.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure sauvegardée → 06_efficiency_frontier.png")


# %% CELLULE 6 — Distribution des profils de partners
profil_dist = plot_partners["profil"].value_counts()
print(f"\n📊 Répartition des Field Partners par profil :\n")
for profil, count in profil_dist.items():
    pct = count / len(plot_partners) * 100
    print(f"   {profil.replace(chr(10), ' '):<40} {count:>4} ({pct:.1f}%)")

# Volume mobilisé par profil
print(f"\n📊 Volume mobilisé par profil :")
vol_profil = plot_partners.groupby("profil")["total_volume"].sum().sort_values(ascending=False)
total_vol  = vol_profil.sum()
for profil, vol in vol_profil.items():
    print(f"   {profil.replace(chr(10),' '):<40} ${vol/1e6:>6.1f}M ({vol/total_vol*100:.1f}%)")


# %% CELLULE 7 — Modèle : Déterminants du financement complet
# ─────────────────────────────────────────────────────────
# Variable cible : is_fully_funded (1 = prêt entièrement financé)
#
# Features candidates :
#   log_loan_amount     → taille du prêt (log-transformé)
#   term_in_months      → durée
#   is_female           → genre
#   sector              → secteur (encodé)
#   repayment_interval  → fréquence de remboursement
#   sub_region          → sous-région
#
# Méthode : Régression Logistique + Random Forest
# Validation : Cross-validation 5-fold stratifiée
# ─────────────────────────────────────────────────────────

features = [
    "log_loan_amount",
    "term_in_months",
    "is_female",
    "sector",
    "repayment_interval",
    "sub_region",
]
target = "is_fully_funded"

model_df = loans[features + [target]].copy().dropna()

print(f"📊 Dataset de modélisation :")
print(f"   Observations : {len(model_df):,}")
print(f"   Taux financés: {model_df[target].mean()*100:.1f}%")
print(f"   Features     : {len(features)}")

# Encodage one-hot des catégorielles
cat_cols = ["sector","repayment_interval","sub_region"]
model_enc = pd.get_dummies(model_df, columns=cat_cols, drop_first=True, dtype=float)

X = model_enc.drop(columns=[target])
y = model_enc[target]

print(f"\n   Features après encodage : {X.shape[1]}")


# %% CELLULE 8 — Régression Logistique
print("🔄 Régression Logistique — Cross-validation 5-fold...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42))
])

scores_lr = cross_val_score(pipe_lr, X, y, cv=cv, scoring="roc_auc")

print(f"\n📊 RÉGRESSION LOGISTIQUE :")
print(f"   AUC-ROC (CV=5) : {scores_lr.mean():.3f} ± {scores_lr.std():.3f}")
print(f"   Scores par fold: {[f'{s:.3f}' for s in scores_lr]}")

auc_lr = scores_lr.mean()
interprete = (
    "BON pouvoir discriminant"        if auc_lr >= 0.80 else
    "ACCEPTABLE pouvoir discriminant" if auc_lr >= 0.70 else
    "FAIBLE — features insuffisantes" if auc_lr >= 0.60 else
    "TRÈS FAIBLE — modèle peu informatif"
)
print(f"\n   → {interprete}")


# %% CELLULE 9 — Coefficients de la Régression Logistique
pipe_lr.fit(X, y)
lr_model = pipe_lr.named_steps["model"]

coef_df = pd.DataFrame({
    "feature"    : X.columns,
    "coef"       : lr_model.coef_[0],
    "odds_ratio" : np.exp(lr_model.coef_[0])
}).sort_values("coef", key=abs, ascending=False).head(15)

print(f"\n📊 Top 15 déterminants (Régression Logistique) :\n")
print(f"{'Feature':<40} {'Coef':>8} {'OR':>8}  Interprétation")
print("-" * 70)
for _, row in coef_df.iterrows():
    direction = "↑ favorise" if row["coef"] > 0 else "↓ réduit"
    print(f"  {row['feature']:<40} {row['coef']:>7.3f} {row['odds_ratio']:>7.3f}  {direction}")

print(f"\n💡 Lecture odds ratio :")
print(f"   OR > 1 → augmente la probabilité de financement complet")
print(f"   OR < 1 → réduit la probabilité de financement complet")


# %% CELLULE 10 — Random Forest
print("🔄 Random Forest — Cross-validation 5-fold...")

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ))
])

scores_rf = cross_val_score(pipe_rf, X, y, cv=cv, scoring="roc_auc")

print(f"\n📊 RANDOM FOREST :")
print(f"   AUC-ROC (CV=5) : {scores_rf.mean():.3f} ± {scores_rf.std():.3f}")
print(f"   Scores par fold: {[f'{s:.3f}' for s in scores_rf]}")

# Comparaison LR vs RF
print(f"\n📊 COMPARAISON :")
print(f"   Régression Logistique : {scores_lr.mean():.3f}")
print(f"   Random Forest         : {scores_rf.mean():.3f}")
print(f"   Écart                 : {abs(scores_rf.mean() - scores_lr.mean()):.3f}")
print(f"\n   → Si écart < 0.02 : relations essentiellement linéaires")
print(f"   → Si écart > 0.05 : interactions non-linéaires importantes")


# %% CELLULE 11 — Feature Importance (Random Forest)
pipe_rf.fit(X, y)
rf_model = pipe_rf.named_steps["model"]

importance_df = pd.DataFrame({
    "feature"    : X.columns,
    "importance" : rf_model.feature_importances_
}).sort_values("importance", ascending=False).head(15)

print(f"\n📊 Top 15 features importantes (Random Forest) :\n")
for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 500)
    print(f"  {row['feature']:<40} {bar:<30} {row['importance']:.4f}")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Déterminants du Financement Complet — Kiva ASS",
             fontsize=13, fontweight="bold")

# ── LR Coefficients ──────────────────────────────────────
coef_plot = coef_df.sort_values("coef")
bar_colors = ["#C73E1D" if c < 0 else "#1A6B5C" for c in coef_plot["coef"]]
bars = axes[0].barh(
    coef_plot["feature"],
    coef_plot["coef"],
    color=bar_colors, alpha=0.85
)
axes[0].axvline(x=0, color="black", linewidth=1)
axes[0].set_title("Régression Logistique\n(coefficients standardisés)")
axes[0].set_xlabel("Coefficient")
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# ── RF Feature Importance ────────────────────────────────
imp_plot = importance_df.sort_values("importance")
axes[1].barh(
    imp_plot["feature"],
    imp_plot["importance"],
    color="#2E86AB", alpha=0.85
)
axes[1].set_title("Random Forest\n(feature importance)")
axes[1].set_xlabel("Importance")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(FIGURES / "06_determinants_financement.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Figure sauvegardée → 06_determinants_financement.png")


# %% CELLULE 12 — Synthèse finale
print(f"""
{'='*60}
 SYNTHÈSE FINALE — NOTEBOOK 06
{'='*60}

 FIELD PARTNERS :
   Partners actifs (≥100 prêts) : {len(partner_stats)}
   Gini concentration           : {gini_partners:.3f}
   Partners pour 80% du volume  : {n_for_80}

 MODÉLISATION :
   AUC Régression Logistique    : {scores_lr.mean():.3f}
   AUC Random Forest            : {scores_rf.mean():.3f}

 Figures produites :
   → 06_efficiency_frontier.png
   → 06_determinants_financement.png

{'='*60}
 PROJET ANALYTIQUE TERMINÉ
{'='*60}

 Notebooks complétés : 01 → 06
 Figures produites   : 12+
 Findings documentés : 7

   Données prêtes dans data/processed/loans_ssa_mpi.parquet
""")
# %%
