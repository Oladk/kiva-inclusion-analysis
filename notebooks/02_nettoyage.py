# ============================================================
# NOTEBOOK 02 : Nettoyage & Feature Engineering
# Projet : Analyse Inclusion Financière ASS
# Auteur : Ronald Dossou-Kohi
# ============================================================

# %% CELLULE 1 : Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

ROOT      = Path("..")
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

print(" Imports OK")


# %% CELLULE 2 : Rechargement des données brutes
loans = pd.read_csv(DATA_RAW / "kiva_loans.csv", low_memory=False)
print(f" {loans.shape[0]:,} lignes × {loans.shape[1]} colonnes")


# %% CELLULE 3 : Copie défensive
# ─────────────────────────────────────────────────────────
# On ne modifie JAMAIS le DataFrame original.
# Si une étape de nettoyage produit un résultat inattendu,
# on peut toujours revenir à `loans` sans recharger le CSV.
# ─────────────────────────────────────────────────────────
clean = loans.copy()
print(f" Copie créée : {len(clean):,} lignes")


# %% CELLULE 4 : Nettoyage des dates
# ─────────────────────────────────────────────────────────
# Les colonnes de dates arrivent comme des strings depuis le CSV.
# Toute analyse temporelle (tendances, délais) nécessite
# des vrais objets datetime, pas des chaînes de caractères.
# ─────────────────────────────────────────────────────────
date_cols = ["posted_time", "disbursed_time", "funded_time", "date"]

for col in date_cols:
    if col in clean.columns:
        clean[col] = pd.to_datetime(clean[col], errors="coerce", utc=True)

# Extraire les composantes temporelles utiles
clean["posted_year"]  = clean["posted_time"].dt.year
clean["posted_month"] = clean["posted_time"].dt.month

# Délai de financement en jours
# ─────────────────────────────────────────────────────────
# NOTE ANALYTIQUE IMPORTANTE :
# days_to_fund peut être NÉGATIF pour les prêts "pré-décaissés".
# Kiva permet aux IMF de décaisser le prêt AVANT de le financer
# sur la plateforme. Ce n'est pas une erreur : c'est une feature
# du modèle Kiva. On garde donc ces valeurs et on les flagge.
# ─────────────────────────────────────────────────────────
clean["days_to_fund"] = (
    clean["funded_time"] - clean["posted_time"]
).dt.total_seconds() / 86400

n_negative = (clean["days_to_fund"] < 0).sum()
print(f" Dates converties")
print(f"   days_to_fund négatifs (pré-décaissés) : {n_negative:,} ({n_negative/len(clean)*100:.1f}%)")


# %% CELLULE 5 : Gestion des valeurs manquantes
# ─────────────────────────────────────────────────────────
# PHILOSOPHIE : documenter chaque décision
# On ne supprime pas sans justifier.
# On n'impute pas sans documenter le biais introduit.
# ─────────────────────────────────────────────────────────

journal = []  # On garde une trace de chaque décision

# ── funded_amount et loan_amount ─────────────────────────
# Décision : SUPPRIMER les lignes où le montant est manquant
# Justification : le montant est la variable centrale.
# L'imputer reviendrait à inventer la donnée principale.
# Biais : < 0.5% des données : impact minimal
before = len(clean)
clean = clean.dropna(subset=["funded_amount", "loan_amount"])
n_dropped = before - len(clean)
journal.append(f"Supprimées (montant manquant)  : {n_dropped:,} lignes")

# ── region ───────────────────────────────────────────────
# Décision : REMPLACER NaN par "Unknown" et créer un flag
# Justification : le pays est connu même si la région ne l'est pas.
# Supprimer ces lignes ferait perdre des données valides.
# Biais : les prêts "Unknown" sont probablement plus urbains
#         (les zones urbaines ont moins de précision régionale)
n_region_nan = clean["region"].isnull().sum()
clean["region_missing"] = clean["region"].isnull().astype(int)
clean["region"]         = clean["region"].fillna("Unknown")
journal.append(f"region NaN → 'Unknown' (flag)  : {n_region_nan:,} lignes")

# ── tags ─────────────────────────────────────────────────
# Décision : REMPLACER NaN par chaîne vide
# Justification : l'absence de tag est une information valide,
# pas une donnée manquante. Un prêt sans tag ≠ donnée incomplète.
clean["tags"] = clean["tags"].fillna("")
journal.append(f"tags NaN → '' (absence valide) : {clean['tags'].eq('').sum():,} lignes")

# ── use ──────────────────────────────────────────────────
clean["use"] = clean["use"].fillna("Not specified")
journal.append("use NaN → 'Not specified'")

# ── funded_time ──────────────────────────────────────────
# Décision : CONSERVER les NaN et créer un flag
# Justification : funded_time = NaN signifie soit que le prêt
# n'a pas encore été entièrement financé, soit que la date
# n'est pas renseignée. C'est une information utile en soi.
n_unfunded = clean["funded_time"].isnull().sum()
clean["is_unfunded"] = clean["funded_time"].isnull().astype(int)
journal.append(f"funded_time NaN conservés (flag) : {n_unfunded:,} lignes")

# ── Rapport des décisions ─────────────────────────────────
print("\n JOURNAL DES DÉCISIONS DE NETTOYAGE :")
for decision in journal:
    print(f"   → {decision}")

print(f"\n Après nettoyage : {len(clean):,} lignes (était {len(loans):,})")


# %% CELLULE 6 : Normalisation des textes
# ─────────────────────────────────────────────────────────
# POURQUOI normaliser les textes ?
# "agriculture", "Agriculture", "AGRICULTURE" sont la même valeur.
# Sans normalisation, les groupby produisent des catégories
# en double et faussent toutes les statistiques sectorielles.
# ─────────────────────────────────────────────────────────
text_cols = ["sector", "activity", "country", "repayment_interval", "currency"]

for col in text_cols:
    if col in clean.columns:
        clean[col] = clean[col].str.strip().str.title()

print(" Textes normalisés (strip + title case)")
print(f"   Secteurs distincts : {clean['sector'].nunique()}")
print(f"   Pays distincts     : {clean['country'].nunique()}")


# %% CELLULE 7 : Feature Engineering : Genre
# ─────────────────────────────────────────────────────────
# COMPLEXITÉ DE LA COLONNE borrower_genders :
# Elle contient :
#   "female"                     → 1 femme seule
#   "male"                       → 1 homme seul
#   "female, female, male"       → groupe mixte
#   "female, female, female"     → groupe de femmes
#   NaN                          → non renseigné
#
# On crée 4 variables dérivées :
#   gender_clean  : female / male / mixed / unknown
#   is_female     : True si emprunteur individuel féminin
#   is_group      : True si plusieurs emprunteurs
#   pct_female    : proportion de femmes dans le groupe
# ─────────────────────────────────────────────────────────

def parse_gender(raw):
    """
    Classifie la colonne borrower_genders en catégories propres.
    Retourne un DataFrame avec les 4 colonnes dérivées.
    """
    raw_str = raw.fillna("").str.lower().str.strip()

    # Taille du groupe
    def count_total(s):
        if not s:
            return np.nan
        parts = [x.strip() for x in s.split(",") if x.strip()]
        return len(parts)

    def count_female(s):
        if not s:
            return np.nan
        parts = [x.strip() for x in s.split(",") if x.strip()]
        return sum(1 for p in parts if p == "female")

    group_size   = raw_str.apply(count_total)
    female_count = raw_str.apply(count_female)
    is_group     = group_size > 1
    pct_female   = (female_count / group_size).where(group_size > 0)

    # Classification principale
    def classify(s):
        if not s:
            return "unknown"
        parts  = [x.strip() for x in s.split(",") if x.strip()]
        unique = set(parts)
        if not unique:
            return "unknown"
        if unique == {"female"}:
            return "female"
        if unique == {"male"}:
            return "male"
        if "female" in unique and "male" in unique:
            return "mixed"
        return "unknown"

    gender_clean = raw_str.apply(classify)
    is_female    = (gender_clean == "female")

    return pd.DataFrame({
        "gender_clean"  : gender_clean,
        "is_female"     : is_female,
        "is_group"      : is_group,
        "group_size"    : group_size,
        "pct_female"    : pct_female,
    })

gender_df = parse_gender(clean["borrower_genders"])
clean = pd.concat([clean, gender_df], axis=1)

print(" Variables genre créées")
print("\n Distribution gender_clean :")
dist = clean["gender_clean"].value_counts(normalize=True).mul(100).round(1)
for cat, pct in dist.items():
    bar = "█" * int(pct / 2)
    print(f"   {cat:<12} {bar:<35} {pct:>5.1f}%")


# %% CELLULE 8 : Feature Engineering : Montants & Financement
# ─────────────────────────────────────────────────────────
# VARIABLES CRÉÉES :
#
# funding_ratio     : funded_amount / loan_amount
#   → mesure le taux d'atteinte de l'objectif sur Kiva
#   → peut dépasser 1.0 (sur-financement possible)
#
# is_fully_funded   : funding_ratio >= 0.999
#   → variable cible pour l'analyse du financement
#   → NOTE : ce n'est PAS le remboursement du prêt à l'IMF
#
# loan_size_category : classification CGAP par taille de prêt
#   → Micro (<300$), Petite (300-2000$), Moyenne (2000-10000$)
#   → seuils standards utilisés par CGAP et la Banque Mondiale
#
# log_loan_amount   : log(1 + loan_amount)
#   → nécessaire pour la modélisation (ratio moy/méd = 1.68)
#   → normalise la distribution asymétrique
# ─────────────────────────────────────────────────────────

# Taux de financement
clean["funding_ratio"]   = clean["funded_amount"] / clean["loan_amount"].replace(0, np.nan)
clean["is_fully_funded"] = (clean["funding_ratio"] >= 0.999).astype(int)

# Taille des prêts (seuils CGAP)
clean["loan_size_category"] = pd.cut(
    clean["loan_amount"],
    bins   = [0, 300, 2000, 10000, float("inf")],
    labels = ["Micro (<$300)", "Petite ($300-2k)", "Moyenne ($2k-10k)", "Grande (>$10k)"],
    right  = False
)

# Log-transformation
clean["log_loan_amount"] = np.log1p(clean["loan_amount"])

# Vérifications
n_over_funded = (clean["funding_ratio"] > 1.01).sum()
n_fully_funded = clean["is_fully_funded"].sum()

print(" Variables montants créées")
print(f"\n   funding_ratio — min/max : {clean['funding_ratio'].min():.2f} / {clean['funding_ratio'].max():.2f}")
print(f"   Sur-financés (>100%)    : {n_over_funded:,} ({n_over_funded/len(clean)*100:.1f}%)")
print(f"   Entièrement financés    : {n_fully_funded:,} ({n_fully_funded/len(clean)*100:.1f}%)")
print(f"\n Répartition par taille de prêt (CGAP) :")
size_dist = clean["loan_size_category"].value_counts(normalize=True).mul(100).round(1)
for cat, pct in size_dist.items():
    print(f"   {str(cat):<22} {pct:>5.1f}%")


# %% CELLULE 9 : Feature Engineering : Géographie
# ─────────────────────────────────────────────────────────
# On crée deux variables géographiques :
#
# is_subsaharan  : True si le pays est en Afrique Subsaharienne
#   → Basé sur la classification Banque Mondiale (codes ISO-2)
#
# sub_region     : sous-région africaine (Ouest / Est / Centrale / Australe)
#   → Basé sur la classification de l'Union Africaine
# ─────────────────────────────────────────────────────────

# Codes ISO-2 des pays d'Afrique Subsaharienne
SSA_CODES = {
    # Afrique de l'Ouest
    "BJ","BF","CV","CI","GM","GH","GN","GW","LR","ML",
    "MR","NE","NG","SN","SL","TG",
    # Afrique de l'Est
    "BI","KM","DJ","ER","ET","KE","MG","MW","MU","MZ",
    "RW","SC","SO","SS","TZ","UG","ZM","ZW",
    # Afrique Centrale
    "CM","CF","TD","CG","CD","GQ","GA",
    # Afrique Australe
    "AO","BW","LS","NA","ZA","SZ"
}

WEST  = {"BJ","BF","CV","CI","GM","GH","GN","GW","LR","ML","MR","NE","NG","SN","SL","TG"}
EAST  = {"BI","KM","DJ","ER","ET","KE","MG","MW","MU","MZ","RW","SC","SO","SS","TZ","UG","ZM","ZW"}
CENT  = {"CM","CF","TD","CG","CD","GQ","GA"}
SOUTH = {"AO","BW","LS","NA","ZA","SZ"}

def get_sub_region(code):
    if code in WEST:  return "Afrique de l'Ouest"
    if code in EAST:  return "Afrique de l'Est"
    if code in CENT:  return "Afrique Centrale"
    if code in SOUTH: return "Afrique Australe"
    return "Hors ASS"

clean["is_subsaharan"] = clean["country_code"].isin(SSA_CODES)
clean["sub_region"]    = clean["country_code"].apply(get_sub_region)

# Stats
n_ssa     = clean["is_subsaharan"].sum()
n_pays    = clean[clean["is_subsaharan"]]["country"].nunique()

print(" Variables géographiques créées")
print(f"\n   Prêts en Afrique Subsaharienne : {n_ssa:,} ({n_ssa/len(clean)*100:.1f}%)")
print(f"   Pays ASS couverts              : {n_pays}")
print(f"\n Répartition par sous-région :")
sub_dist = clean[clean["is_subsaharan"]]["sub_region"].value_counts()
for region, count in sub_dist.items():
    print(f"   {region:<25} {count:>7,} ({count/n_ssa*100:.1f}%)")


# %% CELLULE 10 : Récapitulatif des nouvelles colonnes
nouvelles_cols = [c for c in clean.columns if c not in loans.columns]

print(f"\n{'='*55}")
print(f" RÉCAPITULATIF : {len(nouvelles_cols)} colonnes créées")
print(f"{'='*55}\n")

for col in sorted(nouvelles_cols):
    dtype    = str(clean[col].dtype)
    n_unique = clean[col].nunique()
    n_null   = clean[col].isnull().sum()
    print(f"   {col:<25} | {dtype:<10} | unique: {n_unique:<6} | null: {n_null:,}")


# %% CELLULE 11 : Export du dataset nettoyé

# Dataset global nettoyé
out_global = DATA_PROC / "loans_clean.parquet"
clean.to_parquet(out_global, index=False)
print(f" loans_clean.parquet sauvegardé")
print(f"   {len(clean):,} lignes × {len(clean.columns)} colonnes")

# Dataset ASS uniquement (pour les notebooks 03-06)
loans_ssa = clean[clean["is_subsaharan"]].copy()
out_ssa   = DATA_PROC / "loans_ssa.parquet"
loans_ssa.to_parquet(out_ssa, index=False)
print(f"\n loans_ssa.parquet sauvegardé")
print(f"   {len(loans_ssa):,} prêts ASS")

print(f"""
{'='*55}
 NOTEBOOK 02 TERMINÉ
{'='*55}

 Fichiers produits :
   → data/processed/loans_clean.parquet  ({len(clean):,} prêts)
   → data/processed/loans_ssa.parquet    ({len(loans_ssa):,} prêts ASS)

 Nouvelles variables : {len(nouvelles_cols)} colonnes ajoutées

""")