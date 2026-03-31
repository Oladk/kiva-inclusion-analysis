# Analyse de l'Inclusion Financière : Afrique Subsaharienne
### Données Kiva.org | 171 391 prêts | 28 pays | 2014–2019

**Auteur :** Ronald Dossou-Kohi | ISE Statisticien | Data Analyst  
**Stack :** Python · SQL · Power BI · Git  
**Statut :**  Terminé

---

## Contexte

En Afrique subsaharienne, 58% des adultes sont bancarisés (Global Findex 2024),
avec un gap de genre de 12 points et des taux sous 20% dans plusieurs pays sahéliens.
Ce projet analyse les flux de microcrédit Kiva pour identifier les gaps d'allocation
et produire des recommandations actionnables pour les IMF et les bailleurs.

> **Avertissement méthodologique :** Les données Kiva représentent un sous-ensemble
> biaisé du microcrédit en ASS. Chaque conclusion est accompagnée de ses limites explicites.

---

## 7 Findings Clés

| # | Finding | Chiffre |
|---|---------|---------|
| F1 | Distribution géographique très concentrée | Gini = 0.704 |
| F2 | Aucune corrélation pauvreté-financement | ρ = -0.11, p = 0.61 |
| F3 | Agriculture structurellement sous-financée | Gap = -21.5 pts vs emploi ASS |
| F4 | Saisonnalité agricole significative | Pic en mars, χ² p < 0.001 |
| F5 | Gap de genre faible en volume | 70.1% nombre vs 70.7% volume |
| F6 | Prêts féminins financés 44% plus vite | 9.2j vs 13.3j |
| F7 | Taille du prêt = déterminant #1 | AUC Random Forest = 0.866 |

**Rapport complet :** [reports/findings_report.md](reports/findings_report.md)

---

## Structure du Projet
```
kiva-inclusion-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_exploration.py       ← EDA initiale
│   ├── 02_nettoyage.py         ← Nettoyage & feature engineering
│   ├── 03_geographie.py        ← Analyse géographique + MPI
│   ├── 04_sectoriel.py         ← Analyse sectorielle + benchmark emploi
│   ├── 05_genre.py             ← Gap de genre
│   ├── 06_fieldpartners.py     ← Field Partners & modélisation
│   └── 07_export_powerbi.py    ← Export tables pour Power BI
│
├── data/
│   ├── raw/                    ← ⛔ Gitignored (CSV Kiva ~200 Mo)
│   └── processed/              ← ⛔ Gitignored (générés par les notebooks)
│       └── powerbi/            ← 6 tables CSV pour Power BI
│
├── reports/
│   ├── findings_report.md      ← Rapport des 7 findings
│   └── figures/                ← 12+ visualisations PNG
│
├── sql/
│   └── kiva_analysis.sql       ← Requêtes analytiques SQLite
│
└── docs/
    └── Inclusion Financière - Kiva.pbix      ← Dashboard Power BI
```

---

## Démarrage Rapide

### Prérequis
- Python 3.9+
- Power BI Desktop (pour le dashboard)
- ~200 Mo d'espace disque pour les données brutes

### Installation
```bash
# 1. Cloner le repo
git clone https://github.com/Oladk/kiva-inclusion-analysis.git
cd kiva-inclusion-analysis

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger les données Kiva
# → https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding
# → Placer les 4 fichiers CSV dans data/raw/

# 5. Exécuter les notebooks dans l'ordre (01 → 07)
```

### Données requises (`data/raw/`)
```
kiva_loans.csv                    ← Dataset principal (~196 Mo)
kiva_mpi_region_locations.csv     ← Index de pauvreté multidimensionnelle
loan_themes_by_region.csv         ← Thèmes de prêts par région
loan_theme_ids.csv                ← Identifiants des thèmes
```

---

## Dashboard Power BI

Le dashboard (3 onglets) est construit sur les 6 tables exportées
dans `data/processed/powerbi/` par le notebook 07.

| Onglet | Contenu |
|--------|---------|
| Vue Exécutive | KPIs globaux, carte géographique, évolution temporelle |
| Analyse Sectorielle | Expected vs Actual, heatmap pays × secteur |
| Genre & Field Partners | Gap genre, Efficiency Frontier des partenaires |

---

## Références

- **World Bank** (2024). *Global Findex Database 2024*
- **CGAP** (2023). *Microfinance Consensus Guidelines*
- **OIT / Banque Mondiale** (2022). *Emploi par secteur : Afrique Subsaharienne*
- **BCEAO** (2023). *Rapport annuel sur les services financiers décentralisés*

---
  
*Ronald Dossou-Kohi | Cotonou, Bénin | 2025*
```
