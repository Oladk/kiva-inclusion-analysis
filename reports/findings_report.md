# Analyse de l'Inclusion Financière en Afrique Subsaharienne
### Données Kiva.org : 171 391 prêts | 28 pays | 2014–2019

**Auteur :** Ronald Dossou-Kohi | ISE Statisticien | Data Analyst  
**Outils :** Python · SQL · Power BI  
**Données :** [Kiva Data Science for Good : Kaggle](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding)

---

## Contexte & Objectif

En Afrique subsaharienne, 58% des adultes sont bancarisés (Global Findex 2024),
avec un gap de genre de 12 points et des taux sous 20% dans plusieurs pays sahéliens.
Kiva.org constitue l'un des rares corpus multi-pays, standardisés et publics
sur les flux de microcrédit en ASS.

**Question centrale :** Dans quelle mesure les flux Kiva reflètent-ils une allocation
pro-pauvres, pro-genre et sectoriellement alignée sur les besoins économiques réels ?

---

## Chiffres Clés

| Indicateur | Valeur |
|-----------|--------|
| Prêts analysés | 171 391 |
| Volume total | $69.7M |
| Pays couverts | 28 sur ~48 en ASS |
| % Emprunteuses féminines | 70.1% |
| Taux de financement complet | 96.4% |
| Montant médian | $350 |
| Field Partners actifs | 72 |

---

## 7 Findings

### F1 : Distribution géographique très concentrée
**Gini géographique = 0.704** : plus inégal que la distribution des revenus
en ASS (~0.45). 9 pays sur 28 concentrent **80% des prêts**.
L'accès au microcrédit Kiva dépend de la présence institutionnelle
des Field Partners, pas des besoins des populations.

> *Implication : Les 20 pays absents ne manquent pas de pauvreté :
> ils manquent d'IMF partenaires.*

---

### F2 : Aucune corrélation pauvreté-financement
Corrélation Spearman entre IMP moyen et volume de prêts :
**ρ = -0.11, p = 0.61 (non significatif)**.
Le financement Kiva en ASS est géographiquement neutre par rapport
à la pauvreté multidimensionnelle.

> *Implication : Kiva ne cible pas structurellement les zones les plus pauvres.
> L'allocation suit l'infrastructure, pas les besoins.*

---

### F3 : Agriculture structurellement sous-financée
L'agriculture emploie **54% des actifs** en ASS mais ne capte que
**32.5% des prêts Kiva**; donc un gap de **-21.5 points**.

| Secteur | % Prêts Kiva | % Emploi ASS | Écart |
|---------|-------------|--------------|-------|
| Agriculture | 32.5% | 54.0% | **-21.5 pts** |
| Food/Commerce | 20.1% | 18.0% | +2.1 pts |
| Services | 12.3% | 13.0% | -0.7 pts |

> *Implication : Les IMF sous-financent le secteur qui emploie
> la majorité des populations rurales pauvres.*

---

### F4 : Saisonnalité agricole significative
Les prêts agricoles ne sont pas uniformément distribués sur l'année.
**Pic en mars** (préparation des semailles), creux en fin d'année.
Test χ² hautement significatif (p < 0.001).

> *Implication : Les produits crédit agricole doivent être
> structurés autour du calendrier cultural, pas du calendrier bancaire.*

---

### F5 : Gap de genre faible en volume
Les femmes représentent **70.1% des prêts** en nombre
et **70.7% du volume** total, ce qui fait un écart de seulement **0.6 points**.
Résultat contre-intuitif : contrairement à la narrative habituelle,
Kiva ASS alloue le capital de façon quasi-équitable en volume.

L'écart de montant médian ($350 F vs $400 M) est statistiquement
significatif (Mann-Whitney, p < 0.001) mais d'**effet négligeable
(r = 0.031)**.

> *Implication : Le gap de genre en microfinancement ASS est moins
> sévère en volume qu'en nombre. L'enjeu est la qualité des prêts,
> pas leur taille.*

---

### F6 : Prêts féminins financés 44% plus vite
Délai médian de financement : **9.2 jours (femmes) vs 13.3 jours (hommes)**.
Les prêts féminins sont narrativement plus attractifs pour les
prêteurs Kiva internationaux.

> *Implication : Biais de plateforme favorable aux femmes :
> avantage sur Kiva qui ne se reproduit pas nécessairement
> dans le secteur financier formel.*

---

### F7 : Taille du prêt = déterminant #1 du financement
Modèle Random Forest : **AUC-ROC = 0.866**.
La feature la plus importante est `log_loan_amount` :
les petits prêts se financent systématiquement plus facilement et
plus vite, indépendamment du secteur, du pays ou du genre.

| Modèle | AUC-ROC |
|--------|---------|
| Régression Logistique | 0.791 |
| Random Forest | **0.866** |

L'écart de 0.075 confirme des interactions non-linéaires importantes
entre montant, secteur et géographie.

---

## Analyse des Field Partners

**72 partenaires actifs** (≥100 prêts) avec Gini de concentration = 0.649.
22 partenaires concentrent 80% du volume.

Classification par profil d'impact :

| Profil | N | Description |
|--------|---|-------------|
| **Champion** | **14** | Fort % femmes + clientèle pauvre |
| Pro-genre | ~20 | Fort % femmes, clientèle moins pauvre |
| Pro-pauvres | ~25 | Clientèle pauvre, moins féminisé |
| Standard | **13** | Ni fort % femmes ni clientèle pauvre |

> *Les 14 partners Champions sont les candidats prioritaires
> pour les lignes de crédit concessionnelles (AFD, UNCDF, BEI).*

---

## Limites Méthodologiques

| Limite | Impact | Mention obligatoire |
|--------|--------|-------------------|
| Kiva ≠ microfinance totale ASS | Fort | Dans toute présentation |
| 20 pays absents (sans Field Partners) | Fort | Dans l'analyse géographique |
| `is_fully_funded` ≠ remboursement réel | Critique | Dans la modélisation |
| IMP : match exact = 9.2% seulement | Modéré | Dans l'analyse MPI |
| Données 2014-2019 : pas de données récentes | Modéré | Dans les recommandations |

---

## Recommandations

**Pour une IMF :**
1. Développer des produits crédit agricole saisonniers
   calés sur le calendrier cultural (mars = pic de demande)
2. Cibler les segments sous-financés : agriculture + femmes rurales
3. Benchmarker son portefeuille sur le ratio volume/nombre par genre

**Pour un bailleur (AFD, UNCDF) :**
1. Orienter les lignes de crédit vers les 14 Field Partners Champions
2. Conditionner les financements à la couverture des pays absents
3. Exiger des données de remboursement réel pour compléter l'analyse

**Pour un décideur politique (BCEAO, Ministère des Finances) :**
1. Créer des incitations pour l'implantation d'IMF dans les
   20 pays ASS absents de Kiva
2. Standardiser le reporting sectoriel des IMF pour permettre
   une analyse transversale nationale

---

## Stack Technique
```
Python 3.9+     pandas · numpy · matplotlib · seaborn · scipy · scikit-learn
Power BI        Dashboard 3 onglets — modèle en étoile (6 tables)
Git/GitHub      Versioning complet du projet
```

---

## Structure du Projet
```
kiva-inclusion-analysis/
├── notebooks/          # 7 notebooks Python séquentiels
├── data/processed/     # Données nettoyées + exports Power BI
├── reports/figures/    # 12+ visualisations exportées
├── sql/                # Requêtes analytiques SQL
└── docs/               # Cadre analytique + dictionnaire des données
```

**Repo GitHub :** [github.com/Oladk/kiva-inclusion-analysis](https://github.com/Oladk/kiva-inclusion-analysis)

---
 
*Ronald Dossou-Kohi | Cotonou, Bénin | 2025*