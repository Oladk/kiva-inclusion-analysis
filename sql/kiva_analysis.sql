-- ============================================================
-- ANALYSE KIVA — Requêtes Analytiques SQLite
-- Projet  : Inclusion Financière ASS
-- Auteur  : Ronald Dossou-Kohi | ISE Statisticien
-- Données : 171 391 prêts Kiva — 28 pays ASS — 2014-2019
-- ============================================================
-- Séparateur de requêtes : -- @@
-- Chaque requête répond à une question analytique précise
-- ============================================================


-- @@
-- Q1 : Distribution sectorielle et adéquation à l'emploi ASS
SELECT
    f.sector,
    COUNT(*)                                         AS n_loans,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_kiva,
    s.pct_emploi,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER ()
        - s.pct_emploi,
    2)                                               AS gap_allocation,
    ROUND(AVG(f.loan_amount), 0)                     AS montant_moyen,
    ROUND(AVG(f.is_female) * 100, 1)                 AS pct_femmes
FROM fact_loans f
LEFT JOIN dim_sector s ON f.sector = s.sector
GROUP BY f.sector, s.pct_emploi
ORDER BY gap_allocation ASC


-- @@
-- Q2 : Top 10 Field Partners par impact (volume + inclusivité + pauvreté)
SELECT
    partner_id,
    n_loans,
    ROUND(total_volume, 0)       AS volume_usd,
    ROUND(pct_female, 1)         AS pct_femmes,
    ROUND(avg_mpi, 3)            AS mpi_moyen_clientele,
    ROUND(pct_funded, 1)         AS taux_financement_pct,
    n_countries,
    profil
FROM dim_partner
WHERE n_loans >= 100
ORDER BY
    CASE profil
        WHEN 'Champion'    THEN 1
        WHEN 'Pro-genre'   THEN 2
        WHEN 'Pro-pauvres' THEN 3
        ELSE 4
    END,
    total_volume DESC
LIMIT 10


-- @@
-- Q3 : Gap de genre par secteur — montant moyen F vs M
SELECT
    sector,
    ROUND(AVG(CASE WHEN gender_clean = 'female' THEN loan_amount END), 0) AS mediane_femmes,
    ROUND(AVG(CASE WHEN gender_clean = 'male'   THEN loan_amount END), 0) AS mediane_hommes,
    ROUND(
        AVG(CASE WHEN gender_clean = 'male'   THEN loan_amount END) /
        AVG(CASE WHEN gender_clean = 'female' THEN loan_amount END),
    2)                                                                     AS ratio_H_F,
    COUNT(CASE WHEN gender_clean = 'female' THEN 1 END)                   AS n_femmes,
    COUNT(CASE WHEN gender_clean = 'male'   THEN 1 END)                   AS n_hommes
FROM fact_loans
WHERE gender_clean IN ('female', 'male')
GROUP BY sector
HAVING n_femmes >= 100 AND n_hommes >= 100
ORDER BY ratio_H_F DESC


-- @@
-- Q4 : Pauvreté vs financement par pays
--      Répond à : Kiva atteint-il les pays les plus pauvres ?
SELECT
    c.country,
    c.sub_region,
    c.n_loans,
    ROUND(c.pct_of_ssa, 2)          AS pct_du_total_ssa,
    ROUND(c.avg_mpi, 3)             AS mpi_moyen,
    ROUND(c.pct_female, 1)          AS pct_femmes,
    ROUND(c.median_amount, 0)       AS montant_median,
    c.n_partners,
    CASE
        WHEN c.avg_mpi >= 0.4 AND c.pct_of_ssa >= 5
            THEN 'Bien couvert ET pauvre ✅'
        WHEN c.avg_mpi >= 0.4 AND c.pct_of_ssa < 2
            THEN 'Pauvre ET sous-couvert ⚠️'
        WHEN c.avg_mpi < 0.3 AND c.pct_of_ssa >= 5
            THEN 'Bien couvert, moins pauvre'
        ELSE 'Standard'
    END                             AS diagnostic
FROM dim_country c
ORDER BY c.avg_mpi DESC


-- @@
-- Q5 : Saisonnalité des prêts agricoles par mois
--      Répond à : faut-il des produits crédit saisonniers ?
SELECT
    CAST(posted_month AS INTEGER)    AS mois,
    CASE CAST(posted_month AS INTEGER)
        WHEN 1  THEN 'Janvier'   WHEN 2  THEN 'Février'
        WHEN 3  THEN 'Mars'      WHEN 4  THEN 'Avril'
        WHEN 5  THEN 'Mai'       WHEN 6  THEN 'Juin'
        WHEN 7  THEN 'Juillet'   WHEN 8  THEN 'Août'
        WHEN 9  THEN 'Septembre' WHEN 10 THEN 'Octobre'
        WHEN 11 THEN 'Novembre'  WHEN 12 THEN 'Décembre'
    END                              AS nom_mois,
    COUNT(*)                         AS n_prets_agricoles,
    ROUND(COUNT(*) * 100.0 /
        SUM(COUNT(*)) OVER (), 2)    AS pct_du_total,
    ROUND(AVG(loan_amount), 0)       AS montant_moyen,
    ROUND(AVG(is_female) * 100, 1)   AS pct_femmes
FROM fact_loans
WHERE LOWER(sector) = 'agriculture'
  AND posted_month IS NOT NULL
GROUP BY posted_month
ORDER BY mois
```
