-- ============================================================
-- Fantasy Basketball Ranker — SQL Query Library
-- ESPN H2H Points: PTS=1, 3PM=1, FGM=2, FGA=-1, FTM=1, FTA=-1
--                  REB=1, AST=2, STL=4, BLK=4, TOV=-2
-- ============================================================


-- ------------------------------------------------------------
-- 1. Top 30 by fantasy PPG — current season, min 15 GP
-- ------------------------------------------------------------
SELECT
    p.name,
    p.team,
    p.position,
    ps.gp,
    ROUND(fs.fantasy_ppg, 2)    AS fantasy_ppg,
    ROUND(fs.consistency_score, 2) AS consistency_score,
    ROW_NUMBER() OVER (ORDER BY fs.fantasy_ppg DESC) AS rank
FROM fantasy_scores fs
JOIN players p ON p.player_id = fs.player_id
JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
WHERE fs.season = '2024-25'
  AND ps.gp >= 15
ORDER BY fs.fantasy_ppg DESC
LIMIT 30;


-- ------------------------------------------------------------
-- 2. Positional rankings — top 10 per position (PG, SG, SF, PF, C)
-- ------------------------------------------------------------
WITH ranked AS (
    SELECT
        p.name,
        p.team,
        p.position,
        ps.gp,
        ROUND(fs.fantasy_ppg, 2) AS fantasy_ppg,
        ROW_NUMBER() OVER (
            PARTITION BY p.position
            ORDER BY fs.fantasy_ppg DESC
        ) AS pos_rank
    FROM fantasy_scores fs
    JOIN players p ON p.player_id = fs.player_id
    JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
    WHERE fs.season = '2024-25'
      AND ps.gp >= 15
      AND p.position IN ('PG', 'SG', 'SF', 'PF', 'C')
)
SELECT *
FROM ranked
WHERE pos_rank <= 10
ORDER BY position, pos_rank;


-- ------------------------------------------------------------
-- 3. Value over replacement — player fantasy PPG minus average
--    fantasy PPG at their position (current season, min 15 GP)
-- ------------------------------------------------------------
WITH pos_avg AS (
    SELECT
        p.position,
        AVG(fs.fantasy_ppg) AS avg_pos_ppg
    FROM fantasy_scores fs
    JOIN players p ON p.player_id = fs.player_id
    JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
    WHERE fs.season = '2024-25'
      AND ps.gp >= 15
      AND p.position IN ('PG', 'SG', 'SF', 'PF', 'C')
    GROUP BY p.position
)
SELECT
    p.name,
    p.team,
    p.position,
    ROUND(fs.fantasy_ppg, 2)                          AS fantasy_ppg,
    ROUND(pa.avg_pos_ppg, 2)                          AS pos_avg_ppg,
    ROUND(fs.fantasy_ppg - pa.avg_pos_ppg, 2)         AS value_over_replacement
FROM fantasy_scores fs
JOIN players p ON p.player_id = fs.player_id
JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
JOIN pos_avg pa ON pa.position = p.position
WHERE fs.season = '2024-25'
  AND ps.gp >= 15
ORDER BY value_over_replacement DESC
LIMIT 30;


-- ------------------------------------------------------------
-- 4. Consistency kings — lowest std dev fantasy score,
--    min 20 GP, top 20
-- ------------------------------------------------------------
SELECT
    p.name,
    p.team,
    p.position,
    ps.gp,
    ROUND(fs.fantasy_ppg, 2)        AS fantasy_ppg,
    ROUND(fs.consistency_score, 2)  AS std_dev_fantasy
FROM fantasy_scores fs
JOIN players p ON p.player_id = fs.player_id
JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
WHERE fs.season = '2024-25'
  AND ps.gp >= 20
  AND fs.consistency_score IS NOT NULL
ORDER BY fs.consistency_score ASC
LIMIT 20;


-- ------------------------------------------------------------
-- 5. Boom-or-bust — highest std dev, min 20 GP, top 20
-- ------------------------------------------------------------
SELECT
    p.name,
    p.team,
    p.position,
    ps.gp,
    ROUND(fs.fantasy_ppg, 2)        AS fantasy_ppg,
    ROUND(fs.consistency_score, 2)  AS std_dev_fantasy
FROM fantasy_scores fs
JOIN players p ON p.player_id = fs.player_id
JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
WHERE fs.season = '2024-25'
  AND ps.gp >= 20
  AND fs.consistency_score IS NOT NULL
ORDER BY fs.consistency_score DESC
LIMIT 20;


-- ------------------------------------------------------------
-- 6. Hidden gems — top-40 fantasy PPG this season who appeared
--    after round 8 in any draft_history season
-- ------------------------------------------------------------
WITH top40 AS (
    SELECT
        fs.player_id,
        p.name,
        p.team,
        p.position,
        ROUND(fs.fantasy_ppg, 2) AS fantasy_ppg,
        ROW_NUMBER() OVER (ORDER BY fs.fantasy_ppg DESC) AS season_rank
    FROM fantasy_scores fs
    JOIN players p ON p.player_id = fs.player_id
    JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
    WHERE fs.season = '2024-25'
      AND ps.gp >= 15
    ORDER BY fs.fantasy_ppg DESC
    LIMIT 40
),
late_drafted AS (
    SELECT DISTINCT player_name
    FROM draft_history
    WHERE round > 8
)
SELECT
    t.name,
    t.team,
    t.position,
    t.fantasy_ppg,
    t.season_rank
FROM top40 t
JOIN late_drafted ld ON LOWER(ld.player_name) = LOWER(t.name)
ORDER BY t.fantasy_ppg DESC;


-- ------------------------------------------------------------
-- 7. Trend analysis — players whose fantasy PPG improved by
--    15%+ from 2023-24 to current season (min 15 GP both seasons)
-- ------------------------------------------------------------
SELECT
    p.name,
    p.team,
    p.position,
    ROUND(prev.fantasy_ppg, 2)                                    AS ppg_2023_24,
    ROUND(curr.fantasy_ppg, 2)                                    AS ppg_2024_25,
    ROUND(
        (curr.fantasy_ppg - prev.fantasy_ppg) / prev.fantasy_ppg * 100,
        1
    )                                                             AS pct_improvement
FROM fantasy_scores curr
JOIN fantasy_scores prev
    ON prev.player_id = curr.player_id
   AND prev.season = '2023-24'
JOIN players p ON p.player_id = curr.player_id
JOIN player_stats ps_curr
    ON ps_curr.player_id = curr.player_id AND ps_curr.season = curr.season
JOIN player_stats ps_prev
    ON ps_prev.player_id = prev.player_id AND ps_prev.season = prev.season
WHERE curr.season = '2024-25'
  AND ps_curr.gp >= 15
  AND ps_prev.gp >= 15
  AND curr.fantasy_ppg >= prev.fantasy_ppg * 1.15
ORDER BY pct_improvement DESC;
