-- Fantasy Basketball Ranker -- SQL Query Library
-- ESPN H2H Points: PTS=1, 3PM=1, FGM=2, FGA=-1, FTM=1, FTA=-1
--                  REB=1, AST=2, STL=4, BLK=4, TOV=-2
--
-- Run via: python main.py insights
-- Or directly against data/fantasy.db with any SQLite client.


-- 1. Top 30 by fantasy PPG (current season, min 15 GP)
SELECT
    p.name,
    p.position,
    ps.gp,
    ROUND(fs.fantasy_ppg, 2) AS fantasy_ppg,
    ROW_NUMBER() OVER (ORDER BY fs.fantasy_ppg DESC) AS rank
FROM fantasy_scores fs
JOIN players p       ON p.player_id = fs.player_id
JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
WHERE fs.season = :season
  AND ps.gp >= 15
ORDER BY fs.fantasy_ppg DESC
LIMIT 30;


-- 2. Top 10 per position
WITH ranked AS (
    SELECT
        p.name,
        p.position,
        ps.gp,
        ROUND(fs.fantasy_ppg, 2) AS fantasy_ppg,
        ROW_NUMBER() OVER (
            PARTITION BY p.position
            ORDER BY fs.fantasy_ppg DESC
        ) AS pos_rank
    FROM fantasy_scores fs
    JOIN players p       ON p.player_id = fs.player_id
    JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
    WHERE fs.season = :season
      AND ps.gp >= 15
      AND p.position IN ('PG', 'SG', 'SF', 'PF', 'C')
)
SELECT * FROM ranked WHERE pos_rank <= 10
ORDER BY position, pos_rank;


-- 3. Value over replacement (fantasy PPG minus positional average)
WITH pos_avg AS (
    SELECT p.position, AVG(fs.fantasy_ppg) AS avg_pos_ppg
    FROM fantasy_scores fs
    JOIN players p       ON p.player_id = fs.player_id
    JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
    WHERE fs.season = :season
      AND ps.gp >= 15
      AND p.position IN ('PG', 'SG', 'SF', 'PF', 'C')
    GROUP BY p.position
)
SELECT
    p.name,
    p.position,
    ROUND(fs.fantasy_ppg, 2)                       AS fantasy_ppg,
    ROUND(pa.avg_pos_ppg, 2)                        AS pos_avg,
    ROUND(fs.fantasy_ppg - pa.avg_pos_ppg, 2)       AS value_over_replacement
FROM fantasy_scores fs
JOIN players p       ON p.player_id = fs.player_id
JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
JOIN pos_avg pa      ON pa.position = p.position
WHERE fs.season = :season
  AND ps.gp >= 15
ORDER BY value_over_replacement DESC
LIMIT 30;


-- 4. Hidden gems (top-40 fantasy PPG players drafted after round 8)
WITH top40 AS (
    SELECT
        fs.player_id,
        p.name,
        p.position,
        ROUND(fs.fantasy_ppg, 2) AS fantasy_ppg,
        ROW_NUMBER() OVER (ORDER BY fs.fantasy_ppg DESC) AS season_rank
    FROM fantasy_scores fs
    JOIN players p       ON p.player_id = fs.player_id
    JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
    WHERE fs.season = :season
      AND ps.gp >= 15
    LIMIT 40
),
late_drafted AS (
    SELECT DISTINCT player_name FROM draft_history WHERE round > 8
)
SELECT t.name, t.position, t.fantasy_ppg, t.season_rank
FROM top40 t
JOIN late_drafted ld ON LOWER(ld.player_name) = LOWER(t.name)
ORDER BY t.fantasy_ppg DESC;


-- 5. Year-over-year risers (fantasy PPG up 15%+, min 15 GP both seasons)
SELECT
    p.name,
    p.position,
    ROUND(prev.fantasy_ppg, 2)                                              AS ppg_prev,
    ROUND(curr.fantasy_ppg, 2)                                              AS ppg_curr,
    ROUND((curr.fantasy_ppg - prev.fantasy_ppg) / prev.fantasy_ppg * 100, 1) AS pct_change
FROM fantasy_scores curr
JOIN fantasy_scores prev ON prev.player_id = curr.player_id
JOIN players p           ON p.player_id = curr.player_id
JOIN player_stats ps_c   ON ps_c.player_id = curr.player_id AND ps_c.season = curr.season
JOIN player_stats ps_p   ON ps_p.player_id = prev.player_id AND ps_p.season = prev.season
WHERE curr.season     = :season
  AND prev.season     = :prev_season
  AND ps_c.gp >= 15
  AND ps_p.gp >= 15
  AND curr.fantasy_ppg >= prev.fantasy_ppg * 1.15
ORDER BY pct_change DESC;
