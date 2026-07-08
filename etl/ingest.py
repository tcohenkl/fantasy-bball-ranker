"""
ETL: populate SQLite from ESPN Fantasy Basketball.

6 ESPN mRoster requests (one per season) → ~600 player-season rows.
No external APIs, no rate limiting, no IP blocking.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import ALL_SEASONS, CURRENT_SEASON, create_tables, get_conn
from etl.espn_import import ESPNClient, _season_to_espn_year
from etl.standings import fetch_standings


def _load_dotenv() -> None:
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


def _upsert_player(conn, espn_id: int, name: str, position: str, injury_status: str) -> int:
    conn.execute(
        """
        INSERT INTO players (player_id, name, team, position, espn_id, injury_status)
        VALUES (?, ?, '', ?, ?, ?)
        ON CONFLICT(player_id) DO UPDATE SET
            name=excluded.name,
            position=excluded.position,
            espn_id=excluded.espn_id,
            injury_status=excluded.injury_status
        """,
        (espn_id, name, position, espn_id, injury_status),
    )
    return espn_id


def _upsert_stats(conn, player_id: int, season: str, p: dict) -> None:
    conn.execute(
        """
        INSERT INTO player_stats
            (player_id, season, gp, min, pts, fg3m, fg3a, fgm, fga, ftm, fta, reb, ast, stl, blk, tov,
             team_abbr, team_wins, team_losses, team_win_pct, team_seed, team_conf)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_id, season) DO UPDATE SET
            gp=excluded.gp, min=excluded.min, pts=excluded.pts,
            fg3m=excluded.fg3m, fg3a=excluded.fg3a,
            fgm=excluded.fgm, fga=excluded.fga, ftm=excluded.ftm, fta=excluded.fta,
            reb=excluded.reb, ast=excluded.ast, stl=excluded.stl,
            blk=excluded.blk, tov=excluded.tov,
            team_abbr=excluded.team_abbr, team_wins=excluded.team_wins,
            team_losses=excluded.team_losses, team_win_pct=excluded.team_win_pct,
            team_seed=excluded.team_seed, team_conf=excluded.team_conf
        """,
        (player_id, season, p["gp"], p.get("min_pg", 0), p["pts"],
         p["fg3m"], p.get("fg3a", 0), p["fgm"], p["fga"], p["ftm"], p["fta"],
         p["reb"], p["ast"], p["stl"], p["blk"], p["tov"],
         p.get("team_abbr", ""), p.get("team_wins", 0), p.get("team_losses", 0),
         p.get("team_win_pct", 0.5), p.get("team_seed", 0), p.get("team_conf", "")),
    )


def _upsert_fantasy(conn, player_id: int, season: str, fantasy_ppg: float, gp: int) -> None:
    conn.execute(
        """
        INSERT INTO fantasy_scores (player_id, season, fantasy_ppg, fantasy_total, consistency_score)
        VALUES (?, ?, ?, ?, NULL)
        ON CONFLICT(player_id, season) DO UPDATE SET
            fantasy_ppg=excluded.fantasy_ppg,
            fantasy_total=excluded.fantasy_total
        """,
        (player_id, season, fantasy_ppg, round(fantasy_ppg * gp, 1)),
    )


def _compute_derived(conn) -> None:
    """Set gp_prev_season for every player-season row."""
    for i, season in enumerate(ALL_SEASONS):
        prev = ALL_SEASONS[i - 1] if i > 0 else None
        if not prev:
            continue
        rows = conn.execute("SELECT player_id FROM player_stats WHERE season=?", (season,)).fetchall()
        for row in rows:
            pid = row["player_id"]
            pr  = conn.execute(
                "SELECT gp FROM player_stats WHERE player_id=? AND season=?", (pid, prev)
            ).fetchone()
            gp_prev = pr["gp"] if pr else 0
            conn.execute(
                "UPDATE player_stats SET gp_prev_season=? WHERE player_id=? AND season=?",
                (gp_prev, pid, season),
            )
        conn.commit()


def run_ingest() -> None:
    _load_dotenv()

    league_id = int(os.environ.get("ESPN_LEAGUE_ID", 723769079))
    espn_s2   = os.environ.get("ESPN_S2", "")
    swid      = os.environ.get("ESPN_SWID", "")

    if not espn_s2 or not swid:
        print("ERROR: ESPN_S2 and ESPN_SWID not set in .env")
        return

    client = ESPNClient(league_id, espn_s2, swid)
    conn   = get_conn()
    create_tables(conn)

    print(f"Ingesting {len(ALL_SEASONS)} seasons from ESPN (top 100 each)...\n")

    for season in ALL_SEASONS:
        year = _season_to_espn_year(season)
        print(f"  {season}...", end=" ", flush=True)

        players = client.fetch_season(year, top_n=200)
        if not players:
            print("no data (ESPN may not have this year)")
            continue

        standings = fetch_standings(year)

        for p in players:
            team_info = standings.get(p.get("pro_team_id", 0), {})
            p["team_abbr"]   = team_info.get("abbr", "")
            p["team_wins"]   = team_info.get("wins", 0)
            p["team_losses"] = team_info.get("losses", 0)
            p["team_win_pct"] = team_info.get("win_pct", 0.5)
            p["team_seed"]   = team_info.get("seed", 0)
            p["team_conf"]   = team_info.get("conf", "")

            pid = _upsert_player(
                conn, p["espn_id"], p["name"], p["position"],
                p["injury_status"] if season == CURRENT_SEASON else "ACTIVE",
            )
            if p["team_abbr"]:
                conn.execute(
                    "UPDATE players SET team=? WHERE player_id=?",
                    (p["team_abbr"], pid),
                )
            _upsert_stats(conn, pid, season, p)
            _upsert_fantasy(conn, pid, season, p["fantasy_ppg"], p["gp"])

        conn.commit()
        top = players[0]
        print(f"{len(players)} players  (#{1}: {top['name']} {top['fantasy_ppg']:.1f} fppg)")

    print("\nComputing prior-season GP...", end=" ", flush=True)
    _compute_derived(conn)
    print("done")

    conn.close()
    print("\nIngest complete.")


if __name__ == "__main__":
    run_ingest()
