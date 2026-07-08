"""
Pull player universe + stats from ESPN Fantasy Basketball.

All data comes from the mRoster view which returns every rostered player with
their real NBA season stats and live injury status.  No other API needed.

Confirmed ESPN stat category IDs (verified against Jokic/Curry/Shai 2025-26):
  [0]  PTS total      [1]  BLK total      [2]  STL total
  [3]  AST total      [4]  OREB total     [5]  DREB total
  [6]  REB total      [9]  TOV total
  [13] FGM total      [14] FGA total
  [15] FTM total      [16] FTA total
  [17] 3PM total      [18] 3PA total
  [42] GP (games played)

Percentages (pre-computed by ESPN, ignored — we compute our own):
  [19] FG%            [20] FT%           [21] 3P%

All values with statSplitTypeId=0, statSourceId=0, scoringPeriodId=0 = full season actual.
"""

import json
import os
import sys
import urllib.parse
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import create_tables, get_conn

_API_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba"

_POSITION_MAP = {
    1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C",
    6: "SG", 7: "SF", 8: "PF", 9: "C",
}

# Maps ESPN numeric stat IDs → our field names
_ESPN_STAT = {
    "0":  "pts",
    "1":  "blk",
    "2":  "stl",
    "3":  "ast",
    "6":  "reb",
    "9":  "tov",
    "13": "fgm",
    "14": "fga",
    "15": "ftm",
    "16": "fta",
    "17": "fg3m",
    "42": "gp",
}


def _espn_year_to_season(year: int) -> str:
    return f"{year - 1}-{str(year)[2:]}"


def _season_to_espn_year(season: str) -> int:
    return int(season[:4]) + 1  # "2024-25" → 2025


class ESPNClient:
    def __init__(self, league_id: int, espn_s2: str, swid: str):
        self.league_id = league_id
        self.cookies   = {
            "espn_s2": urllib.parse.unquote(espn_s2),
            "SWID":    swid,
        }
        self.headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

    def _get(self, year: int, view: str, extra_params: dict | None = None) -> dict | None:
        url    = f"{_API_BASE}/seasons/{year}/segments/0/leagues/{self.league_id}"
        params = {"view": view, **(extra_params or {})}
        r      = requests.get(url, params=params, cookies=self.cookies,
                              headers=self.headers, timeout=20)
        text   = r.text.strip()
        if r.status_code == 200 and text.startswith("{"):
            return json.loads(text)
        return None

    def available_years(self, search_from: int = 2027, search_back_to: int = 2019) -> list[int]:
        found = []
        for year in range(search_from, search_back_to, -1):
            if self._get(year, "mSettings") is not None:
                found.append(year)
            elif found:
                break
        return sorted(found)

    def draft_picks(self, year: int) -> list[dict]:
        d = self._get(year, "mDraftDetail")
        return d.get("draftDetail", {}).get("picks", []) if d else []

    def resolve_players(self, player_ids: list[int], reference_year: int = 2026) -> dict[int, dict]:
        """Batch-resolve ESPN player IDs → {name, position}."""
        result = {}
        for i in range(0, len(player_ids), 50):
            batch = player_ids[i : i + 50]
            f     = json.dumps({"players": {"filterIds": {"value": batch}}})
            r     = requests.get(
                f"{_API_BASE}/seasons/{reference_year}/players",
                params={"scoringPeriodId": 0, "view": "players_wl"},
                headers={**self.headers, "x-fantasy-filter": f},
                cookies=self.cookies,
                timeout=15,
            )
            if r.status_code == 200 and r.text.strip().startswith("["):
                for pl in json.loads(r.text):
                    pid  = pl.get("id")
                    name = f"{pl.get('firstName', '')} {pl.get('lastName', '')}".strip()
                    pos  = _POSITION_MAP.get(pl.get("defaultPositionId", 0), "?")
                    result[pid] = {"name": name, "position": pos}
        return result

    def fetch_season(self, year: int, top_n: int = 100) -> list[dict]:
        """
        Fetch all rostered players for `year` with their full-season NBA stats.
        Returns list of player dicts sorted by fantasy_ppg desc, capped at top_n.

        Each dict has:
          espn_id, name, position, injury_status,
          gp, pts, reb, ast, stl, blk, tov, fgm, fga, ftm, fta, fg3m,
          fantasy_ppg (computed from stats)
        """
        from etl.scoring import compute_fantasy_score

        d = self._get(year, "mRoster")
        if d is None:
            return []

        players = []
        seen: set[int] = set()

        for team in d.get("teams", []):
            for entry in team.get("roster", {}).get("entries", []):
                pool   = entry.get("playerPoolEntry", {})
                player = pool.get("player", {})
                pid    = player.get("id")
                if not pid or pid in seen:
                    continue
                seen.add(pid)

                name   = player.get("fullName", "").strip()
                if not name:
                    continue

                pos         = _POSITION_MAP.get(player.get("defaultPositionId", 0), "?")
                injury      = player.get("injuryStatus", "ACTIVE") or "ACTIVE"
                pro_team_id = int(player.get("proTeamId", 0) or 0)

                # Find the full-season actual stat row
                stats_dict: dict[str, float] = {}
                for stat_entry in player.get("stats", []):
                    if (
                        stat_entry.get("statSplitTypeId") == 0
                        and stat_entry.get("statSourceId")  == 0
                        and stat_entry.get("scoringPeriodId") == 0
                    ):
                        stats_dict = {str(k): float(v) for k, v in stat_entry.get("stats", {}).items()}
                        break

                gp = int(stats_dict.get("42", 0))
                if gp == 0:
                    continue

                # Per-game averages
                def _pg(key: str) -> float:
                    return stats_dict.get(key, 0.0) / gp

                pts  = _pg("0")
                blk  = _pg("1")
                stl  = _pg("2")
                ast  = _pg("3")
                reb  = _pg("6")
                tov  = _pg("9")
                fgm  = _pg("13")
                fga  = _pg("14")
                ftm  = _pg("15")
                fta  = _pg("16")
                fg3m = _pg("17")
                fg3a = _pg("18")
                min_pg = float(stats_dict.get("28", 0.0))

                fantasy_ppg = compute_fantasy_score(pts, fg3m, fgm, fga, ftm, fta, reb, ast, stl, blk, tov)

                players.append({
                    "espn_id":       pid,
                    "name":          name,
                    "position":      pos,
                    "injury_status": injury,
                    "pro_team_id":   pro_team_id,
                    "gp":            gp,
                    "min_pg":        round(min_pg, 1),
                    "pts":           round(pts,  2),
                    "reb":           round(reb,  2),
                    "ast":           round(ast,  2),
                    "stl":           round(stl,  2),
                    "blk":           round(blk,  2),
                    "tov":           round(tov,  2),
                    "fgm":           round(fgm,  2),
                    "fga":           round(fga,  2),
                    "ftm":           round(ftm,  2),
                    "fta":           round(fta,  2),
                    "fg3m":          round(fg3m, 2),
                    "fg3a":          round(fg3a, 2),
                    "fantasy_ppg":   round(fantasy_ppg, 2),
                })

        # Sort by fantasy PPG and return top_n
        players.sort(key=lambda p: p["fantasy_ppg"], reverse=True)
        return players[:top_n]


# ── Draft import ───────────────────────────────────────────────────────────────

def import_all_drafts(league_id: int, espn_s2: str, swid: str,
                      team_id: int = 2, replace: bool = True) -> int:
    client = ESPNClient(league_id, espn_s2, swid)

    print("Discovering available seasons...")
    years = client.available_years()
    if not years:
        print("  No seasons found. Check credentials.")
        return 0
    print(f"  Found seasons: {[_espn_year_to_season(y) for y in years]}")

    all_picks: dict[int, list[dict]] = {}
    all_pids:  set[int] = set()
    for year in years:
        picks = client.draft_picks(year)
        all_picks[year] = picks
        all_pids.update(p["playerId"] for p in picks)

    print(f"Resolving {len(all_pids)} unique players...")
    ref_year   = max(y for y in years if y <= 2027)
    player_map = client.resolve_players(list(all_pids), reference_year=ref_year)
    print(f"  Resolved {len(player_map)} players.")

    conn = get_conn()
    create_tables(conn)

    if replace:
        conn.execute("DELETE FROM draft_history")
        conn.commit()

    total = 0
    for year in years:
        season_str = _espn_year_to_season(year)
        our_picks  = [p for p in all_picks[year] if p["teamId"] == team_id]
        rows = []
        for p in our_picks:
            info = player_map.get(p["playerId"], {})
            rows.append((
                season_str, p["roundId"], p["roundPickNumber"],
                info.get("name", f"id={p['playerId']}"),
                info.get("position", "?"),
                0.0,
            ))
        conn.executemany(
            "INSERT INTO draft_history (season,round,pick_number,player_name,position,fantasy_ppg_that_season) VALUES (?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        total += len(rows)
        print(f"  {season_str}: inserted {len(rows)} picks for Team C-K")

    conn.close()
    print(f"\nDone. {total} total picks imported.")
    return total


# ── Public helpers ─────────────────────────────────────────────────────────────

def fetch_player_universe(league_id: int, espn_s2: str, swid: str,
                          year: int = 2026, top_n: int = 100) -> list[dict]:
    """Return top_n players for the given season year with full stats."""
    client  = ESPNClient(league_id, espn_s2, swid)
    players = client.fetch_season(year, top_n=top_n)
    if players:
        print(f"  ESPN: {len(players)} players for {_espn_year_to_season(year)}")
    else:
        print("  Could not fetch ESPN player universe (check credentials or try later).")
    return players


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Import ESPN fantasy data")
    parser.add_argument("--league-id", type=int, default=int(os.environ.get("ESPN_LEAGUE_ID", 723769079)))
    parser.add_argument("--team-id",   type=int, default=int(os.environ.get("ESPN_TEAM_ID", 2)))
    parser.add_argument("--espn-s2",   default=os.environ.get("ESPN_S2", ""))
    parser.add_argument("--swid",      default=os.environ.get("ESPN_SWID", ""))
    parser.add_argument("--universe",  action="store_true")
    args = parser.parse_args()

    if not args.espn_s2 or not args.swid:
        parser.error("ESPN_S2 and SWID required.")

    import_all_drafts(args.league_id, args.espn_s2, args.swid, args.team_id)

    if args.universe:
        print("\nFetching player universe...")
        players = fetch_player_universe(args.league_id, args.espn_s2, args.swid)
        for p in players[:10]:
            print(f"  {p['name']:<25} {p['position']:<4} {p['fantasy_ppg']:>6.1f} fppg  {p['gp']} GP")


if __name__ == "__main__":
    main()
