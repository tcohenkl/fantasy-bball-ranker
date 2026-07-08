"""
Fetch NBA team standings from ESPN public API (no auth required).

Maps ESPN pro_team_id (used in fantasy roster) → team record.
The same numeric ID appears as proTeamId in fantasy player data
and as team.id in the standings response.
"""

import requests

_URL = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"


def fetch_standings(season_year: int) -> dict[int, dict]:
    """
    Returns {espn_team_id: {abbr, conf, wins, losses, win_pct, seed}}
    season_year is the ESPN year int: 2026 = 2025-26 season.
    Falls back to empty dict on any error so ingest still succeeds.
    """
    try:
        r = requests.get(_URL, params={"season": season_year}, timeout=10)
        d = r.json()
    except Exception:
        return {}

    result: dict[int, dict] = {}
    for conf in d.get("children", []):
        conf_name = conf.get("name", "")
        conf_short = "East" if "East" in conf_name else "West"
        for e in conf.get("standings", {}).get("entries", []):
            team = e.get("team", {})
            tid  = int(team.get("id", 0))
            if not tid:
                continue
            stats: dict[str, float] = {}
            for s in e.get("stats", []):
                if "value" in s:
                    stats[s["name"]] = float(s["value"])
                elif "displayValue" in s:
                    try:
                        stats[s["name"]] = float(s["displayValue"])
                    except ValueError:
                        pass
            wins    = int(stats.get("wins", 0))
            losses  = int(stats.get("losses", 0))
            win_pct = round(float(stats.get("winPercent", 0.5)), 3)
            seed    = int(stats.get("playoffSeed", 0))
            result[tid] = {
                "abbr":    team.get("abbreviation", "?"),
                "conf":    conf_short,
                "wins":    wins,
                "losses":  losses,
                "win_pct": win_pct,
                "seed":    seed,
            }
    return result
