#!/usr/bin/env python3
"""
fantasy-bball-ranker CLI

  python main.py ingest                         Pull top-100 stats from ESPN for 6 seasons → SQLite
  python main.py export                         Export all seasons to outputs/player_rankings.xlsx
  python main.py sample-stats                   Show top-10 current season players (verify ingest)
  python main.py rankings                       Generate current-season rankings using trained models
  python main.py rank [SEASON] [PLAYERS]        Rank any season / player subset by your preferences
                                                  SEASON  e.g. "2023-24" (default: current)
                                                  PLAYERS comma-separated names e.g. "Jokic,Luka,SGA"
  python main.py swipe                          Launch swipe comparison UI at localhost:5001
  python main.py train                          Retrain intuition model on swipe + draft history
  python main.py stat-model                     Run stat model analysis, save plots to outputs/
  python main.py espn-import                    Re-sync draft history from ESPN league
  python main.py insights                       Run analytical SQL queries (VOR, risers, hidden gems)
  python main.py test                           Run pytest suite
  python main.py status                         Print DB counts, model info, injury flags

Setup: credentials are read from .env in the project root (already configured).
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def _load_dotenv() -> None:
    env_path = BASE_DIR / ".env"
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


_load_dotenv()


def cmd_ingest() -> None:
    print("=== Running ETL Ingest (ESPN → SQLite) ===")
    from etl.ingest import run_ingest
    run_ingest()


def cmd_export() -> None:
    print("=== Exporting to Excel ===")
    from db import DB_PATH
    from etl.export import export_to_excel
    if not DB_PATH.exists():
        print("No database found. Run `python main.py ingest` first.")
        return
    out = export_to_excel(DB_PATH)
    print(f"Saved: {out}")


def cmd_sample_stats() -> None:
    import sqlite3
    from db import CURRENT_SEASON, DB_PATH

    if not DB_PATH.exists():
        print("No database found. Run `python main.py ingest` first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT p.name, p.position, p.injury_status,
               ps.gp, ps.pts, ps.reb, ps.ast, ps.stl, ps.blk, ps.tov,
               ps.fg3m, ps.gp_prev_season,
               ROUND(fs.fantasy_ppg, 1) AS fantasy_ppg
        FROM player_stats ps
        JOIN players p         ON p.player_id = ps.player_id
        JOIN fantasy_scores fs ON fs.player_id = ps.player_id AND fs.season = ps.season
        WHERE ps.season = ?
        ORDER BY fs.fantasy_ppg DESC
        LIMIT 10
        """,
        (CURRENT_SEASON,),
    ).fetchall()
    conn.close()

    if not rows:
        print(f"No data for {CURRENT_SEASON}. Run `python main.py ingest` first.")
        return

    cols   = ["Player", "Pos", "Injury", "GP", "FPPG", "PTS", "REB", "AST", "STL", "BLK", "TOV", "3PM", "PrevGP"]
    widths = [22, 4, 12, 4, 7, 6, 6, 6, 6, 6, 6, 5, 6]

    def _row(vals: list) -> str:
        return "  ".join(str(v).ljust(w) for v, w in zip(vals, widths))

    print(f"\n=== Top 10 — {CURRENT_SEASON} ===\n")
    print(_row(cols))
    print("-" * 105)
    for r in rows:
        print(_row([
            r["name"][:21], r["position"] or "?",
            r["injury_status"] or "ACTIVE",
            r["gp"], r["fantasy_ppg"],
            round(r["pts"], 1), round(r["reb"], 1), round(r["ast"], 1),
            round(r["stl"], 1), round(r["blk"], 1), round(r["tov"], 1),
            round(r["fg3m"] or 0, 1),
            r["gp_prev_season"] or "—",
        ]))
    print()


def cmd_rankings() -> None:
    print("=== Generating Rankings (sklearn — no LLM) ===")
    from models.ranker import generate_rankings
    generate_rankings()


def cmd_rank() -> None:
    """
    Flexible ranking command: rank any season, optionally filtered to specific players.
      python main.py rank                          → current season, top 20
      python main.py rank 2023-24                  → 2023-24 season, top 20
      python main.py rank 2023-24 "Jokic,Luka,SGA" → 2023-24, just those players
    """
    from db import ALL_SEASONS, CURRENT_SEASON
    from models.ranker import rank_players

    args = sys.argv[2:]  # everything after "rank"
    season = CURRENT_SEASON
    player_names = None

    if args:
        first = args[0]
        # Detect if first arg looks like a season string (YYYY-YY)
        if len(first) == 7 and first[4] == "-" and first in ALL_SEASONS:
            season = first
            args = args[1:]
        elif first not in ALL_SEASONS and "-" in first and len(first) == 7:
            print(f"Unknown season '{first}'. Available: {', '.join(ALL_SEASONS)}")
            return
        if args:
            player_names = [n.strip() for n in " ".join(args).split(",") if n.strip()]

    label = f"Season: {season}"
    if player_names:
        label += f"  Players: {', '.join(player_names)}"
    print(f"=== Ranking — {label} ===")
    rank_players(season, player_names)


def cmd_swipe() -> None:
    print("=== Launching Swipe UI at http://localhost:5001 ===")
    from app.tinder import run_app
    run_app(debug=False)


def cmd_train() -> None:
    print("=== Training Intuition Model ===")
    from models.intuition_model import get_preference_profile, train_model
    metrics = train_model()
    print("\n--- Preference Profile ---")
    print(get_preference_profile())
    print(f"\nCV accuracy: {metrics['cv_mean_accuracy']:.2%}")


def cmd_stat_model() -> None:
    print("=== Running Statistical Model Analysis ===")
    from models.stat_model import run_analysis
    run_analysis()


def cmd_espn_import() -> None:
    print("=== Importing ESPN Draft History ===")
    from etl.espn_import import main as espn_main
    espn_main()


def cmd_insights() -> None:
    import sqlite3
    from db import ALL_SEASONS, CURRENT_SEASON, DB_PATH

    if not DB_PATH.exists():
        print("No database. Run `python main.py ingest` first.")
        return

    prev_season = ALL_SEASONS[ALL_SEASONS.index(CURRENT_SEASON) - 1] if CURRENT_SEASON in ALL_SEASONS else None
    sql_path    = BASE_DIR / "sql" / "queries.sql"
    raw_sql     = sql_path.read_text()

    # Split on blank lines between statements, skip comment-only blocks
    blocks = [b.strip() for b in raw_sql.split(";\n\n") if b.strip()]
    queries = []
    for block in blocks:
        lines = [l for l in block.splitlines() if not l.strip().startswith("--") and l.strip()]
        if lines:
            queries.append(block.rstrip(";"))

    queries_meta = [
        (
            "Top 30 by Fantasy PPG",
            "Baseline ranking by raw production. Use this to see who the model"
            " has to work with before any personal preferences are applied.",
        ),
        (
            "Top 10 per Position",
            "Positional scarcity matters in drafts. A PG ranked 8th overall might"
            " be the 2nd-best PG available -- this shows the depth at each spot"
            " so you know when to reach vs. wait.",
        ),
        (
            "Value Over Replacement",
            "Raw PPG does not tell the full story. A center averaging 43 points"
            " is more valuable than a point guard averaging 43 points if the"
            " average center scores 30 and the average PG scores 34. VOR"
            " measures how much better a player is than the baseline you could"
            " get off the waiver wire at that position.",
        ),
        (
            "Hidden Gems (drafted after round 8, now top-40)",
            "Cross-references your personal draft history against current"
            " performance. Any player here was considered a late-round flier"
            " at draft time but is now producing like a top-40 asset -- the"
            " kind of pick that wins leagues.",
        ),
        (
            "Year-over-Year Risers (up 15%+)",
            "Identifies breakout candidates before a draft. A player who jumped"
            " 30% in fantasy PPG from one season to the next is either in a"
            " better role, on a better team, or has genuinely improved --"
            " worth targeting early before the rest of the league catches on.",
        ),
    ]

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    for (title, why), sql in zip(queries_meta, queries):
        if ":prev_season" in sql and not prev_season:
            continue

        params = {"season": CURRENT_SEASON}
        if prev_season:
            params["prev_season"] = prev_season

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception as e:
            print(f"\n[{title}] skipped: {e}")
            continue

        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"  Why: {why}")
        print(f"{'='*60}")
        if not rows:
            print("  (no results)")
            continue

        cols   = rows[0].keys()
        widths = [max(len(str(c)), max(len(str(r[c])) for r in rows)) for c in cols]
        header = "  " + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
        print(header)
        print("  " + "-" * (len(header) - 2))
        for r in rows:
            print("  " + "  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)))

    conn.close()


def cmd_test() -> None:
    print("=== Running pytest Suite ===")
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=BASE_DIR)
    sys.exit(result.returncode)


def cmd_status() -> None:
    import sqlite3
    from db import DB_PATH
    from models.intuition_model import get_model_confidence

    if not DB_PATH.exists():
        print("No database. Run `python main.py ingest` first.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    players  = conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
    seasons  = [r[0] for r in conn.execute("SELECT DISTINCT season FROM player_stats ORDER BY season").fetchall()]
    swipes   = conn.execute("SELECT COUNT(*) FROM swipe_history").fetchone()[0]
    model    = conn.execute("SELECT * FROM model_versions ORDER BY id DESC LIMIT 1").fetchone()
    injured  = conn.execute("SELECT COUNT(*) FROM players WHERE injury_status NOT IN ('ACTIVE','NORMAL','')").fetchone()[0]
    conn.close()

    outputs      = sorted((BASE_DIR / "outputs").glob("*.md"), reverse=True)
    last_run     = outputs[0].stem if outputs else "—"
    last_trained = model["trained_at"][:16] if model else "—"

    print("=== fantasy-bball-ranker status ===")
    print(f"Players in DB:     {players}")
    print(f"Currently injured: {injured}")
    print(f"Seasons loaded:    {', '.join(seasons) or '—'}")
    print(f"Swipes recorded:   {swipes}")
    print(f"Model confidence:  {get_model_confidence()}")
    print(f"Last trained:      {last_trained}")
    if model:
        print(f"Train/Test acc:    {model['train_accuracy']:.2%} / {model['test_accuracy']:.2%}")
    print(f"Last rankings:     {last_run}")


COMMANDS = {
    "ingest":       cmd_ingest,
    "export":       cmd_export,
    "sample-stats": cmd_sample_stats,
    "rankings":     cmd_rankings,
    "rank":         cmd_rank,
    "swipe":        cmd_swipe,
    "train":        cmd_train,
    "stat-model":   cmd_stat_model,
    "espn-import":  cmd_espn_import,
    "insights":     cmd_insights,
    "test":         cmd_test,
    "status":       cmd_status,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
