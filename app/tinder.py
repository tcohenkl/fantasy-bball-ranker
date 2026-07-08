"""Flask swipe UI for collecting player preference data."""

import random
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import CURRENT_SEASON, get_conn
from models.intuition_model import _usg_pg, get_model_confidence, get_preference_profile, train_model
from models.ranker import _build_ranking

app = Flask(__name__)

RETRAIN_EVERY   = 20
MIN_GP_FOR_SWIPE = 15


def _top_players(limit: int = 100) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT p.player_id, p.position,
               ps.gp, ps.min, ps.pts, ps.reb, ps.ast, ps.stl, ps.blk, ps.tov,
               ps.fg3m, ps.fg3a, ps.fgm, ps.fga, ps.ftm, ps.fta,
               ps.gp_prev_season,
               ps.team_abbr, ps.team_wins, ps.team_losses, ps.team_seed, ps.team_conf,
               fs.fantasy_ppg
        FROM fantasy_scores fs
        JOIN players p       ON p.player_id = fs.player_id
        JOIN player_stats ps ON ps.player_id = fs.player_id AND ps.season = fs.season
        WHERE fs.season = ?
          AND ps.gp >= ?
        ORDER BY fs.fantasy_ppg DESC
        LIMIT ?
        """,
        (CURRENT_SEASON, MIN_GP_FOR_SWIPE, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _seen_pairs() -> set[frozenset]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT winner_id, loser_id FROM swipe_history WHERE winner_id != 0"
    ).fetchall()
    conn.close()
    return {frozenset([r["winner_id"], r["loser_id"]]) for r in rows}


def _pick_matchup() -> tuple[dict, dict] | None:
    players = _top_players()
    if len(players) < 2:
        return None

    seen = _seen_pairs()
    candidates = [
        (a, b)
        for i, a in enumerate(players)
        for b in players[max(0, i - 20): i + 20]
        if a["player_id"] != b["player_id"]
        and frozenset([a["player_id"], b["player_id"]]) not in seen
    ]

    if not candidates:
        # All within-20-rank pairs exhausted — open to any unseen pair
        candidates = [
            (a, b)
            for i, a in enumerate(players)
            for b in players[i + 1:]
            if frozenset([a["player_id"], b["player_id"]]) not in seen
        ]

    return random.choice(candidates) if candidates else None


def _swipe_count() -> int:
    conn = get_conn()
    n = conn.execute("SELECT COUNT(*) FROM swipe_history").fetchone()[0]
    conn.close()
    return n


def _trigger_retrain() -> None:
    def _work():
        try:
            train_model()
        except Exception as e:
            print(f"[retrain] {e}", flush=True)

    threading.Thread(target=_work, daemon=True).start()


@app.route("/")
def index():
    return render_template("swipe.html", page="landing",
                           swipe_count=_swipe_count(),
                           confidence=get_model_confidence())


@app.route("/swipe", methods=["GET"])
def swipe_get():
    matchup = _pick_matchup()
    if not matchup:
        return render_template("swipe.html", page="done",
                               swipe_count=_swipe_count(),
                               confidence=get_model_confidence())

    a, b = matchup
    return render_template("swipe.html", page="swipe",
                           player_a=a, player_b=b,
                           swipe_count=_swipe_count(),
                           confidence=get_model_confidence())


@app.route("/swipe", methods=["POST"])
def swipe_post():
    winner_id = request.form.get("winner_id")
    loser_id  = request.form.get("loser_id")
    if not winner_id or not loser_id:
        return redirect(url_for("swipe_get"))

    conn   = get_conn()
    w_row  = conn.execute(
        "SELECT fantasy_ppg FROM fantasy_scores WHERE player_id = ? AND season = ?",
        (winner_id, CURRENT_SEASON),
    ).fetchone()
    l_row  = conn.execute(
        "SELECT fantasy_ppg FROM fantasy_scores WHERE player_id = ? AND season = ?",
        (loser_id, CURRENT_SEASON),
    ).fetchone()

    w_ppg = w_row["fantasy_ppg"] if w_row else 0.0
    l_ppg = l_row["fantasy_ppg"] if l_row else 0.0
    upset = 1 if w_ppg < l_ppg else 0

    conn.execute(
        """
        INSERT INTO swipe_history
            (timestamp, winner_id, loser_id, winner_fantasy_ppg, loser_fantasy_ppg, upset_flag)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(timespec="seconds"),
         winner_id, loser_id, w_ppg, l_ppg, upset),
    )
    conn.commit()
    conn.close()

    if _swipe_count() % RETRAIN_EVERY == 0:
        _trigger_retrain()

    return redirect(url_for("swipe_get"))


@app.route("/train", methods=["POST"])
def train_endpoint():
    _trigger_retrain()
    return jsonify({"status": "retraining started"})


@app.route("/rankings")
def rankings():
    from db import ALL_SEASONS
    season = request.args.get("season", CURRENT_SEASON)
    if season not in ALL_SEASONS:
        season = CURRENT_SEASON
    rows, _ranking_mode = _build_ranking(season, top_n=40)
    for r in rows:
        mpg       = float(r.get("min") or 0)
        fgm       = float(r.get("fgm") or 0)
        fga       = float(r.get("fga") or 0)
        ftm       = float(r.get("ftm") or 0)
        fta       = float(r.get("fta") or 0)
        fg3m      = float(r.get("fg3m") or 0)
        fg3a      = float(r.get("fg3a") or 0)
        r["fg_pct"]  = round(fgm / fga * 100, 1) if fga else 0
        r["ft_pct"]  = round(ftm / fta * 100, 1) if fta else 0
        r["fg3_pct"] = round(fg3m / fg3a * 100, 1) if fg3a else 0
        r["usg_pct"] = round(_usg_pg(r) / mpg * 36, 1) if mpg else 0
    return render_template(
        "swipe.html", page="rankings",
        rows=rows, season=season,
        confidence=get_model_confidence(),   # always from swipe DB, never "NO MODEL"
        ranking_mode=_ranking_mode,          # "HIGH" / "NO MODEL" etc for table label
        all_seasons=ALL_SEASONS,
        swipe_count=_swipe_count(),
    )


@app.route("/stats")
def stats():
    conn      = get_conn()
    totals    = conn.execute(
        "SELECT COUNT(*) AS total, SUM(upset_flag) AS upsets FROM swipe_history"
    ).fetchone()
    model_ver = conn.execute(
        "SELECT * FROM model_versions ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()

    return render_template(
        "swipe.html", page="stats",
        swipe_count=totals["total"],
        upset_count=totals["upsets"] or 0,
        confidence=get_model_confidence(),
        model_version=dict(model_ver) if model_ver else None,
        preference_profile=get_preference_profile(),
    )


def run_app(debug: bool = False) -> None:
    print("Swipe UI running at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=debug)
