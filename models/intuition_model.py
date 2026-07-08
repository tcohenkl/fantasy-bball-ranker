# Logistic regression trained on swipe history and draft picks.
#
# Each training sample is a delta vector (winner stats minus loser stats),
# so the model learns which stat differences actually drove my choices.
# fantasy_ppg is excluded because it is a weighted sum of the other stats --
# including it would double-count and hide which individual stats I value.

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import CURRENT_SEASON, OUTPUT_DIR, get_conn

# Ordinal encoding so the delta captures guard/big preference (PG=1, C=5)
_POS_CODE: dict[str, float] = {
    "PG": 1.0, "SG": 2.0, "G": 1.5,
    "SF": 3.0, "F": 3.5,
    "PF": 4.0, "C": 5.0,
    "?":  3.0,
}

FEATURES = [
    "delta_pts",
    "delta_reb",
    "delta_ast",
    "delta_stl",
    "delta_blk",
    "delta_tov",
    "delta_fg3m",
    "delta_fg3a",
    "delta_fgm",
    "delta_fga",
    "delta_ftm",
    "delta_fta",
    "delta_usg_pg",        # FGA + 0.44*FTA + TOV per game (ball-dominant possession proxy)
    "delta_min",
    "delta_gp",
    "delta_position",
    "delta_gp_prev",       # prior-season GP as an injury availability signal
    "delta_team_win_pct",
]

# Draft round weights: early picks are strong signals, late picks are weak.
# Pairs more than 4 rounds apart are excluded to avoid spurious signals.
ROUND_WEIGHTS = {
    1: 1.00, 2: 0.85, 3: 0.75, 4: 0.65, 5: 0.55,
    6: 0.45, 7: 0.38, 8: 0.30, 9: 0.25, 10: 0.20,
    11: 0.15, 12: 0.12, 13: 0.10,
}

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"


def _player_stats_map(season: str = CURRENT_SEASON) -> dict[int, dict]:
    """Returns {player_id: stat_dict} for all players with data in the given season."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT
            p.player_id, p.name, p.position, p.injury_status,
            ps.gp, ps.min, ps.pts, ps.reb, ps.ast, ps.stl, ps.blk, ps.tov,
            ps.fg3m, ps.fg3a, ps.fgm, ps.fga, ps.ftm, ps.fta,
            ps.gp_prev_season, ps.team_win_pct,
            ps.team_abbr, ps.team_wins, ps.team_losses, ps.team_seed, ps.team_conf,
            fs.fantasy_ppg
        FROM player_stats ps
        JOIN players p         ON p.player_id = ps.player_id
        JOIN fantasy_scores fs ON fs.player_id = ps.player_id AND fs.season = ps.season
        WHERE ps.season = ?
        """,
        (season,),
    ).fetchall()
    conn.close()
    return {r["player_id"]: dict(r) for r in rows}


def _name_to_id_map() -> dict[str, int]:
    conn = get_conn()
    rows = conn.execute("SELECT player_id, name FROM players").fetchall()
    conn.close()
    return {r["name"].lower(): r["player_id"] for r in rows}


def _pos_code(player: dict) -> float:
    return _POS_CODE.get(str(player.get("position") or "?"), 3.0)


def _usg_pg(p: dict) -> float:
    """Usage possessions per game: FGA + 0.44*FTA + TOV."""
    return float(p.get("fga") or 0) + 0.44 * float(p.get("fta") or 0) + float(p.get("tov") or 0)


def _delta(winner: dict, loser: dict) -> list[float]:
    def _v(d: dict, key: str) -> float:
        return float(d.get(key) or 0)

    return [
        _v(winner, "pts")            - _v(loser, "pts"),
        _v(winner, "reb")            - _v(loser, "reb"),
        _v(winner, "ast")            - _v(loser, "ast"),
        _v(winner, "stl")            - _v(loser, "stl"),
        _v(winner, "blk")            - _v(loser, "blk"),
        _v(winner, "tov")            - _v(loser, "tov"),
        _v(winner, "fg3m")           - _v(loser, "fg3m"),
        _v(winner, "fg3a")           - _v(loser, "fg3a"),
        _v(winner, "fgm")            - _v(loser, "fgm"),
        _v(winner, "fga")            - _v(loser, "fga"),
        _v(winner, "ftm")            - _v(loser, "ftm"),
        _v(winner, "fta")            - _v(loser, "fta"),
        _usg_pg(winner)              - _usg_pg(loser),
        _v(winner, "min")            - _v(loser, "min"),
        _v(winner, "gp")             - _v(loser, "gp"),
        _pos_code(winner)            - _pos_code(loser),
        _v(winner, "gp_prev_season") - _v(loser, "gp_prev_season"),
        _v(winner, "team_win_pct")   - _v(loser, "team_win_pct"),
    ]


def _latest_artifact_path() -> Path | None:
    paths = sorted(OUTPUT_DIR.glob("intuition_model_*.pkl"), reverse=True)
    return paths[0] if paths else None


def _load_artifact() -> dict | None:
    path = _latest_artifact_path()
    if not path:
        return None
    artifact = joblib.load(path)
    # Reject models trained before the feature vector was expanded
    if len(artifact["clf"].coef_[0]) != len(FEATURES):
        return None
    return artifact


def _build_swipe_dataset(stats_map: dict[int, dict]) -> tuple[list, list]:
    conn = get_conn()
    swipes = conn.execute(
        "SELECT winner_id, loser_id FROM swipe_history WHERE winner_id != 0 AND loser_id != 0"
    ).fetchall()
    conn.close()

    X, y = [], []
    for s in swipes:
        w = stats_map.get(s["winner_id"])
        l = stats_map.get(s["loser_id"])
        if w and l:
            X.append(_delta(w, l));  y.append(1)
            X.append(_delta(l, w));  y.append(0)
    return X, y


def _build_draft_dataset() -> tuple[list, list]:
    """
    Synthetic preference signals from draft order.
    Each pick is matched against stats from that same season so the model
    sees the right context (e.g. a 2022-23 first-rounder compared using 2022-23 stats).
    """
    from collections import defaultdict
    from db import ALL_SEASONS

    conn = get_conn()
    picks = conn.execute(
        "SELECT season, round, player_name FROM draft_history ORDER BY season, round, pick_number"
    ).fetchall()
    conn.close()

    name_to_id   = _name_to_id_map()
    season_stats = {s: _player_stats_map(s) for s in ALL_SEASONS}

    by_season: dict[str, list] = defaultdict(list)
    for p in picks:
        pid = name_to_id.get(p["player_name"].lower())
        stats = None
        for candidate_season in [p["season"]] + ALL_SEASONS:
            if pid and pid in season_stats.get(candidate_season, {}):
                stats = season_stats[candidate_season][pid]
                break
        if pid and stats:
            by_season[p["season"]].append((int(p["round"]), stats))

    X, y = [], []
    for season_picks in by_season.values():
        for i, (early_round, early_stats) in enumerate(season_picks):
            for late_round, late_stats in season_picks[i + 1:]:
                round_diff = abs(late_round - early_round)
                weight     = ROUND_WEIGHTS.get(early_round, 0.10) * min(round_diff / 4, 1.0)
                if weight < 0.05:
                    continue
                reps = max(1, min(5, int(weight * 10)))
                for _ in range(reps):
                    X.append(_delta(early_stats, late_stats));  y.append(1)
                    X.append(_delta(late_stats, early_stats));  y.append(0)
    return X, y


def train_model() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_map = _player_stats_map()

    X_sw, y_sw = _build_swipe_dataset(stats_map)
    X_dr, y_dr = _build_draft_dataset()

    X_all = np.array(X_sw + X_dr, dtype=float)
    y_all = np.array(y_sw + y_dr, dtype=int)

    if len(X_all) < 10:
        raise RuntimeError("Not enough training data. Run ingest then collect swipes.")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc  = clf.score(X_test, y_test)
    cv_mean   = float(cross_val_score(clf, X_scaled, y_all, cv=5).mean())

    top_features = sorted(
        zip(FEATURES, clf.coef_[0].tolist()),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:5]

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = OUTPUT_DIR / f"intuition_model_{timestamp}.pkl"
    joblib.dump({"clf": clf, "scaler": scaler}, model_path)

    swipe_count = len(X_sw) // 2
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO model_versions
            (trained_at, swipe_count, train_accuracy, test_accuracy, top_features, model_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(timespec="seconds"), swipe_count,
         round(train_acc, 4), round(test_acc, 4),
         json.dumps(top_features), str(model_path)),
    )
    conn.commit()
    conn.close()

    _plot_learning_curve(clf, scaler, X_all, y_all)

    metrics = {
        "train_accuracy":   round(train_acc, 4),
        "test_accuracy":    round(test_acc, 4),
        "cv_mean_accuracy": round(cv_mean, 4),
        "swipe_count":      swipe_count,
        "model_path":       str(model_path),
        "top_features":     top_features,
    }

    print(f"  Train accuracy : {train_acc:.2%}")
    print(f"  Test accuracy  : {test_acc:.2%}")
    print(f"  CV mean        : {cv_mean:.2%}")
    print(f"  Swipe count    : {swipe_count}")
    print(f"  Saved          : {model_path.name}")
    return metrics


def _plot_learning_curve(clf: LogisticRegression, scaler: StandardScaler,
                         X_raw: np.ndarray, y: np.ndarray) -> None:
    X = scaler.transform(X_raw)
    sizes, train_scores, test_scores = learning_curve(
        clf, X, y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring="accuracy"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)

    for scores, color, label in [
        (train_scores, "#4ade80", "Training accuracy"),
        (test_scores,  "#f97316", "CV accuracy"),
    ]:
        mu  = scores.mean(axis=1)
        std = scores.std(axis=1)
        ax.plot(sizes, mu, "o-", color=color, label=label)
        ax.fill_between(sizes, mu - std, mu + std, alpha=0.15, color=color)

    ax.axvline(20, color="#60a5fa", linestyle="--", alpha=0.6, label="LOW -> MEDIUM (20 swipes)")
    ax.axvline(40, color="#a78bfa", linestyle="--", alpha=0.6, label="MEDIUM -> HIGH (40 swipes)")

    ax.set_xlabel("Training samples", color="white")
    ax.set_ylabel("Accuracy", color="white")
    ax.set_title("Intuition Model -- Learning Curve", color="white", fontsize=14)
    ax.tick_params(colors="white")
    ax.legend(facecolor=DARK_BG, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#444")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print("  Saved: outputs/learning_curve.png")


def get_preference_profile() -> str:
    artifact = _load_artifact()
    if not artifact:
        return "No model trained yet. Run `python main.py train` first."

    coefs  = artifact["clf"].coef_[0]
    ranked = sorted(zip(FEATURES, coefs), key=lambda t: abs(t[1]), reverse=True)

    top3 = ranked[:3]
    negs = [(s, c) for s, c in ranked if c < -0.3]

    def label(f: str) -> str:
        return f.replace("delta_", "").replace("_", " ").upper()

    opening = (
        f"You overweight {label(top3[0][0])} (+{top3[0][1]:.1f}x) and "
        f"{label(top3[1][0])} (+{top3[1][1]:.1f}x) relative to raw fantasy value."
        if top3[0][1] > 0.5
        else f"Your top signals are {label(top3[0][0])} and {label(top3[1][0])}."
    )
    mid = f" {label(top3[2][0])} is your 3rd strongest signal (+{top3[2][1]:.1f}x)." if len(top3) >= 3 else ""
    neg = f" You underweight {label(negs[0][0])} ({negs[0][1]:.1f}x)." if negs else ""

    conn = get_conn()
    try:
        early      = conn.execute("SELECT position FROM draft_history WHERE round <= 3").fetchall()
        guard_frac = sum(1 for r in early if r["position"] in ("PG", "SG", "G")) / max(len(early), 1)
        guard_note = " Based on draft history, you lean toward guards in early rounds." if guard_frac > 0.4 else ""
    except Exception:
        guard_note = ""
    finally:
        conn.close()

    return opening + mid + neg + guard_note


def predict_preference(player_a_stats: dict, player_b_stats: dict) -> str:
    artifact = _load_artifact()
    if not artifact:
        raise RuntimeError("No trained model found. Run `python main.py train`.")
    X = artifact["scaler"].transform([_delta(player_a_stats, player_b_stats)])
    return "A" if artifact["clf"].predict(X)[0] == 1 else "B"


def get_model_confidence() -> str:
    from db import DB_PATH
    if not DB_PATH.exists():
        return "LOW"
    conn = get_conn()
    try:
        row   = conn.execute(
            "SELECT swipe_count FROM model_versions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        count = row["swipe_count"] if row else 0
    except Exception:
        count = 0
    finally:
        conn.close()

    if count < 20:
        return "LOW"
    if count < 40:
        return "MEDIUM"
    return "HIGH"
