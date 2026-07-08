"""
Statistical model: regression-based player ranking and diagnostic plots.
Replaces stat_ranker.R — same four outputs, pure Python.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import CURRENT_SEASON, OUTPUT_DIR, get_conn

FEATURES = ["pts", "fg3m", "fgm", "fga", "ftm", "fta", "reb", "ast", "stl", "blk", "tov"]

DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
TEXT_CLR  = "white"
GRID_CLR  = "#2a2a2a"

POS_COLORS = {"PG": "#f97316", "SG": "#4ade80", "SF": "#60a5fa",
              "PF": "#a78bfa", "C":  "#f43f5e"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT p.player_id, p.name, p.team, p.position, ps.season, ps.gp,
               ps.pts, ps.fg3m, ps.fgm, ps.fga, ps.ftm, ps.fta,
               ps.reb, ps.ast, ps.stl, ps.blk, ps.tov,
               fs.fantasy_ppg, fs.consistency_score
        FROM player_stats ps
        JOIN players p        ON p.player_id = ps.player_id
        JOIN fantasy_scores fs ON fs.player_id = ps.player_id
                               AND fs.season = ps.season
        WHERE ps.gp >= 10
        """,
        conn,
    )
    conn.close()
    return df


# ── Models ─────────────────────────────────────────────────────────────────────

def _fit_regression(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def run_predictive_model(df: pd.DataFrame) -> dict:
    """Train on 2022-24, predict 2024-25. Returns metrics and test predictions."""
    train = df[df["season"].isin(["2022-23", "2023-24"])].dropna(subset=FEATURES + ["fantasy_ppg"])
    test  = df[df["season"] == CURRENT_SEASON].dropna(subset=FEATURES + ["fantasy_ppg"])

    X_train = train[FEATURES].values
    y_train = train["fantasy_ppg"].values
    X_test  = test[FEATURES].values
    y_test  = test["fantasy_ppg"].values

    model  = _fit_regression(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_train, model.predict(X_train)))

    coef_table = pd.DataFrame({
        "feature":     FEATURES,
        "coefficient": model.coef_,
    }).sort_values("coefficient", ascending=False)

    print(f"\n=== Predictive Regression (train: 2022-24  →  test: {CURRENT_SEASON}) ===")
    print(f"  Train R²:  {r2:.4f}")
    print(f"  Test RMSE: {rmse:.3f}")
    print(coef_table.to_string(index=False))

    test = test.copy()
    test["predicted_ppg"] = y_pred
    return {"model": model, "rmse": rmse, "r2": r2, "test_df": test}


def run_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """Fit on full dataset. Coefficients show which stats drive fantasy value most."""
    full = df.dropna(subset=FEATURES + ["fantasy_ppg"])
    model = _fit_regression(full[FEATURES].values, full["fantasy_ppg"].values)

    table = pd.DataFrame({
        "feature":     FEATURES,
        "coefficient": model.coef_,
        "abs_coef":    np.abs(model.coef_),
    }).sort_values("abs_coef", ascending=False).drop(columns="abs_coef")

    print("\n=== Feature Importance (full dataset) ===")
    print(table.to_string(index=False))
    return table


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _dark_fig(w: float, h: float):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_color(GRID_CLR)
    return fig, ax


def _save(fig, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: outputs/{name}")


# ── Plot 1: Player tier scatter ────────────────────────────────────────────────

def plot_tier_chart(df: pd.DataFrame) -> None:
    data = (
        df[(df["season"] == CURRENT_SEASON) & df["consistency_score"].notna() & (df["gp"] >= 15)]
        .sort_values("fantasy_ppg", ascending=False)
        .reset_index(drop=True)
    )
    if data.empty:
        print("  Tier chart skipped — no current-season data with consistency scores.")
        return

    med_x = data["consistency_score"].median()
    med_y = data["fantasy_ppg"].median()
    top40 = set(data.head(40)["name"])

    fig, ax = _dark_fig(15, 10)

    for pos, grp in data.groupby("position"):
        color = POS_COLORS.get(pos, "#888")
        ax.scatter(grp["consistency_score"], grp["fantasy_ppg"],
                   color=color, s=40, alpha=0.75, label=pos, zorder=3)

    # Labels for top-40 — stagger vertically to reduce collisions
    for _, row in data[data["name"].isin(top40)].iterrows():
        ax.annotate(
            row["name"],
            (row["consistency_score"], row["fantasy_ppg"]),
            fontsize=6.5, color="#ddd",
            xytext=(4, 3), textcoords="offset points",
        )

    ax.axvline(med_x, linestyle="--", color="white", alpha=0.35, linewidth=0.8)
    ax.axhline(med_y, linestyle="--", color="white", alpha=0.35, linewidth=0.8)

    x_max = data["consistency_score"].max()
    y_max = data["fantasy_ppg"].max()
    y_min = data["fantasy_ppg"].min()

    for label, xpos, ypos, color in [
        ("Safe Stars",          med_x * 0.25, y_max * 0.96, "#4ade80"),
        ("Boom or Bust",        x_max * 0.82, y_max * 0.96, "#f97316"),
        ("Reliable Bench",      med_x * 0.25, y_min * 1.10, "#60a5fa"),
        ("Skip",                x_max * 0.82, y_min * 1.10, "#f43f5e"),
    ]:
        ax.text(xpos, ypos, label, color=color, fontsize=9, fontweight="bold")

    ax.set_xlabel("Consistency Score (std dev — lower = more consistent)", color=TEXT_CLR)
    ax.set_ylabel("Fantasy PPG", color=TEXT_CLR)
    ax.set_title(f"Player Tier Chart — {CURRENT_SEASON}", color=TEXT_CLR, fontsize=14)
    ax.legend(facecolor=DARK_BG, labelcolor=TEXT_CLR, title="Position",
              title_fontsize=8, framealpha=0.8)

    _save(fig, "player_tiers.png")


# ── Plot 2: Correlation heatmap ────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    cols  = FEATURES + ["fantasy_ppg"]
    corr  = df[cols].dropna().corr()
    n     = len(cols)

    fig, ax = _dark_fig(12, 10)
    cmap    = plt.get_cmap("RdBu_r")
    im      = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(cols, rotation=45, ha="right", color=TEXT_CLR, fontsize=9)
    ax.set_yticklabels(cols, color=TEXT_CLR, fontsize=9)

    for i in range(n):
        for j in range(n):
            val  = corr.values[i, j]
            text = ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                           fontsize=7.5, color="white" if abs(val) > 0.4 else "#aaa")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04).ax.yaxis.set_tick_params(color=TEXT_CLR)
    ax.set_title("Stat Correlation Heatmap", color=TEXT_CLR, fontsize=14)

    _save(fig, "stat_correlation_heatmap.png")


# ── Plot 3: Positional boxplot ─────────────────────────────────────────────────

def plot_positional_boxplot(df: pd.DataFrame) -> None:
    positions = ["PG", "SG", "SF", "PF", "C"]
    data = (
        df[(df["season"] == CURRENT_SEASON) & (df["gp"] >= 10) & df["position"].isin(positions)]
        .dropna(subset=["fantasy_ppg"])
    )
    if data.empty:
        print("  Positional boxplot skipped — no current-season data.")
        return

    grouped = [data[data["position"] == pos]["fantasy_ppg"].values for pos in positions]

    fig, ax = _dark_fig(12, 7)
    bp = ax.boxplot(grouped, patch_artist=True, medianprops={"color": "white", "linewidth": 2})

    for patch, pos in zip(bp["boxes"], positions):
        patch.set_facecolor(POS_COLORS.get(pos, "#888"))
        patch.set_alpha(0.75)
    for element in ["whiskers", "caps", "fliers"]:
        for item in bp[element]:
            item.set_color("#aaa")

    ax.set_xticks(range(1, len(positions) + 1))
    ax.set_xticklabels(positions, color=TEXT_CLR)
    ax.set_ylabel("Fantasy PPG", color=TEXT_CLR)
    ax.set_title(f"Fantasy PPG by Position — {CURRENT_SEASON}", color=TEXT_CLR, fontsize=14)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.6)

    _save(fig, "positional_value_boxplot.png")


# ── Plot 4: Predicted vs actual ────────────────────────────────────────────────

def plot_predicted_vs_actual(test_df: pd.DataFrame, rmse: float) -> None:
    data = test_df.dropna(subset=["fantasy_ppg", "predicted_ppg"])

    fig, ax = _dark_fig(11, 8)

    for pos, grp in data.groupby("position"):
        color = POS_COLORS.get(pos, "#888")
        ax.scatter(grp["predicted_ppg"], grp["fantasy_ppg"],
                   color=color, s=45, alpha=0.7, label=pos, zorder=3)

    lo = min(data["predicted_ppg"].min(), data["fantasy_ppg"].min()) - 2
    hi = max(data["predicted_ppg"].max(), data["fantasy_ppg"].max()) + 2
    ax.plot([lo, hi], [lo, hi], "--", color="white", alpha=0.4, linewidth=1)

    ax.text(0.05, 0.93, f"RMSE = {rmse:.2f}", transform=ax.transAxes,
            color="#facc15", fontsize=13, fontweight="bold")

    ax.set_xlabel("Predicted Fantasy PPG", color=TEXT_CLR)
    ax.set_ylabel("Actual Fantasy PPG",    color=TEXT_CLR)
    ax.set_title(f"Predicted vs Actual — {CURRENT_SEASON} Test Set", color=TEXT_CLR, fontsize=14)
    ax.legend(facecolor=DARK_BG, labelcolor=TEXT_CLR, title="Position",
              title_fontsize=8, framealpha=0.8)

    _save(fig, "predicted_vs_actual.png")


# ── Public entry point ─────────────────────────────────────────────────────────

def run_analysis() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading data from DB...")
    df = load_data()

    if df.empty:
        print("No data found. Run `python main.py ingest` first.")
        return

    print(f"Loaded {len(df)} player-season records.")

    pred_results = run_predictive_model(df)
    run_feature_importance(df)

    print("\nGenerating plots...")
    plot_tier_chart(df)
    plot_correlation_heatmap(df)
    plot_positional_boxplot(df)
    plot_predicted_vs_actual(pred_results["test_df"], pred_results["rmse"])

    print("\n=== Stat model analysis complete ===")


if __name__ == "__main__":
    run_analysis()
