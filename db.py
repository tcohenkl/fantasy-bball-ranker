"""Shared database utilities — single source of truth for paths and schema."""

import sqlite3
from pathlib import Path

BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "data" / "fantasy.db"
OUTPUT_DIR = BASE_DIR / "outputs"

CURRENT_SEASON = "2025-26"
ALL_SEASONS    = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS players (
            player_id      INTEGER PRIMARY KEY,
            name           TEXT,
            team           TEXT,
            position       TEXT,
            espn_id        INTEGER,
            injury_status  TEXT DEFAULT 'ACTIVE'
        );

        CREATE TABLE IF NOT EXISTS player_stats (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id         INTEGER,
            season            TEXT,
            gp                INTEGER,
            min               REAL,
            pts               REAL,
            fg3m              REAL,
            fg3a              REAL,
            fgm               REAL,
            fga               REAL,
            ftm               REAL,
            fta               REAL,
            reb               REAL,
            ast               REAL,
            stl               REAL,
            blk               REAL,
            tov               REAL,
            is_rookie         INTEGER DEFAULT 0,
            team_win_pct_prev REAL    DEFAULT 0.5,
            gp_prev_season    INTEGER DEFAULT 0,
            team_abbr         TEXT    DEFAULT '',
            team_wins         INTEGER DEFAULT 0,
            team_losses       INTEGER DEFAULT 0,
            team_win_pct      REAL    DEFAULT 0.5,
            team_seed         INTEGER DEFAULT 0,
            team_conf         TEXT    DEFAULT '',
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            UNIQUE(player_id, season)
        );

        CREATE TABLE IF NOT EXISTS fantasy_scores (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id         INTEGER,
            season            TEXT,
            fantasy_ppg       REAL,
            fantasy_total     REAL,
            consistency_score REAL,
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            UNIQUE(player_id, season)
        );

        CREATE TABLE IF NOT EXISTS draft_history (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            season                  TEXT,
            round                   INTEGER,
            pick_number             INTEGER,
            player_name             TEXT,
            position                TEXT,
            fantasy_ppg_that_season REAL
        );

        CREATE TABLE IF NOT EXISTS swipe_history (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp          TEXT,
            winner_id          INTEGER,
            loser_id           INTEGER,
            winner_fantasy_ppg REAL,
            loser_fantasy_ppg  REAL,
            upset_flag         INTEGER,
            FOREIGN KEY (winner_id) REFERENCES players(player_id),
            FOREIGN KEY (loser_id)  REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            trained_at     TEXT,
            swipe_count    INTEGER,
            train_accuracy REAL,
            test_accuracy  REAL,
            top_features   TEXT,
            model_path     TEXT
        );
        """
    )

    # Migrate existing DBs — SQLite ignores these if column already exists
    _add_column_if_missing(conn, "players",      "espn_id        INTEGER")
    _add_column_if_missing(conn, "players",      "injury_status  TEXT DEFAULT 'ACTIVE'")
    _add_column_if_missing(conn, "player_stats", "fg3a              REAL    DEFAULT 0")
    _add_column_if_missing(conn, "player_stats", "is_rookie         INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "player_stats", "team_win_pct_prev REAL    DEFAULT 0.5")
    _add_column_if_missing(conn, "player_stats", "gp_prev_season    INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "player_stats", "team_abbr         TEXT    DEFAULT ''")
    _add_column_if_missing(conn, "player_stats", "team_wins         INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "player_stats", "team_losses       INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "player_stats", "team_win_pct      REAL    DEFAULT 0.5")
    _add_column_if_missing(conn, "player_stats", "team_seed         INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "player_stats", "team_conf         TEXT    DEFAULT ''")

    conn.commit()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, col_def: str) -> None:
    col_name = col_def.split()[0]
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if col_name not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def}")
