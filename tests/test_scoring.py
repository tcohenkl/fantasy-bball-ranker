import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from etl.scoring import compute_fantasy_score, SCORING_WEIGHTS


def test_three_pointer_made():
    # 3 pts scored + 1 FGM + 1 3PM + 1 FGA = 3(1) + 2(1) + 1(1) + (-1)(1) = 5
    assert compute_fantasy_score(pts=3, fg3m=1, fgm=1, fga=1, ftm=0, fta=0,
                                  reb=0, ast=0, stl=0, blk=0, tov=0) == 5.0


def test_two_pointer_made():
    # 2 pts + 1 FGM + 1 FGA = 2 + 2 - 1 = 3
    assert compute_fantasy_score(pts=2, fg3m=0, fgm=1, fga=1, ftm=0, fta=0,
                                  reb=0, ast=0, stl=0, blk=0, tov=0) == 3.0


def test_free_throw_made():
    assert compute_fantasy_score(pts=1, fg3m=0, fgm=0, fga=0, ftm=1, fta=1,
                                  reb=0, ast=0, stl=0, blk=0, tov=0) == 1.0


def test_steal():
    assert compute_fantasy_score(pts=0, fg3m=0, fgm=0, fga=0, ftm=0, fta=0,
                                  reb=0, ast=0, stl=1, blk=0, tov=0) == 4.0


def test_turnover():
    assert compute_fantasy_score(pts=0, fg3m=0, fgm=0, fga=0, ftm=0, fta=0,
                                  reb=0, ast=0, stl=0, blk=0, tov=1) == -2.0


def test_full_game_line():
    # 30 pts (8 FGM, 2 3PM, 2 FTM), 8 REB, 6 AST, 2 STL, 1 BLK, 3 TOV
    # PTS: 30, FGM: 16, 3PM: 2, FGA: -10, FTM: 2, FTA: -2
    # REB: 8, AST: 12, STL: 8, BLK: 4, TOV: -6
    # Total = 30+16+2-10+2-2+8+12+8+4-6 = 64
    score = compute_fantasy_score(pts=30, fg3m=2, fgm=8, fga=10, ftm=2, fta=2,
                                   reb=8, ast=6, stl=2, blk=1, tov=3)
    assert score == 64.0


def test_all_zeros():
    assert compute_fantasy_score(pts=0, fg3m=0, fgm=0, fga=0, ftm=0, fta=0,
                                  reb=0, ast=0, stl=0, blk=0, tov=0) == 0.0


def test_negative_total_possible():
    # Bad shooting night: 2 pts (1 FGM, 1 FGA) + 5 TOV = 2+2-1-10 = -7
    score = compute_fantasy_score(pts=2, fg3m=0, fgm=1, fga=1, ftm=0, fta=0,
                                   reb=0, ast=0, stl=0, blk=0, tov=5)
    assert score == -7.0


def test_block_value():
    assert compute_fantasy_score(pts=0, fg3m=0, fgm=0, fga=0, ftm=0, fta=0,
                                  reb=0, ast=0, stl=0, blk=1, tov=0) == 4.0


def test_assist_value():
    assert compute_fantasy_score(pts=0, fg3m=0, fgm=0, fga=0, ftm=0, fta=0,
                                  reb=0, ast=1, stl=0, blk=0, tov=0) == 2.0


def test_rebound_value():
    assert compute_fantasy_score(pts=0, fg3m=0, fgm=0, fga=0, ftm=0, fta=0,
                                  reb=1, ast=0, stl=0, blk=0, tov=0) == 1.0


def test_scoring_weights_dict():
    assert SCORING_WEIGHTS["stl"] == 4
    assert SCORING_WEIGHTS["blk"] == 4
    assert SCORING_WEIGHTS["tov"] == -2
    assert SCORING_WEIGHTS["ast"] == 2
    assert SCORING_WEIGHTS["fgm"] == 2
    assert SCORING_WEIGHTS["fga"] == -1


def test_get_model_confidence_valid():
    from models.intuition_model import get_model_confidence
    confidence = get_model_confidence()
    assert confidence in ("LOW", "MEDIUM", "HIGH")
