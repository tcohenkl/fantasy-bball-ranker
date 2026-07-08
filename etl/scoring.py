SCORING_WEIGHTS = {
    "pts": 1,
    "fg3m": 1,
    "fgm": 2,
    "fga": -1,
    "ftm": 1,
    "fta": -1,
    "reb": 1,
    "ast": 2,
    "stl": 4,
    "blk": 4,
    "tov": -2,
}


def compute_fantasy_score(
    pts: float,
    fg3m: float,
    fgm: float,
    fga: float,
    ftm: float,
    fta: float,
    reb: float,
    ast: float,
    stl: float,
    blk: float,
    tov: float,
) -> float:
    """
    ESPN H2H Points scoring:
    PTS=1, 3PM=1, FGM=2, FGA=-1, FTM=1, FTA=-1,
    REB=1, AST=2, STL=4, BLK=4, TOV=-2
    """
    return (
        pts * SCORING_WEIGHTS["pts"]
        + fg3m * SCORING_WEIGHTS["fg3m"]
        + fgm * SCORING_WEIGHTS["fgm"]
        + fga * SCORING_WEIGHTS["fga"]
        + ftm * SCORING_WEIGHTS["ftm"]
        + fta * SCORING_WEIGHTS["fta"]
        + reb * SCORING_WEIGHTS["reb"]
        + ast * SCORING_WEIGHTS["ast"]
        + stl * SCORING_WEIGHTS["stl"]
        + blk * SCORING_WEIGHTS["blk"]
        + tov * SCORING_WEIGHTS["tov"]
    )
