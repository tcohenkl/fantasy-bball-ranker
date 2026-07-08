# Fantasy Basketball AI Ranker

**An LLM that ranks NBA players the way I would, trained on my own draft history and player preferences.**

---

## How It Works

Three layers work together to produce rankings that reflect my actual preferences — not a generic model's:

```
NBA Stats API
     │
     ▼
 ETL Pipeline (Python)
 SQLite Database
     │
     ├──────────────────────┐
     ▼                      ▼
Stat Model (R)       Intuition Model (sklearn)
Regression +         Trained on swipe history
Visualizations       + draft picks
     │                      │
     └──────────┬───────────┘
                ▼
         LLM Ranker (Claude)
         Top 20 + reasoning
         in user's voice
```

1. **ETL Pipeline** pulls three seasons of NBA per-game stats via `nba_api`, computes ESPN fantasy points per game for every player, and stores everything in SQLite.
2. **Stat Model (R)** runs a predictive regression trained on past seasons to project current-season fantasy value, produces four diagnostic plots, and quantifies which raw stats drive fantasy output under this scoring.
3. **Intuition Model (sklearn)** trains a logistic regression on my past draft picks and head-to-head player swipes to learn what I *actually* value — not what the stats say I should value. The Flask swipe UI makes this feel like a game.
4. **LLM Ranker (Claude)** receives the top 30 stat-model players plus my preference profile, and produces a top 20 ranking written in my voice — first-person reasoning, casual tone, like texting my league group chat.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | Python 3.11+, nba_api |
| Storage | SQLite |
| Statistical model | R 4.x (tidyverse, broom, ggplot2, RSQLite) |
| ML intuition model | Python, scikit-learn, joblib |
| LLM ranker | Python, Anthropic API (claude-sonnet-4-6) |
| Swipe UI | Python, Flask |
| Analysis notebook | Jupyter |
| Tests | pytest |

---

## ESPN H2H Points Scoring

```
PTS = +1    (every point scored)
3PM = +1    (three-pointer made bonus)
FGM = +2    (field goal made)
FGA = -1    (field goal attempt)
FTM = +1    (free throw made)
FTA = -1    (free throw attempt)
REB = +1    (rebound)
AST = +2    (assist)
STL = +4    (steal)
BLK = +4    (block)
TOV = -2    (turnover)
```

**Derived per-shot values:**
- Three-pointer made = **5 pts** (3+1+2-1)
- Two-pointer made = **3 pts** (2+2-1)
- Free throw made = **1 pt** (1+1-1)

This scoring heavily rewards efficiency, elite perimeter defenders, and playmakers — pure volume scorers are penalized by FGA and TOV.

---

## The Intuition Model

The core differentiator. Most fantasy tools optimize for a single objective (max fantasy PPG). This model learns what *I* optimize for.

**Training data:**
- **Swipe history** — each time I pick Player A over Player B in the swipe UI, the model records a delta vector: the difference in stats between winner and loser. Over time, consistent patterns emerge (e.g., I always choose the higher-STL player when STL differs by 1.5+).
- **Draft history** — players I drafted early are treated as implicit preferences over players I drafted later in the same draft. A round 1 pick is a strong signal (weight 1.0); a round 8 pick is a weak signal (weight 0.30).

**Feature vector:** `[Δpts, Δfg3m, Δast, Δstl, Δblk, Δreb, Δtov, Δfantasy_ppg, Δconsistency]`

**Output:** A preference profile string like:
> "You overweight STL (+2.3x) and BLK (+1.9x) relative to raw fantasy value. AST is your 3rd strongest signal (+1.4x). You underweight high-volume scorers with poor efficiency (-0.8x on FGA). Based on draft history, you lean toward guards in early rounds."

**Confidence levels:** LOW (<20 swipes) · MEDIUM (20-40) · HIGH (40+)

---

## Setup

**Prerequisites:**
- Python 3.11+
- R 4.x
- An Anthropic API key

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Install R packages:**
```bash
Rscript install_r_packages.R
```

**Set your API key:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

---

## How to Run

```bash
# 1. Pull NBA stats and populate the database (~10-20 min first run)
python main.py ingest

# 2. Launch the swipe UI and start building your preference model
python main.py swipe

# 3. Train the intuition model on your swipes + draft history
python main.py train

# 4. Run the R statistical analysis and generate plots
python main.py r-analysis

# 5. Generate your personalized weekly top 20
python main.py rankings

# 6. Check system status
python main.py status

# 7. Run tests
python main.py test
```

---

## Sample Output

```
---
Rank 1 | Nikola Jokic | DEN | C | Proj: 76.8 pts | Model rank: #1 (—)
He's the clear #1, no debate needed. Triple-double floor every night, elite efficiency, and 
the assists put him in a tier of his own under this scoring.
---
Rank 2 | Shai Gilgeous-Alexander | OKC | PG | Proj: 70.3 pts | Model rank: #2 (—)
SGA's STL numbers are what push him here — averaging 2+ steals a game at 4 pts each is 
a massive bonus that pure-scorer rankings miss.
---
Rank 3 | Alex Caruso | OKC | SG | Proj: 36.4 pts | Model rank: #31 (↑28) ⚠️ Big mover
This is all intuition model — his raw fantasy PPG doesn't justify this rank, but 2.4 STL 
and 0.7 BLK with clean shot selection is exactly my kind of player. The model learned 
I always pick the defensive specialist in toss-ups.
---
Rank 4 | Tyrese Haliburton | IND | PG | Proj: 57.2 pts | Model rank: #5 (↑1)
10+ assists a night at 2 pts each is just math. If he's healthy this is an easy top-5 pick.
---
Rank 5 | Giannis Antetokounmpo | MIL | PF | Proj: 66.4 pts | Model rank: #3 (↓2)
I'd normally have him higher but the FTA penalty is real — he shoots over 10 a game and 
misses enough that it chips into his total. STL/BLK make up for it, but barely at this cost.
---
```

---

## What I'd Add Next

- **Waiver wire recommender** — surface top available players each week based on preference profile
- **Trade analyzer** — evaluate proposed trades using stat model + intuition weighting
- **Opponent scouting** — analyze upcoming matchup opponent's roster weaknesses
- **Injury news scraping** — automate the injury context input for the LLM ranker so I don't have to type it manually
- **Multi-season preference drift** — track how my preferences change year over year as my league format evolves
