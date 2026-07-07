# NBA Game Outcome Prediction

Machine learning models that predict whether the home team wins an NBA game, using leakage-free rolling-window features built from 18 seasons of historical data.

**Course:** Foundations of Machine Learning, Master of Data Analytics, Western University
**Authors:** Waleed Chaudhry, Mian Abdullah Zahid, Bilal Naeemuddin

## Overview

Most game-prediction models rely on static, season-aggregate statistics. This project engineers dynamic features — rolling 10-game windows of team form, cumulative player "star power," and contextual factors like home-court advantage — from five relational Kaggle datasets covering NBA games from 2004–2022, with strict chronological splitting to prevent look-ahead bias.

## Results (4,007-game holdout set)

| Model | Accuracy | ROC-AUC |
|---|---|---|
| XGBoost | **71.3%** | **0.791** |
| Logistic Regression | 70.7% | 0.786 |
| Random Forest | 69.6% | 0.779 |

**Case study — 2018-19 Toronto Raptors (championship season):** Random Forest accuracy jumps to **82.0%** with 0.919 ROC-AUC, showing that consistent elite teams produce a much cleaner statistical signal than league-wide averages.

## Feature Engineering

- **Team-level:** rolling 10-game win %, average points for/against, rebounds, assists
- **Player-level:** per-player scoring averages, star-player max ("star power"), roster scoring depth
- **Contextual:** season win % standing, conference, arena capacity, home/away flag
- **Leakage prevention:** all features computed strictly from games *before* the predicted game; chronological train/test split

Feature-importance analysis confirmed win percentage and cumulative points as dominant predictors, with the custom star-power metric adding a "talent-gap" signal invisible to team-level stats alone.

## Repository Contents

- `nba_game_prediction.ipynb` — data merging, feature engineering, model training, and evaluation
- `report.pdf` — full write-up (IEEE format)

## Data

Uses the [NBA Games dataset on Kaggle](https://www.kaggle.com/datasets/nathanlauga/nba-games) (`games.csv`, `games_details.csv`, `players.csv`, `ranking.csv`, `teams.csv`). Download and place the CSVs in the project directory before running.

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
jupyter notebook nba_game_prediction.ipynb
```
