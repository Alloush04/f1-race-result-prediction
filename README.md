# F1 Race Result Prediction

Predicting where a Formula 1 driver will finish a race using only their qualifying session.

This is the code for my final year dissertation (BSc Computer Science, University of Reading, 2025-2026).

## The question

Grid position is already a strong predictor of race result. The question here is how much better a model can do using the rest of the qualifying session: the lap times set in Q1, Q2 and Q3, the gap to pole, and how far through qualifying the driver got.

## Data

- Source: the [FastF1](https://github.com/theOehrly/Fast-F1) library, qualifying and race results for every round from 2017 to 2025.
- Sprint weekends are skipped, because their qualifying format does not set the race grid in the same way.
- 3,377 driver-race entries collected, 3,327 after dropping rows with no Q1 time, grid position or finishing position.

The data itself is not committed. The data collection notebook rebuilds it from FastF1.

## Features

Six inputs, all known before the race starts:

| Feature | Meaning |
|---|---|
| `grid_position` | Qualifying position |
| `Q1_s`, `Q2_s`, `Q3_s` | Best lap time in each qualifying segment, in seconds (0 if the driver was eliminated before it) |
| `delta_to_pole_s` | Driver's best qualifying time minus the pole time for that race |
| `q_session_reached` | Furthest segment reached: 1, 2 or 3 |

Target: `finish_position`.

## Method

- **Time-based split.** Train on 2017-2023, test on 2024-2025, so the model is always scored on seasons it has never seen.
- **Baseline.** Predict that every driver finishes where they start.
- **Models compared.** Linear, Ridge and Lasso regression, decision tree, k-nearest neighbours, random forest, gradient boosting, a voting ensemble and a stacking ensemble, plus a FLAML AutoML search.
- **Tuning.** Grid search on gradient boosting using `TimeSeriesSplit`, so the cross-validation folds also respect time order.
- **Metrics.** Mean absolute error, RMSE and Spearman rank correlation between predicted and actual finishing order.
- **Explanation.** SHAP (TreeExplainer) and partial dependence plots on the best gradient boosting model.

## Results

The best model reached a Spearman correlation of 0.712 with actual finishing positions on the held-out seasons.

Running the modelling notebook writes the full comparison table and figures to `outputs/`: model comparison, predicted vs actual, SHAP importance and partial dependence.

## Repository layout

```
notebooks/
  01_data_collection.ipynb
  02_EDA.ipynb
  03_preprocessing.ipynb
  04_modelling.ipynb
data/       created locally, not committed
outputs/    figures and results table, plus requirements.txt
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r outputs/requirements.txt
jupyter notebook
```

Run data collection first, then preprocessing, then modelling. Data collection is rate limited by the F1 timing API, so the first run takes a while. Later runs use the local FastF1 cache.

## Tools

Python, pandas, scikit-learn, FLAML, SHAP, FastF1, matplotlib, seaborn.
