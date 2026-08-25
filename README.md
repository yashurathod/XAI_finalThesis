# XAI for Household Electricity Demand Forecasting

MSc thesis project. Short-term forecasting of household electricity demand,
combining a stacked tree ensemble with a Shapley-based account of *where in
time* the model draws its information from.

## Data

Two datasets, deliberately unlike each other:

| | UCI household power | Tetouan City |
|---|---|---|
| Location | near Paris, France | Tetouan, Morocco |
| Period | Dec 2006 to Nov 2010 | full year 2017 |
| Resolution | 1 minute | 10 minutes |
| Target | global active power (kW) | Zone 1 network load (kW) |
| Weather covariates | none | 5 variables |

Both are downloaded by the notebook on first run, so no data is committed here.

## Method

**Features.** 71 candidate columns: cyclical calendar encodings, a Fourier
seasonal basis, lags from 1 to 1440 minutes, rolling mean, standard deviation,
minimum and maximum, first differences, exponentially weighted means,
memory-gap terms measured against the previous day, and a Holt-Winters
decomposition refitted on a rolling 18-month window with a 30-day step. A
feature-family ablation removes the Holt-Winters family from the prediction
path, leaving 68 features; the decomposition is retained as an analysis layer.

**Model.** XGBoost, LightGBM and CatBoost, each tuned by Bayesian optimisation,
plus a random forest for algorithmic diversity. Out-of-fold predictions are
produced by forward chaining, so the combiner is never fitted to predictions
built from future data. The combiner is a convex blend with non-negative
weights summing to one, chosen on the validation split from among the
individual learners, their equal-weight average, and blends fitted to
out-of-fold and to validation predictions.

**Calibration.** A bias correction fitted on the validation split alone. Any
non-zero mean residual survives temporal aggregation and becomes a floor on the
RMSE as the averaging window grows, which is why unbiased linear models can
otherwise overtake a pointwise more accurate ensemble at daily granularity.

**Leakage control.** Every feature is computed on a series shifted by at least
one step; every split is chronological; the Holt-Winters refit only ever sees
past observations; scalers are fitted on the training split alone; and the test
split is touched exactly once.

## Results

Test split, one-minute resolution, 311,073 observations:

| Model | MAE | RMSE | R2 | MAPE | MASE |
|---|---|---|---|---|---|
| **Proposed ensemble** | **0.0711** | **0.1996** | **0.9451** | **7.70** | **0.0879** |
| XGBoost | 0.0743 | 0.2025 | 0.9435 | 8.23 | 0.0918 |
| Random forest | 0.0771 | 0.2056 | 0.9418 | 8.97 | 0.0953 |
| LASSO | 0.0843 | 0.2140 | 0.9369 | 10.44 | 0.1042 |
| Linear regression | 0.0845 | 0.2140 | 0.9369 | 10.45 | 0.1044 |
| Persistence | 0.0694 | 0.2170 | 0.9351 | 7.53 | 0.0858 |
| k-nearest neighbours | 0.2255 | 0.3434 | 0.8376 | 32.78 | 0.2786 |
| ARIMA (hourly) | 0.6426 | 0.7427 | -0.1145 | 124.55 | n/a |

Persistence attains the lowest MAE. The load is unchanged across most minutes,
so repeating the last reading is close to optimal under an absolute-error
criterion; it loses on the squared-error measures the trained models minimise.
ARIMA is fitted to hourly aggregates and is not directly comparable.

Diebold-Mariano tests reject equal predictive accuracy against every reference
model at p < 0.001, using a Newey-West long-run variance with a Bartlett kernel
and the Harvey-Leybourne-Newbold small-sample correction.

Accuracy by native resolution, and across datasets:

| Resolution | R2 | RMSE | MAPE |
|---|---|---|---|
| 1-minute | 0.9451 | 0.1996 | 7.70% |
| 5-minute | 0.7496 | 0.4146 | 39.23% |
| Hourly | 0.9777 | 0.1049 | 7.88% |
| Tetouan Zone 1, 10-minute | 0.9895 | 620.72 | 1.39% |

## Explainability

Because the combiner is linear and convex, the weighted sum of the per-learner
Shapley values is the *exact* decomposition of the ensemble forecast, not an
approximation. The notebook verifies this numerically: base value plus summed
attributions reproduces the forecast for every explained instance to within
single-precision tolerance.

**Temporal SHAP.** Feature-level attributions are projected onto a lookback
axis by mapping each feature to a dominant lag. 67% of the attributional mass
sits on lag-resolved features, and the lag profile is sharply concentrated at
the immediate past: fitting an exponential decay gives a memory horizon of
**0.35 minutes**, and the normalised entropy of the profile is 0.32 on a scale
where 0 is a single horizon and 1 is uniform.

That finding predicts a behavioural one. If accuracy rests almost entirely on
the most recent reading, then feeding predictions back in place of observations
should erode skill within a few steps, and the recursive evaluation in the
notebook shows exactly that. The one-minute model is therefore scoped to
one-step dispatch, and the hourly model is the right instrument for next-hour
planning.

**Cross-checks.** SHAP is compared against LIME and against permutation
importance. Near-constant binary indicators are excluded from the headline
agreement figures: the discretised perturbation scheme LIME uses assigns them
large weights, whereas both other measures attribute them almost nothing, which
is a property of the surrogate rather than of the model.

## Repository layout

```
XAI_Electricity_demand_forecast_final.ipynb   full pipeline, top to bottom
requirements.txt                              pinned minimum versions
```

The notebook writes its figures and tables to `results/`, which is not tracked.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook XAI_Electricity_demand_forecast_final.ipynb
```

Run the cells in order. Expect a long runtime: the hyperparameter search, the
multi-resolution sweep and the SHAP computation dominate. A CUDA device is used
for CatBoost tuning if one is present.

Reproducibility: the random seed is fixed at 42 and the final cell reports the
versions of every library, queried from the running kernel rather than recorded
by hand. Results in this repository were produced on Python 3.10.9 with NumPy
1.23.5, pandas 1.5.3, scikit-learn 1.5.2, XGBoost 1.7.6, LightGBM 4.6.0,
CatBoost 1.2.10 and SHAP 0.49.1.

## Datasets

- Hebrail, G. and Berard, A. Individual Household Electric Power Consumption.
  UCI Machine Learning Repository.
- Salam, A. and El Hibaoui, A. (2018). Comparison of machine learning
  algorithms for the power consumption prediction. IEEE IRSEC.
