# XAI for Household Electricity Demand Forecasting

MSc thesis project. Short-term forecasting of household electricity demand,
with the model's behaviour explained rather than left as a black box.

## Data

UCI Individual Household Electric Power Consumption: about 2.07 million
one-minute readings from a single dwelling near Paris, December 2006 to
November 2010. The forecast target is global active power, in kW.

The notebook downloads the archive on first run, so no data file needs to be
committed here.

## Pipeline so far

1. Parse the raw meter file and quantify missing readings.
2. Exploratory analysis: correlation structure, daily demand profile, and an
   Augmented Dickey-Fuller test on hourly aggregates.
3. Impute interior gaps by time-weighted interpolation and store the result in
   single precision.
4. Feature engineering:
   - cyclical calendar encodings and a Fourier seasonal basis;
   - lags at 1 to 1440 minutes;
   - rolling mean, standard deviation, minimum and maximum;
   - first differences and exponentially weighted means;
   - an hour-of-day interaction term;
   - a Holt-Winters decomposition, refitted on a rolling window.
5. Chronological 70/15/15 split, with standardisation fitted on the training
   split alone.

Every feature is computed on a series shifted by at least one step, so no row
is conditioned on its own value or on anything later.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook XAI_Electricity_demand_forecast_final.ipynb
```

## Status

Feature pipeline complete. Modelling and evaluation next.
