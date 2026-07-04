# Tanzania Currency Demand Forecasting System

A production-ready Streamlit application for forecasting monthly currency demand in Tanzania with a SARIMAX time-series model.

## What the App Does

- Upload a monthly CSV dataset or enter data manually.
- Validate the dataset before modeling.
- Forecast 1, 3, or 6 months using SARIMAX.
- Display MAE and RMSE.
- Show forecast and historical data tables.
- Plot actual vs forecast values and forecast trends.
- Download the forecast table as CSV.
- Download the forecast plot as PNG.

## Required CSV Format

Your CSV must contain one row per month with no missing months.

Required columns:

```text
month,tzs_circulation_bn,cpi_index,gdp_growth_pct,festive_month,mobile_txn_volume_mn,mobile_txn_value_bn
```

Column notes:

- `month`: monthly date such as `2024-07` or `2024-07-01`
- `tzs_circulation_bn`: currency circulation in billion TZS
- `cpi_index`: CPI index
- `gdp_growth_pct`: GDP growth percentage
- `festive_month`: `1` for festive months, otherwise `0`
- `mobile_txn_volume_mn`: mobile transaction volume in millions
- `mobile_txn_value_bn`: mobile transaction value in billion TZS

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Select **New app**.
4. Choose this repository and branch.
5. Set the main file path to:

```text
app.py
```

6. Deploy the app.

Streamlit Cloud will install packages from `requirements.txt` automatically. No FastAPI server or backend URL configuration is required.

## Model

The application preserves the original SARIMAX forecasting approach:

- Target: `tzs_circulation_bn`
- Exogenous variables: CPI index, GDP growth, festive month flag, mobile transaction volume, and mobile transaction value
- SARIMAX order: `(1, 1, 1)`
- Seasonal order: `(0, 0, 0, 0)`

The selected forecast horizon is evaluated against the latest available months in the uploaded or manually entered dataset.
