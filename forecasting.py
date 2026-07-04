from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


DATE_COLUMN = "month"
TARGET_COLUMN = "tzs_circulation_bn"
FEATURE_COLUMNS = [
    "cpi_index",
    "gdp_growth_pct",
    "festive_month",
    "mobile_txn_volume_mn",
    "mobile_txn_value_bn",
]
REQUIRED_COLUMNS = [DATE_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
HORIZON_OPTIONS = {
    "1 month": 1,
    "3 months": 3,
    "6 months": 6,
}


@dataclass(frozen=True)
class ForecastResult:
    mae: float
    rmse: float
    forecast: pd.DataFrame
    historical: pd.DataFrame
    fitted_values: pd.Series
    actual_test: pd.Series
    forecast_plot: BytesIO
    trend_plot: BytesIO


class DatasetValidationError(ValueError):
    """Raised when an uploaded dataset cannot be used for forecasting."""


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise DatasetValidationError("The uploaded dataset is empty.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        expected = ", ".join(REQUIRED_COLUMNS)
        raise DatasetValidationError(
            f"The dataset is missing required column(s): {missing}. Expected columns: {expected}."
        )

    cleaned = df.loc[:, REQUIRED_COLUMNS].copy()
    cleaned[DATE_COLUMN] = pd.to_datetime(cleaned[DATE_COLUMN], errors="coerce")
    if cleaned[DATE_COLUMN].isna().any():
        raise DatasetValidationError(
            "Some month values could not be read as dates. Use values such as 2024-07 or 2024-07-01."
        )

    for column in [TARGET_COLUMN, *FEATURE_COLUMNS]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid_columns = cleaned.columns[cleaned.isna().any()].tolist()
    if invalid_columns:
        invalid = ", ".join(invalid_columns)
        raise DatasetValidationError(
            f"The dataset contains blank or non-numeric values in: {invalid}."
        )

    if cleaned[DATE_COLUMN].duplicated().any():
        raise DatasetValidationError("The month column contains duplicate dates.")

    cleaned = cleaned.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    cleaned.index = cleaned.index.to_period("M").to_timestamp()

    inferred_frequency = pd.infer_freq(cleaned.index)
    if inferred_frequency not in {"MS", "M"}:
        raise DatasetValidationError(
            "The dataset must contain one row per month without gaps."
        )

    if len(cleaned) < 18:
        raise DatasetValidationError(
            "Please provide at least 18 monthly records so the model can train and validate reliably."
        )

    if not cleaned[TARGET_COLUMN].gt(0).all():
        raise DatasetValidationError("Currency circulation values must be greater than zero.")

    if not cleaned["festive_month"].isin([0, 1]).all():
        raise DatasetValidationError("The festive_month column must contain only 0 or 1.")

    return cleaned


def load_dataset(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("The CSV file could not be read. Please upload a UTF-8 CSV file.") from exc
    except Exception as exc:
        raise DatasetValidationError("The CSV file could not be loaded. Please check the file format.") from exc


def forecast_currency(df: pd.DataFrame, horizon: int) -> ForecastResult:
    cleaned = validate_dataset(df)

    train, test = _split_train_test(cleaned, horizon)

    y_train = train[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]
    x_train = train[FEATURE_COLUMNS]
    x_test = test[FEATURE_COLUMNS]

    model = SARIMAX(
        y_train,
        exog=x_train,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False, maxiter=200)

    forecast = model_fit.get_forecast(steps=len(test), exog=x_test)
    predictions = forecast.predicted_mean
    confidence_interval = forecast.conf_int()

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    fitted_values = model_fit.get_prediction(
        start=y_train.index[1],
        end=y_train.index[-1],
        exog=x_train.iloc[1:],
        dynamic=False,
    ).predicted_mean

    forecast_df = pd.DataFrame(
        {
            "month": predictions.index,
            "actual_tzs_circulation_bn": y_test.values,
            "forecast_tzs_circulation_bn": predictions.values,
            "lower_ci": confidence_interval.iloc[:, 0].values,
            "upper_ci": confidence_interval.iloc[:, 1].values,
        }
    )
    forecast_df["month"] = forecast_df["month"].dt.strftime("%Y-%m")

    historical_df = cleaned.reset_index().rename(columns={DATE_COLUMN: "month"})
    historical_df["month"] = historical_df["month"].dt.strftime("%Y-%m")

    forecast_plot = build_actual_vs_forecast_plot(y_train, y_test, predictions)
    trend_plot = build_forecast_trend_plot(predictions, confidence_interval)

    return ForecastResult(
        mae=round(float(mae), 3),
        rmse=round(rmse, 3),
        forecast=forecast_df,
        historical=historical_df,
        fitted_values=fitted_values,
        actual_test=y_test,
        forecast_plot=forecast_plot,
        trend_plot=trend_plot,
    )


def build_actual_vs_forecast_plot(
    y_train: pd.Series,
    y_test: pd.Series,
    predictions: pd.Series,
) -> BytesIO:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(y_train.index, y_train.values, label="Training data", color="#2563eb", linewidth=2)
    ax.plot(y_test.index, y_test.values, label="Actual data", color="#111827", linewidth=2)
    ax.plot(predictions.index, predictions.values, label="Forecast", color="#dc2626", linewidth=2)
    ax.set_title("Actual vs Forecast Currency Demand")
    ax.set_xlabel("Month")
    ax.set_ylabel("TZS circulation (billion)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    return _figure_to_png(fig)


def build_forecast_trend_plot(
    predictions: pd.Series,
    confidence_interval: pd.DataFrame,
) -> BytesIO:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(predictions.index, predictions.values, label="Forecast trend", color="#047857", linewidth=2.5)
    ax.fill_between(
        predictions.index,
        confidence_interval.iloc[:, 0].values,
        confidence_interval.iloc[:, 1].values,
        color="#86efac",
        alpha=0.35,
        label="Confidence interval",
    )
    ax.set_title("Forecast Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("TZS circulation (billion)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    return _figure_to_png(fig)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _split_train_test(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if horizon not in HORIZON_OPTIONS.values():
        valid = ", ".join(str(value) for value in HORIZON_OPTIONS.values())
        raise DatasetValidationError(f"Unsupported forecast horizon. Choose one of: {valid} months.")

    if len(df) <= horizon + 12:
        raise DatasetValidationError(
            f"The selected {horizon}-month horizon needs more historical records. Upload a longer monthly dataset."
        )

    split_index = len(df) - horizon
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def _figure_to_png(fig) -> BytesIO:
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer
