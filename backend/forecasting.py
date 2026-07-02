import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


def forecast_currency(df):

    # Convert month column to datetime
    df["month"] = pd.to_datetime(df["month"])
    df.set_index("month", inplace=True)

    # Train/Test Split
    train = df[df.index < "2024-07-01"]
    test = df[df.index >= "2024-07-01"]

    y_train = train["tzs_circulation_bn"]
    y_test = test["tzs_circulation_bn"]

    features = [
        "cpi_index",
        "gdp_growth_pct",
        "festive_month",
        "mobile_txn_volume_mn",
        "mobile_txn_value_bn"
    ]

    X_train = train[features]
    X_test = test[features]

    # Build SARIMAX model
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=(1, 1, 1),
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    # Train model
    model_fit = model.fit(disp=False)

    # Save model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(
        model_fit,
        "../models/sarimax_currency_model.pkl"
    )

    # Forecast
    forecast = model_fit.get_forecast(
        steps=len(test),
        exog=X_test
    )

    predictions = forecast.predicted_mean

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.plot(
        y_train.index,
        y_train,
        label="Training Data"
    )

    plt.plot(
        y_test.index,
        y_test,
        label="Actual Data"
    )

    plt.plot(
        predictions.index,
        predictions,
        label="Forecast"
    )

    plt.title("Currency Demand Forecast")
    plt.xlabel("Month")
    plt.ylabel("TZS Circulation (Billion)")
    plt.legend()
    plt.grid(True)

    os.makedirs("../results", exist_ok=True)

    plt.savefig("../results/forecast_plot.png")
    plt.close()

    # Save forecast
    forecast_df = pd.DataFrame({
        "Forecast": predictions
    })

    forecast_df.to_csv(
        "../results/forecast_results.csv",
        index=True
    )

    # Evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "Forecast": predictions.tolist()
    }