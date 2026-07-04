from __future__ import annotations

import pandas as pd
import streamlit as st

from forecasting import (
    FEATURE_COLUMNS,
    HORIZON_OPTIONS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    DatasetValidationError,
    dataframe_to_csv_bytes,
    forecast_currency,
    load_dataset,
    validate_dataset,
)


st.set_page_config(
    page_title="Tanzania Currency Demand Forecasting System",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


def main() -> None:
    apply_theme()
    initialize_state()

    with st.sidebar:
        st.title("Currency Forecast")
        selected_page = st.radio(
            "Navigation",
            ["Dashboard", "Data", "Forecast Results", "About"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Tanzania Currency Demand Forecasting System")

    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "Data":
        render_data_page()
    elif selected_page == "Forecast Results":
        render_results_page()
    else:
        render_about_page()


def initialize_state() -> None:
    st.session_state.setdefault("dataset", None)
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("data_source", "Upload CSV")


def render_dashboard() -> None:
    st.title("Tanzania Currency Demand Forecasting System")
    st.write(
        "A production-ready SARIMAX dashboard for forecasting monthly Tanzanian currency "
        "circulation using inflation, GDP growth, festive season, and mobile transaction indicators."
    )

    render_data_controls()

    dataset = st.session_state.get("dataset")
    if dataset is None:
        st.info("Upload a CSV or enter data manually to begin forecasting.")
        return

    try:
        validated = validate_dataset(dataset)
    except DatasetValidationError as exc:
        st.error(str(exc))
        return

    st.success(f"Dataset ready: {len(validated):,} monthly records from {validated.index.min():%b %Y} to {validated.index.max():%b %Y}.")

    horizon_label = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()), index=1)
    horizon = HORIZON_OPTIONS[horizon_label]

    if st.button("Forecast", type="primary", use_container_width=True):
        run_forecast(dataset, horizon)

    result = st.session_state.get("result")
    if result is not None:
        render_metrics(result)
        render_charts(result)
        render_forecast_table(result)
        render_downloads(result)
        render_historical_table(result)


def render_data_controls() -> None:
    data_source = st.segmented_control(
        "Data input",
        ["Upload CSV", "Manual input"],
        default=st.session_state.data_source,
    )
    st.session_state.data_source = data_source

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        if uploaded_file is not None:
            try:
                st.session_state.dataset = load_dataset(uploaded_file)
                st.session_state.result = None
            except DatasetValidationError as exc:
                st.error(str(exc))
    else:
        manual_df = st.data_editor(
            starter_dataset(),
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "month": st.column_config.TextColumn("month", help="YYYY-MM format"),
                TARGET_COLUMN: st.column_config.NumberColumn("tzs_circulation_bn", min_value=0.01),
                "festive_month": st.column_config.NumberColumn("festive_month", min_value=0, max_value=1, step=1),
            },
            key="manual_data_editor",
        )
        if st.button("Use manual data", use_container_width=True):
            st.session_state.dataset = manual_df
            st.session_state.result = None


def render_data_page() -> None:
    st.title("Data")
    dataset = st.session_state.get("dataset")
    if dataset is None:
        st.info("No dataset has been loaded yet.")
        return

    try:
        validated = validate_dataset(dataset)
    except DatasetValidationError as exc:
        st.error(str(exc))
        return

    st.subheader("Historical Data Table")
    display_df = validated.reset_index()
    display_df["month"] = display_df["month"].dt.strftime("%Y-%m")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download historical CSV",
        data=dataframe_to_csv_bytes(display_df),
        file_name="historical_currency_data.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_results_page() -> None:
    st.title("Forecast Results")
    result = st.session_state.get("result")
    if result is None:
        st.info("Run a forecast from the Dashboard page to view results.")
        return

    render_metrics(result)
    render_forecast_table(result)
    render_downloads(result)
    render_charts(result)
    render_historical_table(result)


def render_about_page() -> None:
    st.title("Project Description")
    st.write(
        "This application forecasts monthly currency demand in Tanzania using a SARIMAX "
        "time-series model with external economic and payment-system indicators. It is designed "
        "for direct Streamlit deployment without a separate API server."
    )
    st.subheader("Required CSV Columns")
    st.code(", ".join(REQUIRED_COLUMNS), language="text")


def run_forecast(dataset: pd.DataFrame, horizon: int) -> None:
    try:
        with st.spinner("Running SARIMAX forecast..."):
            st.session_state.result = forecast_currency(dataset, horizon)
        st.success("Forecast completed successfully.")
    except DatasetValidationError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(
            "The forecast could not be completed. Please review the dataset and try again. "
            f"Details: {exc}"
        )


def render_metrics(result) -> None:
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{result.mae:,.3f}")
    col2.metric("RMSE", f"{result.rmse:,.3f}")


def render_charts(result) -> None:
    st.subheader("Actual vs Forecast Chart")
    st.image(result.forecast_plot, use_container_width=True)

    st.subheader("Forecast Trend Chart")
    st.image(result.trend_plot, use_container_width=True)


def render_forecast_table(result) -> None:
    st.subheader("Forecast Table")
    st.dataframe(result.forecast, use_container_width=True, hide_index=True)


def render_historical_table(result) -> None:
    st.subheader("Historical Data Table")
    st.dataframe(result.historical, use_container_width=True, hide_index=True)


def render_downloads(result) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download forecast CSV",
            data=dataframe_to_csv_bytes(result.forecast),
            file_name="currency_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download forecast plot",
            data=result.forecast_plot.getvalue(),
            file_name="currency_forecast_plot.png",
            mime="image/png",
            use_container_width=True,
        )


def starter_dataset() -> pd.DataFrame:
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    trend = pd.Series(range(24), dtype=float)
    data = pd.DataFrame(
        {
            "month": months.strftime("%Y-%m"),
            "tzs_circulation_bn": (14.8 + trend * 0.22).round(2),
            "cpi_index": (130 + trend * 0.45).round(1),
            "gdp_growth_pct": [6.9, 6.8, 7.0, 6.7, 7.1, 6.9] * 4,
            "festive_month": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1] * 2,
            "mobile_txn_volume_mn": (260 + trend * 3.1).round(1),
            "mobile_txn_value_bn": (7.2 + trend * 0.18).round(2),
        }
    )
    return data


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: #0f172a;
        }
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 16px;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
