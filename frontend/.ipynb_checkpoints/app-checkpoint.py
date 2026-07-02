import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="Currency Demand Forecasting System",
    layout="wide"
)

st.title("🇹🇿 Tanzania Currency Demand Forecasting System")

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    st.success("Dataset uploaded successfully.")

    if st.button("Run Forecast"):

        files = {
            "file": uploaded_file
        }

        response = requests.post(
            "http://127.0.0.1:8000/upload",
            files=files
        )

        data = response.json()

        if data["status"] == "Success":

            st.success(data["message"])

            col1, col2 = st.columns(2)

            col1.metric("MAE", data["results"]["MAE"])
            col2.metric("RMSE", data["results"]["RMSE"])

            forecast = data["results"]["Forecast"]

            forecast_df = pd.DataFrame({
                "Forecast (Billion TZS)": forecast
            })

            st.subheader("Forecast Results")

            st.dataframe(forecast_df)

        else:

            st.error(data["message"])