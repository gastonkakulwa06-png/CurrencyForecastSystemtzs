from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
from io import StringIO
import os
import logging

from forecasting import forecast_currency

app = FastAPI()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


@app.get("/")
def home():
    return {
        "message": "Welcome to the Tanzania Currency Demand Forecasting API"
    }


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        contents = await file.read()

        logger.info(f"Dataset uploaded: {file.filename}")

        # Save uploaded file
        os.makedirs("../uploads", exist_ok=True)

        with open(f"../uploads/{file.filename}", "wb") as f:
            f.write(contents)

        # Load into DataFrame
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        logger.info("Dataset loaded successfully.")

        # Forecast
        logger.info("Forecast started.")

        results = forecast_currency(df)

        logger.info("Forecast completed successfully.")

        return {
            "status": "Success",
            "message": "Forecast completed successfully.",
            "results": results
        }

    except Exception as e:
        logger.error(str(e))

        return {
            "status": "Error",
            "message": str(e)
        }


@app.get("/download/csv")
def download_csv():
    return FileResponse(
        "../results/forecast_results.csv",
        media_type="text/csv",
        filename="forecast_results.csv"
    )
    