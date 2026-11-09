import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "customer_churn_model.pkl"
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
CONTRACT_CATEGORIES = ["Month-to-Month", "One Year", "Two Year"]
INTERNET_CATEGORIES = ["DSL", "Fiber Optic", "None"]


@asynccontextmanager
async def lifespan(api: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Train the model before starting the API."
        )
    api.state.model_artifact = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


class Customer(BaseModel):
    tenure_months: int = Field(ge=1)
    monthly_charges: float = Field(gt=0)
    total_charges: float | None = Field(default=None, ge=0)
    support_calls: int = Field(ge=0)
    contract_type: Literal["Month-to-Month", "One Year", "Two Year"]
    internet_service: Literal["DSL", "Fiber Optic", "None"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer, request: Request) -> dict[str, object]:
    artifact = request.app.state.model_artifact
    row = customer.model_dump()
    if row["total_charges"] is None:
        row["total_charges"] = row["tenure_months"] * row["monthly_charges"]

    frame = pd.DataFrame([row])
    frame["contract_type"] = pd.Categorical(
        frame["contract_type"], categories=CONTRACT_CATEGORIES
    )
    frame["internet_service"] = pd.Categorical(
        frame["internet_service"], categories=INTERNET_CATEGORIES
    )
    encoded = pd.get_dummies(
        frame,
        columns=["contract_type", "internet_service"],
        drop_first=True,
        dtype=int,
    )
    encoded = encoded.reindex(columns=artifact["feature_columns"], fill_value=0)

    model = artifact["model"]
    prediction = int(model.predict(encoded)[0])
    probability = float(model.predict_proba(encoded)[0][1])
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "status": "High Risk" if probability > 0.5 else "Low Risk",
    }
