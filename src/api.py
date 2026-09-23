from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_churn


app = FastAPI(
    title="Telecom Customer Churn API",
    description=(
        "Predict telecom customer churn risk using the "
        "final tuned Gradient Boosting model."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------

class CustomerInput(BaseModel):

    gender: Literal["Female", "Male"]

    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]

    Dependents: Literal["Yes", "No"]

    tenure: int = Field(
        ge=0,
        description="Number of months the customer has stayed",
    )

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service",
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    MonthlyCharges: float = Field(
        ge=0,
    )

    TotalCharges: float = Field(
        ge=0,
    )


# ---------------------------------------------------------
# Response schema
# ---------------------------------------------------------

class PredictionResponse(BaseModel):

    churn_probability: float
    decision_threshold: float
    prediction: Literal[
        "Churn",
        "No Churn",
    ]


# ---------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Telecom Customer Churn Prediction API",
        "status": "running",
    }


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerInput):

    customer_data = customer.model_dump()

    result = predict_churn(
        customer_data
    )

    return result