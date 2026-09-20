from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/churn_model.joblib"


# ---------------------------------------------------------
# Load saved model artifact
# ---------------------------------------------------------

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
threshold = artifact["threshold"]
expected_features = artifact["features"]


def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn risk for one telecom customer.

    Parameters
    ----------
    customer_data : dict
        Customer information containing the same features
        used during model training.

    Returns
    -------
    dict
        Churn probability, prediction, and decision threshold.
    """

    # ---------------------------------------------------------
    # Validate input features
    # ---------------------------------------------------------

    missing_features = [
        feature
        for feature in expected_features
        if feature not in customer_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep features in training order
    customer_df = pd.DataFrame(
        [
            {
                feature: customer_data[feature]
                for feature in expected_features
            }
        ]
    )

    # ---------------------------------------------------------
    # Predict probability
    # ---------------------------------------------------------

    churn_probability = model.predict_proba(
        customer_df
    )[0, 1]

    # Apply our selected threshold
    churn_prediction = (
        churn_probability >= threshold
    )

    prediction_label = (
        "Churn"
        if churn_prediction
        else "No Churn"
    )

    return {
        "churn_probability": round(
            float(churn_probability),
            4,
        ),
        "decision_threshold": threshold,
        "prediction": prediction_label,
    }


# ---------------------------------------------------------
# Simple local test
# ---------------------------------------------------------

if __name__ == "__main__":

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 448.75,
    }

    result = predict_churn(
        sample_customer
    )

    print("Customer Churn Prediction")
    print("=" * 35)

    print(
        f"Probability : "
        f"{result['churn_probability']:.2%}"
    )

    print(
        f"Threshold   : "
        f"{result['decision_threshold']:.2f}"
    )

    print(
        f"Prediction  : "
        f"{result['prediction']}"
    )