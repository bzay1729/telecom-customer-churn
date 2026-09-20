from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.35


def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load training data
    train_data = pd.read_csv(TRAIN_PATH)

    X_train = train_data.drop(
        columns=["customerID", "Churn"]
    )

    y_train = train_data["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # Identify columns
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_columns,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
        ]
    )

    # Final tuned model
    model = GradientBoostingClassifier(
        learning_rate=0.15,
        n_estimators=125,
        max_depth=1,
        min_samples_split=2,
        min_samples_leaf=2,
        subsample=0.85,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Train on all training data
    pipeline.fit(
        X_train,
        y_train,
    )

    # Save model and metadata together
    model_artifact = {
        "model": pipeline,
        "threshold": DECISION_THRESHOLD,
        "features": X_train.columns.tolist(),
    }

    joblib.dump(
        model_artifact,
        MODEL_PATH,
    )

    print("Final model trained successfully.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Decision threshold: {DECISION_THRESHOLD}")
    print(f"Number of features: {len(X_train.columns)}")


if __name__ == "__main__":
    main()