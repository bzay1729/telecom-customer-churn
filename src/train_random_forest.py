from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"


RANDOM_STATE = 42


def main():
    # Load Training Data
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

    # Identify features type
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.to_list()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.to_list()

    print(f"Training rows:: {len(X_train)}")
    print(f"Number of predictor: {X_train.shape[1]}")
    print(f"Number of numeric features: {len(numeric_columns)}")
    print(f"Number of categoric features: {len(categorical_columns)}")


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
                    handle_unknown="ignore"
                ),
                categorical_columns,
            ),
        ]
    )

    # Random Forest Baseline

    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1        
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", random_forest),
        ]
    )

    # Cross validation
    cv = StratifiedKFold(
        n_splits=5, 
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    print("\n Random Forest Baseline")
    print("=" * 40)

    cv_results = cross_validate(
        model_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    print("\n 5-Fold Cross Validation Results")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]

        print(
            f"{metric.upper():10s}:"
            f"{scores.mean():.4f}"
            f"(+-{scores.std():.4f})"
        )

if __name__ == "__main__":
    main()