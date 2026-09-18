from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"

RANDOM_STATE = 42


def main():

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

    print(f"Training rows: {len(X_train)}")
    print(f"Number of predictors: {X_train.shape[1]}")
    print(f"Number of numeric features: {len(numeric_columns)}")
    print(f"Number of categorical features: {len(categorical_columns)}")

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

    # Baseline Gradient Boosting model
    gradient_boosting = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", gradient_boosting),
        ]
    )

    # Cross-validation
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

    print("\nGradient Boosting Baseline")
    print("=" * 40)

    cv_results = cross_validate(
        model_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    print("\n5-Fold Cross-Validation Results:")

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]

        print(
            f"{metric.upper():10s}: "
            f"{scores.mean():.4f} "
            f"(+/- {scores.std():.4f})"
        )


if __name__ == "__main__":
    main()