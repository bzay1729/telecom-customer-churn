from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"

RANDOM_STATE = 42


def build_pipeline(numeric_columns, categorical_columns, class_weight=None):

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
             )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric", 
                numeric_transformer, 
                numeric_columns,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_columns
            ),

        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

def evaluate_model(name, model, X_train, y_train, cv):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",

    }

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1, 

    )

    print(f"\n {name}")
    print("-" * len(name))

    summary = {}

    for metric in scoring:
        scores = results[f"test_{metric}"]

        mean_score = scores.mean()
        std_score = scores.std()

        summary[metric] = mean_score

        print(
            f"{metric.upper():10s}:"
            f"{mean_score:.4f} (+/- {std_score:.4f})"
        )
    return summary

def main():
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

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    baseline_model = build_pipeline(
        numeric_columns,
        categorical_columns,
        class_weight=None,
    )

    balanced_model = build_pipeline(
        numeric_columns,
        categorical_columns,
        class_weight="balanced",
    )

    print("Logistic Regression Model Comparison")
    print("=" * 40)

    baseline_results = evaluate_model(
        "Baseline Logistic Regression",
        baseline_model,
        X_train,
        y_train,
        cv,
    )

    balanced_results = evaluate_model(
        "Class-Weighted Logistic Regression",
        balanced_model,
        X_train,
        y_train,
        cv,
    )

    print("\nComparison")
    print("-" * 60)

    print(
        f"{'Metric':<12}"
        f"{'Baseline':>12}"
        f"{'Balanced':>12}"
        f"{'Change':>12}"
    )

    for metric in baseline_results:
        baseline = baseline_results[metric]
        balanced = balanced_results[metric]
        change = balanced - baseline

        print(
            f"{metric.upper():<12}"
            f"{baseline:>12.4f}"
            f"{balanced:>12.4f}"
            f"{change:>+12.4f}"
        )

if __name__ == "__main__":
    main()