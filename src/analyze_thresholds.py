from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"

RANDOM_STATE = 42


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
    }


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

    # Feature types
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

    # Final tuned Gradient Boosting configuration
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

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ---------------------------------------------------------
    # Generate out-of-fold probabilities
    # ---------------------------------------------------------

    probabilities = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    auc = roc_auc_score(
        y_train,
        probabilities,
    )

    print("Tuned Gradient Boosting Threshold Analysis")
    print("=" * 50)

    print(
        f"\nOut-of-Fold ROC-AUC: {auc:.4f}"
    )

    # ---------------------------------------------------------
    # Show useful threshold range
    # ---------------------------------------------------------

    print("\nThreshold Comparison")
    print("-" * 67)

    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    thresholds = np.arange(
        0.20,
        0.71,
        0.05,
    )

    for threshold in thresholds:

        metrics = calculate_metrics(
            y_train,
            probabilities,
            threshold,
        )

        print(
            f"{metrics['threshold']:<12.2f}"
            f"{metrics['accuracy']:>12.4f}"
            f"{metrics['precision']:>12.4f}"
            f"{metrics['recall']:>12.4f}"
            f"{metrics['f1']:>12.4f}"
        )

    # ---------------------------------------------------------
    # Search for threshold with best F1
    # ---------------------------------------------------------

    fine_thresholds = np.arange(
        0.20,
        0.81,
        0.01,
    )

    results = []

    for threshold in fine_thresholds:
        results.append(
            calculate_metrics(
                y_train,
                probabilities,
                threshold,
            )
        )

    best_f1 = max(
        results,
        key=lambda result: result["f1"],
    )

    print("\nBest F1 Threshold")
    print("-" * 30)

    print(
        f"Threshold : {best_f1['threshold']:.2f}"
    )
    print(
        f"Accuracy  : {best_f1['accuracy']:.4f}"
    )
    print(
        f"Precision : {best_f1['precision']:.4f}"
    )
    print(
        f"Recall    : {best_f1['recall']:.4f}"
    )
    print(
        f"F1        : {best_f1['f1']:.4f}"
    )


if __name__ == "__main__":
    main()