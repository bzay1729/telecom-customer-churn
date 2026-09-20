from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"
TEST_PATH = PROJECT_ROOT / "data/preprocessed/test.csv"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.35


def main():

    # ---------------------------------------------------------
    # Load train and final test data
    # ---------------------------------------------------------

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    # ---------------------------------------------------------
    # Separate features and target
    # ---------------------------------------------------------

    X_train = train_data.drop(
        columns=["customerID", "Churn"]
    )

    y_train = train_data["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    X_test = test_data.drop(
        columns=["customerID", "Churn"]
    )

    y_test = test_data["Churn"].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    # ---------------------------------------------------------
    # Identify feature types
    # ---------------------------------------------------------

    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Final tuned model
    # ---------------------------------------------------------

    final_model = GradientBoostingClassifier(
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
            ("model", final_model),
        ]
    )

    # ---------------------------------------------------------
    # Train on ALL development training data
    # ---------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Final test probabilities
    # ---------------------------------------------------------

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Apply threshold selected during development
    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    # ---------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    tn, fp, fn, tp = cm.ravel()

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print("FINAL MODEL EVALUATION")
    print("=" * 50)

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Decision threshold: {DECISION_THRESHOLD:.2f}")

    print("\nFinal Test Metrics")
    print("-" * 30)

    print(f"ACCURACY  : {accuracy:.4f}")
    print(f"PRECISION : {precision:.4f}")
    print(f"RECALL    : {recall:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"ROC_AUC   : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print("-" * 30)

    print(f"True Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    print("\nClassification Report")
    print("-" * 30)

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "No Churn",
                "Churn",
            ],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()