from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"

RANDOM_STATE = 42


def main():

    # ---------------------------------------------------------
    # Load training data
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Feature types
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
    # Gradient Boosting
    # ---------------------------------------------------------

    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE,
        max_depth=1,
        min_samples_split=2,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # ---------------------------------------------------------
    # Focused search around RandomizedSearchCV winner
    # ---------------------------------------------------------

    param_grid = {
        "model__learning_rate": [0.15, 0.20, 0.25],
        "model__n_estimators": [75, 100, 125],
        "model__min_samples_leaf": [1, 2],
        "model__subsample": [0.80, 0.85],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Focused Gradient Boosting Grid Search")
    print("=" * 45)

    grid_search.fit(X_train, y_train)

    print("\nBest Parameters:")

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        f"\nBest Cross-Validation ROC-AUC: "
        f"{grid_search.best_score_:.4f}"
    )


if __name__ == "__main__":
    main()