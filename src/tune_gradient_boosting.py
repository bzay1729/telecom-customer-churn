from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"

RANDOM_STATE = 42

def main():
    # Loading Train Data
    train_data = pd.read_csv(TRAIN_PATH)

    X_train = train_data.drop(
        columns=["customerID", "Churn"]
    )

    y_train = train_data["Churn"].map(
        {
            "No":0,
            "Yes":1,
        }
    )

    # Numeric and Categoric column
    numeric_column = X_train.select_dtypes(
        include="number"
    ).columns.to_list()
    categorical_cloumn = X_train.select_dtypes(
        exclude="number"
    ).columns.to_list()

    # Preprocessing
    preprocessing = ColumnTransformer(
        transformers=[(
            "numeric",
            "passthrough",
            numeric_column,
        ),(
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_cloumn
        ),
        ]
    )

    model = GradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(
        steps = [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )

    param_distribution = {
        "model__n_estimators": [50, 100, 150, 200, 300],
        "model__learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
        "model__max_depth": [1, 2, 3, 4],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__subsample": [0.7, 0.85, 1.0],
    }

    cv = StratifiedKFold(
        n_splits=5, 
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distribution,
        n_iter=40,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Gradient Boosting Randomized Search")
    print("=" * 40)

    search.fit(X_train, y_train)
    print("\n Best parameters:")
    for paramater, value in search.best_params_.items():
        print(f"{paramater}: {value}")
    print(
        f"\n Best Cross-validation ROC-AUC: "
        f"{search.best_score_:.4f}"
    )


if __name__ == "__main__":
    main()