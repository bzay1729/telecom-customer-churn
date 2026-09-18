from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Training data created during the train/test split
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"

def main():
    # Loading training data 
    train_data = pd.read_csv(TRAIN_PATH)

    # Separate predictor and target
    X_train = train_data.drop(columns=["customerID", "Churn"])
    y_train = train_data["Churn"].map(
        {
            "No": 0,
            "Yes" : 1
        }
    )

    # Automatically indentify numerical and categorical column
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    print(f"Training rows: {len(X_train)}")
    print(f"Number of predictors: {X_train.shape[1]} ")

    print("\n Numerical columns: ")
    for numeric in numeric_columns:
        print(f"- {numeric}")

    print("\n Categorical Columns: ")
    for categoric in categorical_columns:
        print(f"- {categoric}")

    print("\n Target Distribution:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True).round(4))


    # Preprocessing for numerical features
    numerical_transformer = Pipeline(
        steps=[(
            "scaler",
            StandardScaler()
        )
        ]
    )

    # Preprocessing for Categorical features
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Applies preprocessing to different column types
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numerical_transformer,
                numeric_columns
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_columns
            )

        ]
    )

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
    # First Baseline Model
    logistic_regression = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Combining preprocessing and modeling
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor",preprocessor),
            ("model", logistic_regression)
        ]
    )

    # Trains the entire pipeline
    model_pipeline.fit(X_train, y_train)
    print("\n Baseline Logistic Regression pipeline trained successfully.")

    # 5 fold stratified Cross-validation

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        model_pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1

    )

    print("\n 5-Fold Cross_validation Results:")

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]

        print(
            f"{metric.upper():10s}: "
            f"{scores.mean():.4f} "
            f"(+/- {scores.std():.4f})"
        )



if __name__ == "__main__":
    main()

    



