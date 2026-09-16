from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.datasets import make_regression

# Main project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# File Locations
DATA_PATH = PROJECT_ROOT / "data/preprocessed/telco_customer_churn_cleaned.csv"
TRAIN_PATH = PROJECT_ROOT / "data/preprocessed/train.csv"
TEST_PATH = PROJECT_ROOT / "data/preprocessed/test.csv"

RANDOM_STATE = 42

def main():
    # Loads clean dataset
    data = pd.read_csv(DATA_PATH)

    # Splits the data while keeping the same churn proportion
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data["Churn"]
    )

    # Saves the datasets
    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    # Checks the result
    print(f"Full dataset shape:  {data.shape}")
    print(f"Training dataset shape: {train_data.shape}")
    print(f"Test dataset shape: {test_data.shape}")

    print(" \n Churn percentage by dataset:")

    datasets = {
        "Full": data,
        "Train": train_data,
        "Test": test_data,
    }

    for name, dataset in datasets.items():
        churn_rate = (dataset["Churn"] == "Yes").mean() * 100
        print(f"{name}: {churn_rate:.2f}%")

if __name__ == "__main__":
    main()

