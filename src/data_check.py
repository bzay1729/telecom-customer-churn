from pathlib import Path
import pandas as pd

# Data Path
DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")

data = pd.read_csv(DATA_PATH)

# Data checking.
print("Dataset shape:", data.shape)
print("\n First 5 rows of the dataset:")
print(data.head())
print("\n Column names:")
print(data.columns)
print("Data types and missing values:")
print(data.info())
print("Churn distribution:")
print(data["Churn"].value_counts())
print("Percentage of churned customers:")
print((data["Churn"].value_counts(normalize=True) * 100).round(2))
