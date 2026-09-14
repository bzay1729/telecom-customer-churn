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
data.info()
print("Churn distribution:")
print(data["Churn"].value_counts())
print("Percentage of churned customers:")
print((data["Churn"].value_counts(normalize=True) * 100).round(2))

# Inspecting any hidden blank values inside TotalCharges column.
print("----------------------------------")
blank_total_charges = data["TotalCharges"].astype(str).str.strip().eq("")

print("\n Blank TotalCharges values:")
print(blank_total_charges.sum())

print("\n Customers with blank TotalCharges:")
print(
    data.loc[
        blank_total_charges, 
        ["customerID", "TotalCharges", "MonthlyCharges", "tenure", "Churn"],
    ]
)

print("\n Duplicate rows:")
print(data.duplicated().sum())

print("\n Duplicate customer IDs:")
print(data["customerID"].duplicated().sum())