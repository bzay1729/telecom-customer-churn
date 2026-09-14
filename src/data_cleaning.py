from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
CLEAN_DATA_PATH  = Path("data/preprocessed/telco_customer_churn_cleaned.csv")

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """"Cleaning telecom churn dataset for analysis and modeling."""

    cleaned_data = data.copy()
    charges = cleaned_data["TotalCharges"].str.strip()

    # Only use zero for blank charges when tenure is 0.
    new_customer_blank = charges.eq("") & cleaned_data["tenure"].eq(0)

    # Converting TotalCharges to numeric, coercing errors to NaN
    cleaned_data["TotalCharges"] = pd.to_numeric(charges, errors="raise")
    cleaned_data.loc[new_customer_blank, "TotalCharges"] = 0.0

    # Stop if any other misssing charges need investigation
    if cleaned_data["TotalCharges"].isna().any():
        raise ValueError(
            "Unexpected missing TotalCharges values need to be reviewed."
        )


    return cleaned_data

def main() -> None:

    raw_data = pd.read_csv(RAW_DATA_PATH)
    cleaned_data = clean_data(raw_data)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(CLEAN_DATA_PATH, index=False)

    print("Cleaned dataset shape:", cleaned_data.shape)
    print("Total Charges Data type:", cleaned_data["TotalCharges"].dtype)
    print("Missing Total Charges values:", cleaned_data["TotalCharges"].isna().sum())
    print("\Customer with zero tenure and zero total charges:")

    print(
        cleaned_data.loc[
            (cleaned_data["tenure"]==0) & (cleaned_data["TotalCharges"] == 0.0),
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"],

        ]
    )


if __name__ == "__main__":
    main()
