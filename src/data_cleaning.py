from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
CLEAN_DATA_PATH  = Path("data/preprocessed/telco_customer_churn_cleaned.csv")

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """"Cleaning telecom churn dataset for analysis and modeling."""

    cleaned_data = data.copy()

    # Removing spaces from column charges before converting to numeric
    cleaned_data['TotalCharges'] = cleaned_data["TotalCharges"].str.strip()

    # Blank TotalCharges values belongs to customer with zero tenure.
    # We use 0.0 because new customer with zero tenure has no charges yet.
    cleaned_data["TotalCharges"] = pd.to_numeric(
        cleaned_data["TotalCharges"],
        errors="coerce"
    ).fillna(0.0)

    return cleaned_data

def main() -> None:

    raw_data = pd.read_csv(RAW_DATA_PATH)
    cleaned_data = clean_data(raw_data)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(CLEAN_DATA_PATH, index=False)

    print("Cleaned dataset shape:", cleaned_data.shape)
    print("Total Charges Data type:", cleaned_data["TotalCharges"].dtype)
    print("Missing Total Charges values:", cleaned_data["TotalCharges"].isna().sum())
    print("\Customer with zero tenure and zero total chaeges:")

    print(
        cleaned_data.loc[
            (cleaned_data["tenure"]==0) & (cleaned_data["TotalCharges"] == 0.0),
            ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"],

        ]
    )


if __name__ == "__main__":
    main()
