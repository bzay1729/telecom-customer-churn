from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data/preprocessed/telco_customer_churn_cleaned.csv"
FIGURES_PATH = PROJECT_ROOT / "reports/figures"


def main() -> None:
    """Exploratory Data Analysis (EDA) for the Telco Customer Churn dataset."""

    # Loading the cleaned dataset.
    data = pd.read_csv(DATA_PATH)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    # Using 1 for churn and 0 for no churn.
    data["ChurnFlag"] = data["Churn"].map({"Yes": 1, "No": 0})
# -------------------------------------------------------------------------
    # Comparing customer counts and churn rates for each contract type.
    contract_summary = (data.groupby("Contract").agg(
        customers = ("ChurnFlag", "size"),
        churned_customers = ("ChurnFlag", "sum"),
        churn_rate = ("ChurnFlag", "mean")
    )
    .reindex(["Month-to-month", "One year", "Two year"])
    )

    contract_summary["churn_rate"] *= 100

    print("\n Churn by contract type:")
    print(contract_summary.round(2))

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(
        contract_summary.index,
        contract_summary["churn_rate"],
        color=["#E76F51", "#457B9D", "#2A9D8F"],
    )

    ax.set_title("Customer Churn rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Customer who churned (%)")


    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "churn_by_contract.png", dpi=300)

# -------------------------------------------------------------------------
    # Grouping customers how long they have been with the company.
    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0–12 months", "13–24 months", "25–36 months", "37–48 months", "49–60 months", "61–72 months"],
    )

    tenure_summary = (data.groupby("TenureGroup", observed=True).agg(
        customers= ("ChurnFlag", "size"),
        churned_customers= ("ChurnFlag", "sum"),
        churn_rate= ("ChurnFlag", "mean")
    ))

    tenure_summary["churn_rate"] *= 100
    print("\n Churn by tenure group:")
    print(tenure_summary.round(2))

    # Plotting churn rate by tenure group.
    fig, ax = plt.subplots(figsize=(8, 6))
    bars =ax.bar(
        tenure_summary.index.astype(str),
        tenure_summary["churn_rate"],
        color=["#457B9D"]
    )

    ax.bar_label(bars, fmt="%.1f%%", label_type="edge")
    ax.set_title("Customer Churn rate by Tenure Group")
    ax.set_xlabel(" Time with the company (months)")
    ax.set_ylabel("Customer who churned (%)")

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "churn_by_tenure.png", dpi=150)

# -------------------------------------------------------------------------

    # Comparing monthly charges for customer who stayed and those who left.
    charges_summary = data.groupby("Churn")["MonthlyCharges"].agg(
        ["count", "mean", "std", "min", "max", "median"]
        )

    print("\n Monthly Charges by Churn Status:")
    print(charges_summary.round(2))

    stayed_charges = data.loc[data["Churn"] == "No", "MonthlyCharges"]
    churned_charges = data.loc[data["Churn"]== "Yes", "MonthlyCharges"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([stayed_charges, churned_charges])
    ax.set_xticks([1, 2], ["Stayed", "Churned"])
    ax.set_xticklabels(["Stayed", "Churned"])
    ax.set_title("Monthly Charges by Churn Status")
    ax.set_xlabel("Customer outcome")
    ax.set_ylabel("Monthly Charges ($)")

    fig.tight_layout()
    fig.savefig(FIGURES_PATH / "monthly_charges_by_churn.png", dpi=150)



if __name__ == "__main__":
    main()


