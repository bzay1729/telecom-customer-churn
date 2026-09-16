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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data/preprocessed/telco_customer_churn_cleaned.csv"


data = pd.read_csv(DATA_PATH)


X = data.drop(columns=["Churn", "customerID"])
y = data["Churn"].map({"Yes": 1, "No": 0})
X = pd.get_dummies(X, drop_first=True)


# Train-test split with stratification to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling faeture before before applying regularization
scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Define an array for alpha
alphas = np.logspace(-3, 3, 100)

# Ridge Regresssion
ridge_model = RidgeCV(alphas=alphas, cv=5)
ridge_model.fit(X_trained_scaled, y_train)

# Lasso Regression
lasso_model = LassoCV(alphas=alphas, cv=5, random_state=42 )
lasso_model.fit(X_trained_scaled, y_train)

# Evaluate and compare models

print(f"Optimal Ridge Alpha: {ridge_model.alpha_:.4f}")
print(f"Optimal Lasso Alpha:  {lasso_model.alpha_:.4f}\n")

print(f"Ridge Train R² Score: {ridge_model.score(X_trained_scaled, y_train):.4f}")
print(f"Ridge Test R² Score:  {ridge_model.score(X_test_scaled, y_test):.4f}\n")

print(f"Lasso Train R² Score: {lasso_model.score(X_trained_scaled, y_train):.4f}")
print(f"Lasso Test R² Score:  {lasso_model.score(X_test_scaled, y_test):.4f}")