# Telecom Customer Churn Prediction

An end-to-end machine learning project for predicting telecom customer churn using structured customer, service, and billing data.

The project is being developed as a complete machine learning workflow, including data validation, preprocessing, model training, evaluation, model comparison, API development, containerization, and deployment.

## Business Problem

Customer churn occurs when a customer stops using a company's service.

For a telecom provider, identifying customers who are likely to churn can help retention teams focus their efforts on higher-risk customers before they leave.

The goal of this project is to build a machine learning system that estimates customer churn risk from customer demographics, subscribed services, contract information, and billing behavior.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

* **Customers:** 7,043
* **Original columns:** 21
* **Target:** `Churn`
* **Prediction type:** Binary classification

The target contains two classes:

* `No` — customer did not churn
* `Yes` — customer churned

After removing the customer identifier and target column, the model uses **19 predictors**.

### Feature Types

**Numerical features:**

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`

**Categorical features:**

* `gender`
* `Partner`
* `Dependents`
* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `OnlineBackup`
* `DeviceProtection`
* `TechSupport`
* `StreamingTV`
* `StreamingMovies`
* `Contract`
* `PaperlessBilling`
* `PaymentMethod`

## Data Preparation

The preprocessing workflow currently includes:

* Dataset validation
* Data type inspection
* Missing-value inspection
* Conversion of `TotalCharges` to numeric
* Handling customers with zero tenure
* Removal of `customerID` from model features
* Stratified train/test split
* Numerical feature scaling
* Categorical feature encoding

The dataset is split using an **80/20 stratified train/test split** with `random_state=42`.

| Dataset      |  Rows | Churn Rate |
| ------------ | ----: | ---------: |
| Full Dataset | 7,043 |     26.54% |
| Training Set | 5,634 |     26.54% |
| Test Set     | 1,409 |     26.54% |

Stratification preserves approximately the same churn distribution in both datasets.

The final test set is kept separate during model development and is reserved for final evaluation.

## Preprocessing Pipeline

Preprocessing is implemented using a scikit-learn `ColumnTransformer` and `Pipeline`.

### Numerical Features

Numerical variables are transformed using:

`StandardScaler`

### Categorical Features

Categorical variables are transformed using:

`OneHotEncoder(handle_unknown="ignore")`

Keeping preprocessing inside the machine learning pipeline helps prevent data leakage because transformations are fitted only on the training data.

## Baseline Model

The first baseline model is **Logistic Regression**.

The model is evaluated using **5-fold stratified cross-validation** on the training dataset.

### Baseline Cross-Validation Results

| Metric    | Mean Score | Standard Deviation |
| --------- | ---------: | -----------------: |
| Accuracy  | **0.8021** |           ± 0.0118 |
| Precision | **0.6529** |           ± 0.0262 |
| Recall    | **0.5431** |           ± 0.0411 |
| F1 Score  | **0.5923** |           ± 0.0299 |
| ROC-AUC   | **0.8462** |           ± 0.0126 |

### Initial Observations

The Logistic Regression baseline achieves approximately **80.2% cross-validation accuracy** with a **ROC-AUC of 0.8462**.

The model shows useful ability to distinguish between customers with lower and higher churn risk. However, recall is approximately **54.3%**, meaning there is still room to improve detection of customers who actually churn.

These results will serve as the benchmark for evaluating future models and modeling strategies.

## Why PCA Is Not Used Initially

Principal Component Analysis is not currently required.

The project contains only 19 predictors, many of which are categorical. Keeping the original features also makes the model easier to interpret.

If PCA is evaluated later, it will be included inside the training pipeline so that it is fitted only on training data and does not introduce data leakage.

### Class-Weighted Logistic Regression

To investigate the class imbalance, Logistic Regression was also evaluated with
`class_weight="balanced"` using the same 5-fold stratified cross-validation setup.

| Metric | Baseline | Class-Weighted | Random Forest | Gradient Boosting |
|---|---:|---:|---:|--:|
| Accuracy | 0.8021 | 0.7485 | 0.7861 | 0.8033 |
| Precision | 0.6529 | 0.5169 | 0.6262  | 0.6619 |
| Recall | 0.5431 | 0.8013 | 0.4823 | 0.5298 |
| F1 Score | 0.5923 | 0.6283 | 0.5446 | 0.5883 |
| ROC-AUC | 0.8462 | 0.8460 | 0.8201 | 0.8481 |

Class weighting substantially improved recall from 54.31% to 80.13% and increased the F1 score, while reducing accuracy and precision. ROC-AUC remained almost unchanged, indicating that the model's overall ranking ability was similar.

### Gradient Boosting Hyperparameter Search

Gradient Boosting was tuned using `RandomizedSearchCV` with 40 parameter
combinations and 5-fold stratified cross-validation.

The best configuration achieved a development cross-validation ROC-AUC of
**0.8512**, compared with **0.8481** for the untuned Gradient Boosting baseline.

Best parameters:

- `learning_rate = 0.20`
- `n_estimators = 100`
- `max_depth = 1`
- `min_samples_split = 2`
- `min_samples_leaf = 2`
- `subsample = 0.85`

The final test dataset remains excluded from hyperparameter selection.

## Project Workflow

```text
Raw Dataset
     ↓
Data Validation
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Stratified Train/Test Split
     ↓
Feature Preprocessing
     ↓
Baseline Logistic Regression
     ↓
Cross-Validation
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Classification Threshold Analysis
     ↓
Final Test Evaluation
     ↓
Model Serialization
     ↓
FastAPI Prediction API
     ↓
Streamlit Interface
     ↓
Docker
     ↓
Testing and CI/CD
     ↓
Deployment
```

## Current Project Status

* [x] Dataset inspection
* [x] Data cleaning
* [x] Exploratory data analysis
* [x] Stratified train/test split
* [x] Feature preprocessing pipeline
* [x] Logistic Regression baseline
* [x] 5-fold cross-validation
* [ ] Class imbalance experiments
* [ ] Additional model comparison
* [ ] Hyperparameter tuning
* [ ] Classification threshold analysis
* [ ] Final test-set evaluation
* [ ] Model persistence
* [ ] FastAPI prediction service
* [ ] Streamlit application
* [ ] Automated tests
* [ ] Docker containerization
* [ ] GitHub Actions CI/CD
* [ ] Cloud deployment

## Technology Stack

### Machine Learning and Data

* Python
* Pandas
* NumPy
* scikit-learn

### API and Application

* FastAPI
* Streamlit

### Engineering and Deployment

* Git
* GitHub
* Docker
* GitHub Actions

Additional infrastructure will be added as the project progresses.

## Key Engineering Principles

This project follows several practices intended to make the machine learning workflow reproducible and production-oriented:

* Maintain a completely separate final test set
* Prevent preprocessing leakage
* Use reproducible random seeds
* Evaluate models using stratified cross-validation
* Compare multiple evaluation metrics instead of relying only on accuracy
* Keep preprocessing and modeling inside reusable pipelines
* Track experiments before selecting the final model
* Separate model development from final evaluation

## Next Steps

The next stage is to investigate whether class imbalance treatment improves churn detection.

The current Logistic Regression baseline will be compared with alternative approaches using the same cross-validation strategy.

Future work will include:

* Class-weighted Logistic Regression
* Additional classification models
* Model comparison
* Hyperparameter tuning
* Threshold optimization
* Final evaluation on the untouched test set
* Model serialization
* REST API development with FastAPI
* Streamlit interface
* Docker containerization
* Automated testing and CI/CD
* Cloud deployment
