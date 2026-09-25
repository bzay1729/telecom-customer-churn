# Telecom Customer Churn Prediction

[![CI](https://github.com/bzay1729/telecom-customer-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/bzay1729/telecom-customer-churn/actions/workflows/ci.yml)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://telecom-customer-churn-analysis.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An end-to-end machine learning project for predicting telecom customer churn using customer demographics, service subscriptions, contract information, and billing behavior.

The project covers the complete machine learning lifecycle: data validation, preprocessing, model comparison, hyperparameter tuning, threshold optimization, final evaluation, model serialization, API development, automated testing, containerization, continuous integration, and cloud deployment.

## Live Demo

Try the deployed application:

**https://telecom-customer-churn-analysis.streamlit.app**

---

## Business Problem

Customer churn occurs when a customer stops using a company's service.

For a telecom provider, identifying customers who are likely to churn can help retention teams focus their efforts on higher-risk customers before they leave.

The goal of this project is to build a machine learning system that estimates customer churn risk and provides both:

- A churn probability
- A final churn classification

The final system can be accessed through an interactive Streamlit application or a FastAPI prediction service.

---

## Final Model Performance

The final tuned Gradient Boosting model was evaluated once on the previously untouched test dataset using a decision threshold of `0.35`.

| Metric | Final Test Score |
|---|---:|
| Accuracy | 0.7786 |
| Precision | 0.5643 |
| Recall | **0.7273** |
| F1 Score | **0.6355** |
| ROC-AUC | **0.8487** |

The model correctly identified **272 of 374 customers who churned**, resulting in a churn recall of **72.73%**.

### Final Confusion Matrix

|  | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 825 | 210 |
| Actual Churn | 102 | 272 |

The final test results were close to the development results, indicating that the selected model generalized well to unseen customers.

No additional model or threshold tuning was performed after evaluating the final test set.

---

## Project Highlights

- End-to-end binary classification workflow using **7,043 telecom customer records**
- Leakage-safe preprocessing with scikit-learn `Pipeline` and `ColumnTransformer`
- 80/20 stratified train/test split
- 5-fold stratified cross-validation
- Comparison of multiple machine learning models
- Class imbalance experiment with class-weighted Logistic Regression
- Hyperparameter tuning with `RandomizedSearchCV`
- Focused refinement with `GridSearchCV`
- Classification threshold optimization using out-of-fold predictions
- Final evaluation on a previously untouched test set
- Full preprocessing and model pipeline serialized with `joblib`
- FastAPI REST prediction service
- Pydantic request validation
- Interactive Streamlit application
- Automated API tests with `pytest`
- Docker containerization
- GitHub Actions continuous integration
- Public Streamlit cloud deployment

---

## Dataset

The project uses the **IBM Telco Customer Churn dataset**.

- **Customers:** 7,043
- **Original columns:** 21
- **Target:** `Churn`
- **Problem type:** Binary classification
- **Predictors used:** 19

The target contains:

- `No` — customer did not churn
- `Yes` — customer churned

The `customerID` field is excluded from model training because it is an identifier rather than a predictive feature.

### Numerical Features

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

### Categorical Features

- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

---

## Data Preparation

The data preparation workflow includes:

- Dataset shape and schema inspection
- Missing-value inspection
- Data-type validation
- Conversion of `TotalCharges` to numeric
- Review of customers with zero tenure
- Removal of `customerID` from model features
- Stratified train/test split
- Categorical encoding
- Model-specific numerical preprocessing

### Train/Test Split

The cleaned dataset was divided using an **80/20 stratified split** with `random_state=42`.

| Dataset | Rows | Churn Rate |
|---|---:|---:|
| Full Dataset | 7,043 | 26.54% |
| Training Set | 5,634 | 26.54% |
| Test Set | 1,409 | 26.54% |

Stratification preserved approximately the same churn distribution in both datasets.

The test set remained untouched during model comparison, tuning, and threshold selection.

---

## Preprocessing Pipeline

Preprocessing is implemented using scikit-learn `ColumnTransformer` and `Pipeline` objects.

Keeping preprocessing inside the model pipeline helps prevent data leakage and ensures that the same transformations are consistently applied during training and inference.

### Logistic Regression

For Logistic Regression experiments:

- Numerical features are standardized using `StandardScaler`
- Categorical features are encoded using `OneHotEncoder(handle_unknown="ignore")`

### Tree-Based Models

For Random Forest and Gradient Boosting:

- Numerical features are passed through without scaling
- Categorical features are one-hot encoded

The final production model uses the tree-based preprocessing configuration.

---

## Model Development

Several models were evaluated using the same **5-fold StratifiedKFold cross-validation** setup.

### Baseline Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **0.8021** | 0.6529 | 0.5431 | 0.5923 | 0.8462 |
| Class-Weighted Logistic Regression | 0.7485 | 0.5169 | **0.8013** | **0.6283** | 0.8460 |
| Random Forest | 0.7861 | 0.6262 | 0.4823 | 0.5446 | 0.8201 |
| Gradient Boosting | **0.8033** | **0.6619** | 0.5298 | 0.5883 | **0.8481** |

### Observations

The standard Logistic Regression baseline provided strong overall performance and a ROC-AUC of `0.8462`.

Using `class_weight="balanced"` substantially increased churn recall from `0.5431` to `0.8013`, but reduced precision and accuracy.

Random Forest performed below Logistic Regression and Gradient Boosting on the main evaluation metrics.

Gradient Boosting produced the highest baseline ROC-AUC and was selected for further tuning.

---

## Hyperparameter Tuning

### RandomizedSearchCV

Gradient Boosting was first tuned using `RandomizedSearchCV`.

- **Candidates:** 40
- **Cross-validation folds:** 5
- **Total fits:** 200
- **Optimization metric:** ROC-AUC

Best parameters:

```text
learning_rate     = 0.20
n_estimators      = 100
max_depth         = 1
min_samples_split = 2
min_samples_leaf  = 2
subsample         = 0.85
```

Best development cross-validation ROC-AUC:

```text
0.8512
```

### Focused GridSearchCV

A smaller `GridSearchCV` was then performed around the most promising parameter region.

- **Parameter combinations:** 36
- **Cross-validation folds:** 5
- **Total fits:** 180

Final selected parameters:

```text
learning_rate     = 0.15
n_estimators      = 125
max_depth         = 1
min_samples_split = 2
min_samples_leaf  = 2
subsample         = 0.85
random_state      = 42
```

Best development cross-validation ROC-AUC:

```text
0.8513
```

The improvement over RandomizedSearchCV was only `0.0001`, indicating that hyperparameter tuning had effectively plateaued.

---

## Classification Threshold Analysis

Binary classifiers commonly use a default decision threshold of `0.50`.

Because identifying customers at risk of churn is an important objective, several classification thresholds were evaluated using **out-of-fold training probabilities**.

The tuned model produced an out-of-fold ROC-AUC of:

```text
0.8507
```

### Threshold Comparison

| Threshold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.20 | 0.7167 | 0.4813 | 0.8689 | 0.6195 |
| 0.25 | 0.7448 | 0.5118 | 0.8288 | 0.6328 |
| 0.30 | 0.7638 | 0.5383 | 0.7712 | 0.6340 |
| **0.35** | **0.7879** | **0.5813** | **0.7171** | **0.6421** |
| 0.40 | 0.7971 | 0.6104 | 0.6508 | 0.6300 |
| 0.45 | 0.8074 | 0.6466 | 0.6047 | 0.6250 |
| 0.50 | 0.8058 | 0.6675 | 0.5344 | 0.5936 |

The best F1 score was achieved at a threshold of:

```text
0.35
```

Compared with the default `0.50` threshold, `0.35` substantially improved churn recall while maintaining a more balanced precision/recall tradeoff.

The threshold was selected **before** final test-set evaluation.

---

## Final Model

The production model is a tuned:

```text
GradientBoostingClassifier
```

with:

```text
learning_rate     = 0.15
n_estimators      = 125
max_depth         = 1
min_samples_split = 2
min_samples_leaf  = 2
subsample         = 0.85
random_state      = 42
```

Decision threshold:

```text
0.35
```

The saved artifact contains:

- Preprocessing pipeline
- One-hot encoder
- Gradient Boosting classifier
- Selected classification threshold
- Expected feature list

The complete artifact is stored using `joblib`.

---

## Application Architecture

```text
Customer Input
      ↓
Streamlit UI / FastAPI
      ↓
Input Validation
      ↓
Prediction Module
      ↓
Saved scikit-learn Pipeline
      ↓
Categorical Encoding
      ↓
Gradient Boosting Model
      ↓
Churn Probability
      ↓
Decision Threshold = 0.35
      ↓
Churn / No Churn
```

---

## Prediction Module

The reusable prediction module loads the saved model artifact and exposes a prediction function that accepts customer information and returns:

- Churn probability
- Decision threshold
- Final churn prediction

Example output:

```text
Probability : 74.77%
Threshold   : 0.35
Prediction  : Churn
```

---

## FastAPI Prediction Service

The trained model is also exposed through a FastAPI REST API.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Model/API health check |
| POST | `/predict` | Generate a churn prediction |
| GET | `/docs` | Interactive Swagger documentation |

The `/predict` endpoint validates incoming customer data using Pydantic before passing it to the saved machine learning pipeline.

### Run the API Locally

```bash
uvicorn src.api:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Automated Testing

API behavior is validated using `pytest` and FastAPI's test client.

Current tests cover:

- Root endpoint
- Health endpoint
- Valid prediction request
- Invalid customer request validation

Run the test suite:

```bash
python -m pytest -v
```

Current result:

```text
4 passed
```

---

## Streamlit Application

The project includes an interactive Streamlit interface that allows users to enter telecom customer information and receive a churn-risk prediction.

The interface collects:

- Customer information
- Service subscriptions
- Contract information
- Payment details
- Monthly charges
- Total charges

The output displays:

- Churn probability
- Decision threshold
- Final churn classification

### Run Locally

```bash
streamlit run streamlit_app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

### Live Application

**https://telecom-customer-churn-analysis.streamlit.app**

---

## Docker

The Streamlit application is containerized with Docker so it can run in a consistent environment without requiring manual local dependency installation.

### Build the Docker Image

```bash
docker build -t telecom-churn-app .
```

### Run the Container

```bash
docker run -d -p 8501:8501 --name churn-app telecom-churn-app
```

Open:

```text
http://localhost:8501
```

### Stop the Container

```bash
docker stop churn-app
```

### Restart the Container

```bash
docker start churn-app
```

The Docker image contains the saved machine learning pipeline, prediction module, and Streamlit application.

---

## Continuous Integration

GitHub Actions automatically validates the project on pushes and pull requests to `main`.

The CI workflow performs:

```text
Checkout repository
        ↓
Set up Python 3.12
        ↓
Install dependencies
        ↓
Run pytest
        ↓
Build Docker image
```

The Docker build runs only after the automated test job succeeds.

Current CI status is displayed in the badge at the top of this README.

---

## Cloud Deployment

The Streamlit application is deployed publicly using Streamlit Community Cloud.

Live application:

**https://telecom-customer-churn-analysis.streamlit.app**

The deployment uses the project's GitHub repository and `requirements.txt` to reproduce the application environment in the cloud.

---

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
Class Imbalance Experiment
      ↓
Random Forest
      ↓
Gradient Boosting
      ↓
Model Comparison
      ↓
RandomizedSearchCV
      ↓
Focused GridSearchCV
      ↓
Threshold Analysis
      ↓
Final Test Evaluation
      ↓
Model Serialization
      ↓
Prediction Module
      ↓
FastAPI
      ↓
Automated Testing
      ↓
Streamlit
      ↓
Docker
      ↓
GitHub Actions CI
      ↓
Cloud Deployment
```

---

## Project Structure

```text
telecom-customer-churn/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.joblib
│
├── src/
│   ├── __init__.py
│   ├── data_check.py
│   ├── data_cleaning.py
│   ├── split_data.py
│   ├── train_baseline.py
│   ├── compare_logistic_models.py
│   ├── train_random_forest.py
│   ├── train_gradient_boosting.py
│   ├── tune_gradient_boosting.py
│   ├── gridsearch_gradient_boosting.py
│   ├── analyze_thresholds.py
│   ├── evaluate_final_model.py
│   ├── save_model.py
│   ├── predict.py
│   └── api.py
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── streamlit_app.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technology Stack

### Machine Learning and Data

- Python
- Pandas
- NumPy
- scikit-learn
- joblib

### Modeling

- Logistic Regression
- Random Forest
- Gradient Boosting
- Stratified cross-validation
- RandomizedSearchCV
- GridSearchCV
- Classification threshold optimization

### API and Application

- FastAPI
- Pydantic
- Streamlit

### Testing

- pytest
- FastAPI TestClient

### Engineering and Deployment

- Git
- GitHub
- Docker
- GitHub Actions
- Streamlit Community Cloud

---

## Key Engineering Practices

This project follows several practices intended to make the machine learning workflow reproducible and production-oriented:

- Maintain a separate final test dataset
- Prevent preprocessing leakage
- Use reproducible random seeds
- Perform stratified train/test splitting
- Use stratified cross-validation
- Compare multiple evaluation metrics
- Avoid relying only on accuracy for an imbalanced classification problem
- Keep preprocessing and modeling inside reusable pipelines
- Track model experiments before final selection
- Use out-of-fold probabilities for threshold selection
- Select model parameters and threshold before evaluating the test set
- Serialize the full production pipeline
- Validate API inputs
- Test application endpoints automatically
- Containerize the application
- Run automated CI checks on repository changes

---

## Project Status

The core end-to-end machine learning lifecycle is complete.

- [x] Dataset inspection
- [x] Data cleaning
- [x] Exploratory data analysis
- [x] Stratified train/test split
- [x] Feature preprocessing pipeline
- [x] Logistic Regression baseline
- [x] 5-fold cross-validation
- [x] Class imbalance experiments
- [x] Additional model comparison
- [x] Hyperparameter tuning
- [x] Classification threshold analysis
- [x] Final test-set evaluation
- [x] Model persistence
- [x] FastAPI prediction service
- [x] Streamlit application
- [x] Automated API tests
- [x] Docker containerization
- [x] GitHub Actions CI
- [x] Cloud deployment

---

## Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/bzay1729/telecom-customer-churn.git
cd telecom-customer-churn
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Automated Tests

```bash
python -m pytest -v
```

### 5. Run the Streamlit Application

```bash
streamlit run streamlit_app.py
```

Or run the containerized version:

```bash
docker build -t telecom-churn-app .
docker run -d -p 8501:8501 --name churn-app telecom-churn-app
```

---

## Live Demo

**Telecom Customer Churn Prediction**

https://telecom-customer-churn-analysis.streamlit.app