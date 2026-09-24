# Explainable Credit Risk Prediction System

An end-to-end machine learning project for predicting credit default risk using structured financial/customer data, with a focus on **data quality, leakage prevention, model evaluation, hyperparameter tuning, and decision-threshold optimization**.

The project is being developed incrementally toward a production-oriented ML system with explainability and API deployment.

---

## 🚧 Project Status

**Current Progress: Phase 5 completed**

| Phase                               | Status      |
| ----------------------------------- | ----------- |
| Phase 1 — Data Loading & Validation | ✅ Completed |
| Phase 2 — EDA & Preprocessing       | ✅ Completed |
| Phase 3 — Baseline Models           | ✅ Completed |
| Phase 4 — Hyperparameter Tuning     | ✅ Completed |
| Phase 5 — Threshold Optimization    | ✅ Completed |
| Phase 6 — Probability Calibration   | ⏳ Planned   |
| Phase 7 — SHAP Explainability       | ⏳ Planned   |
| Phase 8 — Model Packaging           | ⏳ Planned   |
| Phase 9 — FastAPI Backend           | ⏳ Planned   |
| Phase 10 — Frontend & Deployment    | ⏳ Planned   |

---

# 🎯 Project Objective

The goal is to build a complete credit-risk prediction pipeline capable of:

* Loading and validating credit-risk data
* Performing exploratory data analysis
* Handling missing values and categorical/numerical features
* Preventing data leakage
* Training baseline classification models
* Training and tuning an XGBoost model
* Evaluating models using appropriate classification metrics
* Optimizing the prediction threshold according to a defined business objective
* Producing reliable probability estimates
* Providing interpretable predictions using SHAP
* Serving predictions through a FastAPI backend
* Providing a user-facing web interface
* Deploying the final application

The final system will separate:

```text
Probability Prediction
        ↓
Threshold Decision
        ↓
Risk Classification
        ↓
Model Explanation
```

---

# 🏗️ Current ML Pipeline

The current pipeline is:

```text
Raw Dataset
     ↓
Data Validation
     ↓
Exploratory Data Analysis
     ↓
Train/Test Split
     ↓
Preprocessing
     ↓
Baseline Models
     ↓
XGBoost Hyperparameter Tuning
     ↓
Threshold Optimization
     ↓
Final Evaluation
```

Future versions will extend this pipeline with calibration, explainability, model packaging, API serving, and deployment.

---

# 📁 Project Structure

```text
credit-risk-ml/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_model_tuning.ipynb
│   └── 04_threshold_optimization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── validator.py
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   └── analysis.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── split.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── tune.py
│   │
│   └── optimization/
│       ├── __init__.py
│       └── threshold.py
│
├── artifacts/
│   ├── models/
│   ├── thresholds/
│   └── metadata/
│
├── reports/
│   ├── evaluation/
│   └── threshold_analysis/
│
├── tests/
│   ├── test_loader.py
│   ├── test_validator.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_tuning.py
│   └── test_threshold.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔹 Phase 1 — Data Loading & Validation

The first phase establishes a reliable data foundation.

### Implemented

* Dataset loading
* File validation
* Target-column validation
* Basic schema validation
* Missing-value analysis
* Numerical/categorical feature identification
* Invalid target detection
* Dataset structure checks

### Main Components

```text
src/data/loader.py
src/data/validator.py
```

The validation layer helps prevent invalid or unexpected data from entering the ML pipeline.

---

# 🔹 Phase 2 — EDA & Preprocessing

The second phase focuses on understanding the dataset and preparing it for machine learning.

### Implemented

* Exploratory data analysis
* Target distribution analysis
* Numerical feature analysis
* Categorical feature analysis
* Missing-value analysis
* Feature identification
* Train/test splitting
* Numerical preprocessing
* Categorical preprocessing
* Reproducible data splitting

### Important Principle

The preprocessing pipeline is designed to prevent **data leakage**.

Preprocessing transformations are fitted only on the training data and then applied to validation/test data.

```text
Training Data
     ↓
Fit Preprocessor
     ↓
Transform Training Data

Validation/Test Data
     ↓
Transform Using Existing Preprocessor
```

---

# 🔹 Phase 3 — Baseline Models

Baseline models establish a performance reference before more advanced optimization.

### Models

* Logistic Regression
* XGBoost

### Evaluation Metrics

The project evaluates classification performance using metrics such as:

* ROC AUC
* PR AUC / Average Precision
* Precision
* Recall
* F1 Score
* Brier Score
* Log Loss

Metrics are selected according to the characteristics of the credit-risk classification problem.

### Validation Strategy

Stratified cross-validation is used where appropriate to preserve class distribution.

The held-out test set is kept separate from model-selection decisions.

---

# 🔹 Phase 4 — Hyperparameter Tuning

After establishing baseline performance, the XGBoost model is tuned using randomized hyperparameter search.

### Tuned Parameters

The search space can include:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
reg_alpha
reg_lambda
```

### Goals

* Improve generalization
* Reduce overfitting
* Find a stronger XGBoost configuration
* Select the final model using validation/cross-validation performance

The primary model-selection metric is configurable and documented with the experiment.

---

# 🔹 Phase 5 — Threshold Optimization

A classification model produces a probability:

```text
P(default)
```

A threshold is then used to convert that probability into a binary prediction.

For example:

```text
Probability >= threshold
        ↓
     Default

Probability < threshold
        ↓
     Non-default
```

The default `0.5` threshold is not automatically assumed to be optimal.

### Supported Optimization Strategies

The threshold optimization framework can support objectives such as:

* Maximize F1
* Meet minimum recall
* Meet minimum precision
* Minimize an explicit business cost function

### Example

```text
Predicted Probability = 0.72

Threshold = 0.50
        ↓
Default

Threshold = 0.80
        ↓
Non-default
```

The selected threshold is saved as a project artifact so that the same decision rule can later be used by the production API.

### Important Principle

Threshold optimization does **not** use the final held-out test set for selecting the threshold.

```text
Training Data
      ↓
Model Training
      ↓
Validation / OOF Predictions
      ↓
Threshold Optimization
      ↓
Final Test Evaluation
```

---

# 📊 Evaluation Philosophy

This project does not rely on accuracy alone.

Credit-risk datasets can contain class imbalance, meaning accuracy may provide an incomplete picture of model performance.

Therefore, the project considers multiple metrics:

| Metric      | Purpose                                     |
| ----------- | ------------------------------------------- |
| ROC AUC     | Ranking ability across thresholds           |
| PR AUC      | Performance under class imbalance           |
| Precision   | Reliability of positive/default predictions |
| Recall      | Ability to identify defaults                |
| F1          | Balance between precision and recall        |
| Brier Score | Probability quality                         |
| Log Loss    | Quality of predicted probabilities          |

The final model-selection strategy and threshold strategy are explicitly recorded rather than hidden inside the implementation.

---

# 🔐 Data Leakage Prevention

Avoiding data leakage is a core design requirement.

The project follows these principles:

* Test data remains isolated
* Preprocessors are fitted only on training data
* Model tuning does not use the final test set
* Threshold selection does not use the final test set
* Validation/cross-validation is used for model-selection decisions
* Test data is reserved for final evaluation

Conceptually:

```text
                 ┌───────────────┐
                 │  Raw Dataset  │
                 └───────┬───────┘
                         │
                    Train / Test
                    Split
                    /       \
                   /         \
              Training       Test
                 │              │
          Model Selection       │
          Tuning                │
          Threshold             │
                 │              │
                 └──────┬───────┘
                        │
                  Final Evaluation
```

---

# 🧪 Testing

The project includes tests for the major components of the ML pipeline.

Current testing areas include:

* Data loading
* Data validation
* Preprocessing
* Train/test splitting
* Model creation
* Model evaluation
* Hyperparameter tuning
* Threshold optimization

The goal is to ensure that ML components behave consistently and fail clearly when invalid inputs are supplied.

---

# 🛠️ Technology Stack

### Programming

* Python

### Data Processing

* pandas
* NumPy

### Machine Learning

* scikit-learn
* XGBoost

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

### Planned

* SHAP
* FastAPI
* Pydantic
* HTML/CSS/JavaScript
* Render or similar deployment platform

---

# 📦 Current Artifacts

The project is designed to save important ML artifacts rather than relying only on notebooks.

Current artifact categories include:

```text
artifacts/
├── models/
├── thresholds/
└── metadata/
```

These will later be expanded to include calibrated models and explainability configuration.

---

# 🔮 Upcoming Phases

## Phase 6 — Probability Calibration

Improve the reliability of predicted default probabilities using calibration techniques such as sigmoid/Platt scaling.

---

## Phase 7 — SHAP Explainability

Add:

* Global feature importance
* SHAP summary plots
* Local explanations
* Per-prediction feature contributions

The explanation layer will explain **model behavior**, not claim that a feature causes a customer's outcome.

---

## Phase 8 — Model Packaging

Create versioned, reusable model artifacts containing:

* Model
* Preprocessor
* Calibration configuration
* Threshold
* Metadata
* Explanation configuration

---

## Phase 9 — FastAPI Backend

Build an API capable of:

```text
Input Customer Data
        ↓
Preprocessing
        ↓
Model
        ↓
Probability
        ↓
Threshold
        ↓
Risk Prediction
        ↓
Explanation
```

---

## Phase 10 — Frontend, Deployment & QA

Build a web interface where users can enter customer information and receive:

* Default probability
* Risk classification
* Prediction explanation
* Model information

The final application will then be tested and deployed.

---

# 🚀 Target Final Architecture

The completed project is intended to follow:

```text
                    ┌─────────────────┐
                    │   User / Client │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    FastAPI      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Preprocessor  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    XGBoost      │
                    └────────┬────────┘
                             │
                   ┌─────────┴─────────┐
                   ▼                   ▼
             Probability          SHAP
                   │             Explanation
                   ▼
             Calibration
                   │
                   ▼
              Threshold
                   │
                   ▼
            Final Prediction
```

---

# 🎓 Learning Objectives

This project is designed to demonstrate practical understanding of:

* End-to-end ML pipelines
* Data validation
* Exploratory data analysis
* Feature preprocessing
* Classification
* Imbalanced classification
* Cross-validation
* Hyperparameter optimization
* XGBoost
* Probability prediction
* Threshold optimization
* Model evaluation
* Explainable AI
* ML artifact management
* API-based model serving
* Production-oriented ML architecture

---

# ⚠️ Disclaimer

This project is an educational and portfolio implementation of a credit-risk prediction system.

It should not be treated as a real-world lending or financial decision system without appropriate validation, governance, fairness assessment, regulatory review, security controls, and domain-specific oversight.

---

# 📌 Development Philosophy

The project is being developed incrementally rather than building everything inside a single notebook.

Core principles:

```text
Reproducibility
      +
Data Leakage Prevention
      +
Modular Architecture
      +
Testability
      +
Explainability
      +
Production Readiness
```

The objective is to move from:

```text
ML Experiment
      ↓
Reusable ML Pipeline
      ↓
Explainable ML System
      ↓
Production API
      ↓
Deployable Application
```

---

## Author

**Ashutosh**

Built as a machine-learning portfolio project with a focus on practical ML engineering and explainable AI.
