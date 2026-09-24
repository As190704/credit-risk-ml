# Explainable Credit Risk Prediction System

An end-to-end, production-oriented machine learning system for predicting credit default risk using **XGBoost, probability calibration, SHAP explainability, and FastAPI**.

The project focuses on building a reliable ML pipeline from data validation through model serving, while maintaining **reproducibility, leakage prevention, explainability, and modular architecture**.

---

## 🚧 Project Status

**Current Progress: Phase 9 completed**

| Phase                                | Status      |
| ------------------------------------ | ----------- |
| Phase 1 — Data Loading & Validation  | ✅ Completed |
| Phase 2 — EDA & Preprocessing        | ✅ Completed |
| Phase 3 — Baseline Models            | ✅ Completed |
| Phase 4 — Hyperparameter Tuning      | ✅ Completed |
| Phase 5 — Threshold Optimization     | ✅ Completed |
| Phase 6 — Probability Calibration    | ✅ Completed |
| Phase 7 — SHAP Explainability        | ✅ Completed |
| Phase 8 — Model Packaging            | ✅ Completed |
| Phase 9 — FastAPI Backend            | ✅ Completed |
| Phase 10 — Frontend, Deployment & QA | ⏳ Planned   |

---

# 🎯 Project Objective

The objective is to build an end-to-end credit-risk prediction system capable of:

* Validating incoming credit-risk data
* Performing exploratory data analysis
* Building leakage-safe preprocessing pipelines
* Training classification models
* Tuning XGBoost
* Optimizing classification thresholds
* Calibrating predicted probabilities
* Generating SHAP-based explanations
* Packaging reusable ML artifacts
* Serving predictions through a REST API
* Providing transparent prediction information
* Deploying the complete application

The final system separates three important concepts:

```text
Probability Prediction
        ↓
Threshold Decision
        ↓
Risk Classification
```

while explainability operates separately:

```text
Processed Input
        ↓
ML Model
        ↓
SHAP
        ↓
Feature Contributions
```

---

# 🏗️ Current ML Pipeline

The project currently follows:

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
XGBoost Tuning
     ↓
Threshold Optimization
     ↓
Probability Calibration
     ↓
SHAP Explainability
     ↓
Model Packaging
     ↓
FastAPI
```

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
│   ├── 04_threshold_optimization.ipynb
│   ├── 05_calibration.ipynb
│   └── 06_shap_analysis.ipynb
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
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── threshold.py
│   │
│   ├── calibration/
│   │   ├── __init__.py
│   │   ├── calibrator.py
│   │   └── evaluate.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   ├── global_explanation.py
│   │   └── local_explanation.py
│   │
│   └── artifacts/
│       ├── __init__.py
│       ├── save.py
│       └── load.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies.py
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── health.py
│   │   ├── prediction.py
│   │   └── model_info.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── prediction.py
│   │   └── response.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── prediction_service.py
│       └── explanation_service.py
│
├── artifacts/
│   ├── models/
│   ├── calibration/
│   ├── thresholds/
│   └── metadata/
│
├── reports/
│   ├── evaluation/
│   ├── calibration/
│   ├── threshold_analysis/
│   └── shap/
│
├── tests/
│   ├── test_loader.py
│   ├── test_validator.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_tuning.py
│   ├── test_threshold.py
│   ├── test_calibration.py
│   ├── test_shap.py
│   ├── test_artifacts.py
│   └── test_api.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🔹 Phase 1 — Data Loading & Validation

The first phase establishes a reliable data foundation.

### Implemented

* Dataset loading
* File validation
* Target-column validation
* Schema validation
* Missing-value analysis
* Numerical/categorical feature identification
* Invalid target detection
* Dataset structure checks

The validation layer prevents malformed or unexpected data from silently entering the ML pipeline.

---

# 🔹 Phase 2 — EDA & Preprocessing

This phase focuses on understanding the dataset and creating a leakage-safe preprocessing pipeline.

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

### Leakage Prevention

Preprocessing transformations are fitted only on training data.

```text
Training Data
     ↓
Fit Preprocessor
     ↓
Transform Training Data

Test Data
     ↓
Transform Using Existing Preprocessor
```

---

# 🔹 Phase 3 — Baseline Models

Baseline models establish a reference point before optimization.

### Models

* Logistic Regression
* XGBoost

### Evaluation Metrics

The system evaluates models using multiple metrics:

* ROC AUC
* PR AUC / Average Precision
* Precision
* Recall
* F1 Score
* Brier Score
* Log Loss

Stratified validation is used where appropriate to preserve class distribution.

The held-out test set is kept separate from model-selection decisions.

---

# 🔹 Phase 4 — XGBoost Hyperparameter Tuning

The XGBoost model is optimized using randomized hyperparameter search.

### Tuned Parameters

The configurable search space includes parameters such as:

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

### Objectives

* Improve generalization
* Reduce overfitting
* Identify a stronger XGBoost configuration
* Select the model using validation/cross-validation performance

The model-selection metric is explicitly documented rather than hidden inside the training process.

---

# 🔹 Phase 5 — Threshold Optimization

Classification models produce probabilities rather than inherently producing a binary decision.

The system therefore separates:

```text
Probability
     ↓
Threshold
     ↓
Binary Prediction
```

The default `0.5` threshold is not automatically assumed to be optimal.

### Supported Strategies

The threshold optimization framework can support:

* Maximizing F1
* Meeting minimum recall
* Meeting minimum precision
* Minimizing an explicit cost function

### Leakage Prevention

Threshold selection is performed using validation data or out-of-fold predictions.

The final held-out test set is not used to choose the threshold.

---

# 🔹 Phase 6 — Probability Calibration

The model's raw probabilities are calibrated to improve their correspondence with observed outcomes.

### Calibration

The project supports probability calibration using methods such as:

* Sigmoid / Platt scaling

The calibrated model is evaluated against the uncalibrated model.

### Evaluation

Calibration analysis includes metrics such as:

* Brier Score
* Log Loss
* ROC AUC
* PR AUC
* Calibration curves

An important distinction is maintained:

> Calibration improves probability reliability; it does not necessarily improve ranking metrics such as ROC AUC.

---

# 🔹 Phase 7 — SHAP Explainability

SHAP is used to provide model explanations at both global and individual prediction levels.

### Global Explainability

The project generates:

* SHAP summary plots
* SHAP feature importance
* Global feature contribution analysis

### Local Explainability

Individual predictions can be explained using feature-level SHAP contributions.

Conceptually:

```text
Input
  ↓
Model
  ↓
Prediction Probability
  ↓
SHAP Contributions
  ↓
Top Influencing Features
```

### Important Limitation

SHAP explanations describe **how the model uses features**.

They should not be interpreted as proof that a feature caused the customer's outcome.

---

# 🔹 Phase 8 — Model Packaging

The trained ML system is packaged into reusable, versioned artifacts.

Example:

```text
artifacts/
├── models/
│   └── credit_risk_model/
│       └── v1/
│           ├── model.joblib
│           ├── preprocessor.joblib
│           └── shap_metadata.json
│
├── calibration/
│   └── v1/
│       └── calibration.joblib
│
├── thresholds/
│   └── v1/
│       └── threshold.json
│
└── metadata/
    └── v1/
        └── model_metadata.json
```

The exact artifact structure may vary depending on the implementation.

### Metadata

Metadata can include:

* Model name
* Model version
* Model type
* Calibration method
* Threshold
* Threshold strategy
* Feature count
* Feature names
* Target definition
* Dataset identifier
* Training timestamp
* Python/library versions
* Evaluation metrics
* Calibration metrics
* Git commit when available

Sensitive applicant information is not stored in model metadata.

---

# 🔹 Phase 9 — FastAPI Backend

The packaged ML system is exposed through a REST API using FastAPI.

The API loads existing artifacts instead of retraining the model.

### Backend Architecture

```text
Client
  ↓
FastAPI
  ↓
Pydantic Validation
  ↓
Prediction Service
  ↓
Preprocessing
  ↓
XGBoost
  ↓
Calibration
  ↓
Probability
  ↓
Threshold
  ↓
Prediction
```

SHAP explanations are available through a separate explanation flow.

---

## API Endpoints

### Health Check

```text
GET /health
```

Used to verify that the API is running and the model artifacts are available.

---

### Model Information

```text
GET /model-info
```

Returns safe model metadata such as:

* Model name
* Model version
* Model type
* Calibration method
* Decision threshold
* Feature information

---

### Prediction

```text
POST /predict
```

Accepts applicant feature data and returns a prediction.

Conceptual response:

```json
{
  "prediction": 1,
  "risk_label": "default",
  "default_probability": 0.73,
  "threshold": 0.50,
  "model_version": "v1.0.0"
}
```

The actual response fields depend on the implemented API schema.

---

### Explanation

```text
POST /explain
```

Generates a prediction together with SHAP-based feature contributions.

Conceptual response:

```json
{
  "default_probability": 0.73,
  "prediction": 1,
  "explanation": [
    {
      "feature": "feature_a",
      "value": 42,
      "contribution": 0.18
    },
    {
      "feature": "feature_b",
      "value": 3,
      "contribution": -0.11
    }
  ]
}
```

---

# 📖 Interactive API Documentation

FastAPI provides interactive documentation through:

```text
/docs
```

and:

```text
/redoc
```

These interfaces allow the API endpoints and schemas to be inspected and tested during development.

---

# 🔐 API Design Principles

The backend follows these principles:

* Models are loaded from trusted artifacts
* Models are not retrained on API startup
* Preprocessing is reused from the ML pipeline
* Thresholds are loaded from configuration/artifacts
* Model versions are loaded from metadata
* Input validation is handled through Pydantic
* API errors are handled without exposing internal details
* Sensitive information is not returned to clients
* ML logic is separated from API logic

---

# 🧪 Testing

Testing covers the major components of the system.

### Data

* Data loading
* Schema validation
* Target validation

### Features

* Preprocessing
* Train/test splitting
* Leakage prevention

### Models

* Model creation
* Model evaluation
* Hyperparameter tuning

### Optimization

* Threshold calculation
* Threshold validation

### Calibration

* Calibration fitting
* Probability range validation
* Calibration metrics

### Explainability

* SHAP explainer initialization
* SHAP value/feature alignment
* Global explanation
* Local explanation

### Artifacts

* Artifact saving
* Artifact loading
* Metadata validation
* Model reload and prediction

### API

* Health endpoint
* Model information
* Valid prediction
* Invalid input
* Missing feature
* Probability validation
* Threshold application
* Explanation endpoint

---

# 📊 Evaluation Philosophy

The project does not rely on accuracy alone.

Credit-risk classification can involve class imbalance, so multiple metrics are considered.

| Metric      | Purpose                                                        |
| ----------- | -------------------------------------------------------------- |
| ROC AUC     | Measures ranking ability across thresholds                     |
| PR AUC      | Useful for evaluating performance under class imbalance        |
| Precision   | Measures correctness among predicted positive/default cases    |
| Recall      | Measures how many actual positive/default cases are identified |
| F1          | Harmonic mean of precision and recall                          |
| Brier Score | Measures probability prediction quality                        |
| Log Loss    | Evaluates probabilistic predictions                            |

Actual model metrics should be reported from the project's experiments rather than manually entered or invented.

---

# 🔐 Data Leakage Prevention

Leakage prevention is maintained throughout the complete pipeline.

```text
Raw Dataset
     ↓
Train / Test Split
     ↓
Training Data
     ├── Preprocessing
     ├── Model Training
     ├── Hyperparameter Tuning
     ├── Calibration
     └── Threshold Optimization
     
Held-out Test Data
     ↓
Final Evaluation
```

The test set is kept isolated from model-selection decisions.

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

### Explainable AI

* SHAP

### API

* FastAPI
* Pydantic

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

### Planned

* HTML
* CSS
* JavaScript
* Cloud deployment

---

# 🔮 Phase 10 — Next Step

The remaining phase is:

## Frontend + Deployment + Production QA

The planned final phase will add:

### Frontend

* User input form
* Prediction interface
* Probability display
* Risk classification
* Threshold display
* SHAP explanation visualization
* Model information

### Deployment

* Production configuration
* API deployment
* Frontend deployment
* CORS configuration
* Environment configuration

### QA

* End-to-end testing
* Error handling
* API/frontend integration testing
* Deployment verification
* Final documentation

The target architecture will be:

```text
                 USER
                   │
                   ▼
          ┌─────────────────┐
          │    Frontend     │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │     FastAPI     │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Preprocessing  │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │    XGBoost      │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │   Calibration   │
          └────────┬────────┘
                   │
                   ▼
           Default Probability
                   │
                   ▼
          ┌─────────────────┐
          │    Threshold    │
          └────────┬────────┘
                   │
                   ▼
            Risk Prediction
                   │
             ┌─────┴─────┐
             ▼           ▼
           SHAP       Response
             │           │
             └─────┬─────┘
                   ▼
               Frontend
```

---

# 🚀 Target Final System

After Phase 10, the project will progress from:

```text
ML Experiment
      ↓
Reusable ML Pipeline
      ↓
Explainable ML System
      ↓
FastAPI Backend
      ↓
Web Application
      ↓
Deployed ML Application
```

---

# 🎓 Learning Objectives

This project demonstrates practical knowledge of:

* Data validation
* Exploratory data analysis
* Feature preprocessing
* Classification
* Imbalanced classification
* Cross-validation
* Hyperparameter optimization
* XGBoost
* Probability calibration
* Threshold optimization
* Model evaluation
* Explainable AI
* SHAP
* ML artifact management
* Model versioning
* REST APIs
* FastAPI
* Pydantic
* ML testing
* Production-oriented ML architecture

---

# ⚠️ Disclaimer

This project is an educational and portfolio implementation of a credit-risk prediction system.

A real-world lending or financial decision system would require additional validation, governance, fairness assessment, security controls, regulatory review, monitoring, and domain-specific oversight.

---

# 📌 Development Principles

The project is developed incrementally with an emphasis on:

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
Versioned Artifacts
      +
Production Readiness
```

The goal is not simply to train a model, but to demonstrate how an ML model can be transformed into a **reusable and explainable software system**.

---

## Author

**Ashutosh**

Machine Learning / Data Analytics Portfolio Project
