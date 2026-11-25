"""
Constants definition module for Customer Churn Prediction project.

This module centralizes configuration values, paths, feature lists,
and model hyperparameters used across the churn library and test suite.

Author: Amit Kumar
Date: October 2026
"""

# Directory paths
EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
LOGS_DIR = "./logs"

# File paths
DATA_PATH = "./data/bank_data.csv"
LOG_FILE = "./logs/churn_library.log"
RFC_MODEL_PATH = "./models/rfc_model.pkl"
LOGISTIC_MODEL_PATH = "./models/logistic_model.pkl"

# EDA Image paths
CHURN_DIST_PATH = "./images/eda/churn_distribution.png"
CUSTOMER_AGE_DIST_PATH = "./images/eda/customer_age_distribution.png"
MARITAL_STATUS_DIST_PATH = "./images/eda/marital_status_distribution.png"
TOTAL_TRANS_CT_DIST_PATH = "./images/eda/total_trans_ct_distribution.png"
HEATMAP_PATH = "./images/eda/heatmap.png"

# Model Result Image paths
ROC_CURVE_PATH = "./images/results/roc_curve_result.png"
RFC_RESULTS_PATH = "./images/results/rf_results.png"
LOGISTIC_RESULTS_PATH = "./images/results/logistic_results.png"
FEATURE_IMPORTANCE_PATH = "./images/results/feature_importances.png"

# Target / Response column
RESPONSE = "Churn"

# Categorical columns to encode
CATEGORY_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

# Quantitative feature columns
QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# Keep columns for training
KEEP_COLUMNS = QUANT_COLUMNS + \
    [f"{col}_{RESPONSE}" for col in CATEGORY_COLUMNS]

# Modeling parameters
RANDOM_STATE = 42
TEST_SIZE = 0.3
RFC_PARAM_DIST = {
    "n_estimators": [200, 300],
    "max_features": ["sqrt"],
    "max_depth": [5, 8, 10],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "criterion": ["gini"],
}
CV_FOLDS = 3
N_ITER_SEARCH = 12
LOGISTIC_MAX_ITER = 3000
