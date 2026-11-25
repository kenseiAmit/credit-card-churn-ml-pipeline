"""
Customer Churn Prediction Library.

This module provides an end-to-end data science and machine learning pipeline
for identifying bank credit card customers at risk of churn. It includes routines
for data ingestion, exploratory data analysis (EDA), categorical encoding, feature
engineering, model training with hyperparameter tuning, model evaluation, and artifact storage.

Author: Amit Kumar
Date: October 2026
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import constants

# Force headless Matplotlib backend and configure writable cache
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
sns.set_theme()


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(constants.EDA_DIR, exist_ok=True)
    os.makedirs(constants.RESULTS_DIR, exist_ok=True)
    os.makedirs(constants.MODELS_DIR, exist_ok=True)
    os.makedirs(constants.LOGS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    """
    Perform exploratory data analysis on df and save figures to the images folder.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()
    eda_df = df.copy()

    # Ensure binary target column exists
    if constants.RESPONSE not in eda_df.columns and "Attrition_Flag" in eda_df.columns:
        eda_df[constants.RESPONSE] = eda_df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

    # 1. Univariate categorical target distribution
    plt.figure(figsize=(10, 6))
    eda_df[constants.RESPONSE].hist()
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn Status (0 = Existing, 1 = Churned)")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(constants.CHURN_DIST_PATH)
    plt.close()

    # 2. Univariate quantitative customer age distribution
    plt.figure(figsize=(10, 6))
    eda_df["Customer_Age"].hist()
    plt.title("Customer Age Distribution")
    plt.xlabel("Customer Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(constants.CUSTOMER_AGE_DIST_PATH)
    plt.close()

    # 3. Univariate categorical marital status distribution
    plt.figure(figsize=(10, 6))
    eda_df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
    plt.title("Marital Status Distribution (Normalized)")
    plt.xlabel("Marital Status")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(constants.MARITAL_STATUS_DIST_PATH)
    plt.close()

    # 4. Univariate quantitative total transaction count distribution with KDE
    plt.figure(figsize=(10, 6))
    sns.histplot(eda_df["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Total Transaction Count Distribution with KDE")
    plt.xlabel("Total Transaction Count")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(constants.TOTAL_TRANS_CT_DIST_PATH)
    plt.close()

    # 5. Bivariate correlation heatmap of numeric features
    plt.figure(figsize=(12, 8))
    numeric_df = eda_df.select_dtypes(include="number")
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="Dark2_r", linewidths=1)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(constants.HEATMAP_PATH)
    plt.close()


def encoder_helper(df, category_lst, response=None):
    """
    Turn each categorical column into a new column with proportion of churn.

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name (default: None, falls back to constants.RESPONSE)
    output:
            df: pandas dataframe with new encoded columns
    """
    df_encoded = df.copy()
    resp_col = response if response is not None else constants.RESPONSE

    if resp_col not in df_encoded.columns and "Attrition_Flag" in df_encoded.columns:
        df_encoded[resp_col] = df_encoded["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

    for category in category_lst:
        encoded_col_name = f"{category}_{resp_col}"
        category_means = df_encoded.groupby(category)[resp_col].mean()
        df_encoded[encoded_col_name] = df_encoded[category].map(category_means)

    return df_encoded


def perform_feature_engineering(df, response=None):
    """
    Split dataset into train and test sets with relevant engineered features.

    input:
            df: pandas dataframe
            response: string of response name (default: None, falls back to constants.RESPONSE)
    output:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    """
    resp_col = response if response is not None else constants.RESPONSE
    fe_df = df.copy()

    if resp_col not in fe_df.columns and "Attrition_Flag" in fe_df.columns:
        fe_df[resp_col] = fe_df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

    # Verify that categorical columns have been encoded
    required_encoded_cols = [
        f"{col}_{resp_col}" for col in constants.CATEGORY_COLUMNS]
    if not all(col in fe_df.columns for col in required_encoded_cols):
        fe_df = encoder_helper(fe_df, constants.CATEGORY_COLUMNS, resp_col)

    y_data = fe_df[resp_col]
    keep_cols = constants.QUANT_COLUMNS + required_encoded_cols
    x_data = fe_df[keep_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=constants.TEST_SIZE,
        random_state=constants.RANDOM_STATE,
    )

    return x_train, x_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """
    Produce classification reports for training and testing results and store in images folder.

    input:
            y_train: training response values
            y_test: test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
            None
    """
    create_output_directories()

    # 1. Random Forest classification report image
    plt.figure(figsize=(8, 7))
    plt.text(
        0.01,
        1.18,
        "Random Forest Train",
        {"fontsize": 10, "fontweight": "bold"},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.65,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 9},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.55,
        "Random Forest Test",
        {"fontsize": 10, "fontweight": "bold"},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.02,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 9},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(constants.RFC_RESULTS_PATH, bbox_inches="tight")
    plt.close()

    # 2. Logistic Regression classification report image
    plt.figure(figsize=(8, 7))
    plt.text(
        0.01,
        1.18,
        "Logistic Regression Train",
        {"fontsize": 10, "fontweight": "bold"},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.65,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 9},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.55,
        "Logistic Regression Test",
        {"fontsize": 10, "fontweight": "bold"},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.02,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 9},
        fontproperties="monospace",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(constants.LOGISTIC_RESULTS_PATH, bbox_inches="tight")
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    """
    Create and store the feature importances plot in output_pth.

    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
            None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(14, 6))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(
        range(x_data.shape[1]),
        feature_names,
        rotation=45,
        ha="right",
    )
    plt.tight_layout()

    target_file = (
        output_pth
        if output_pth.endswith(".png")
        else os.path.join(output_pth, "feature_importances.png")
    )
    plt.savefig(target_file, bbox_inches="tight")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models, store model results (images and scores), and store models.

    input:
            x_train: X training data
            x_test: X testing data
            y_train: y training data
            y_test: y testing data
    output:
            None
    """
    create_output_directories()

    # Initialize models
    rfc = RandomForestClassifier(random_state=constants.RANDOM_STATE)
    logistic_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=constants.LOGISTIC_MAX_ITER,
                    random_state=constants.RANDOM_STATE,
                ),
            ),
        ]
    )

    # Hyperparameter tuning for Random Forest
    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=constants.RFC_PARAM_DIST,
        n_iter=constants.N_ITER_SEARCH,
        cv=constants.CV_FOLDS,
        random_state=constants.RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )

    # Fit models
    cv_rfc.fit(x_train, y_train)
    logistic_pipeline.fit(x_train, y_train)

    best_rfc = cv_rfc.best_estimator_

    # Predict
    y_train_preds_rf = best_rfc.predict(x_train)
    y_test_preds_rf = best_rfc.predict(x_test)

    y_train_preds_lr = logistic_pipeline.predict(x_train)
    y_test_preds_lr = logistic_pipeline.predict(x_test)

    # Save models
    joblib.dump(best_rfc, constants.RFC_MODEL_PATH)
    joblib.dump(logistic_pipeline, constants.LOGISTIC_MODEL_PATH)

    # Plot ROC curves
    fig, axis = plt.subplots(figsize=(12, 7))
    RocCurveDisplay.from_estimator(
        best_rfc,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8,
    )
    RocCurveDisplay.from_estimator(
        logistic_pipeline,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8,
    )
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.tight_layout()
    plt.savefig(constants.ROC_CURVE_PATH, bbox_inches="tight")
    plt.close(fig)

    # Save classification reports as images
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    # Save feature importances
    feature_importance_plot(
        best_rfc,
        x_test,
        constants.FEATURE_IMPORTANCE_PATH)


class ChurnPredictor:
    """
    Object-oriented wrapper class for the customer churn machine learning pipeline.

    Provides a clean, cohesive interface for executing individual stages
    or the complete end-to-end churn prediction pipeline.
    """

    def __init__(self, data_path=constants.DATA_PATH):
        """
        Initialize the churn predictor with a data path.

        input:
                data_path: path to customer data csv
        output:
                None
        """
        self.data_path = data_path
        self.data_frame = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def run_pipeline(self):
        """
        Execute the end-to-end churn prediction pipeline.

        input:
                None
        output:
                None
        """
        create_output_directories()
        self.data_frame = import_data(self.data_path)
        perform_eda(self.data_frame)
        self.data_frame = encoder_helper(
            self.data_frame,
            constants.CATEGORY_COLUMNS,
            constants.RESPONSE,
        )
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = perform_feature_engineering(
            self.data_frame,
            constants.RESPONSE,
        )
        train_models(self.x_train, self.x_test, self.y_train, self.y_test)

    def load_and_predict(self, x_data):
        """
        Load persisted random forest model and generate predictions on new data.

        input:
                x_data: pandas dataframe of features matching model inputs
        output:
                predictions: array of predicted class labels
        """
        rfc_model = joblib.load(constants.RFC_MODEL_PATH)
        return rfc_model.predict(x_data)


if __name__ == "__main__":
    predictor = ChurnPredictor(constants.DATA_PATH)
    predictor.run_pipeline()
