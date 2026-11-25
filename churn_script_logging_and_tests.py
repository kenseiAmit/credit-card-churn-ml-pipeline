"""
Unit Tests and Execution Logging for Customer Churn Prediction Library.

This module validates all data pipeline and modeling routines defined in
`churn_library.py`. Test outcomes, success confirmations, and error diagnostics
are written to `./logs/churn_library.log` using Python's logging facility.
Tests can be executed via pytest or directly from the CLI.

Author: Amit Kumar
Date: October 2026
"""

import logging
import os
import pytest

import churn_library as cls
import constants

# Ensure output and logging directories exist
cls.create_output_directories()

# Configure logging to write to log file
logging.basicConfig(
    filename=constants.LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
LOGGER = logging.getLogger()
if not any(isinstance(handler, logging.FileHandler)
           for handler in LOGGER.handlers):
    _file_handler = logging.FileHandler(constants.LOG_FILE, mode="w")
    _file_handler.setLevel(logging.INFO)
    _file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    LOGGER.addHandler(_file_handler)


@pytest.fixture(name="import_data")
def fixture_import_data():
    """Pytest fixture providing import_data function."""
    return cls.import_data


@pytest.fixture(name="perform_eda")
def fixture_perform_eda():
    """Pytest fixture providing perform_eda function."""
    return cls.perform_eda


@pytest.fixture(name="encoder_helper")
def fixture_encoder_helper():
    """Pytest fixture providing encoder_helper function."""
    return cls.encoder_helper


@pytest.fixture(name="perform_feature_engineering")
def fixture_perform_feature_engineering():
    """Pytest fixture providing perform_feature_engineering function."""
    return cls.perform_feature_engineering


@pytest.fixture(name="train_models")
def fixture_train_models():
    """Pytest fixture providing train_models function."""
    return cls.train_models


def test_import(import_data):
    """
    Test the import_data function for successful loading and exception handling.

    input:
            import_data: reference to import_data function
    output:
            None
    """
    # 1. Test error handling when file does not exist
    try:
        import_data("./data/non_existent_file.csv")
    except FileNotFoundError as err:
        logging.error(
            "TEST ERROR (handled): test_import correctly caught missing file: %s",
            err)

    # 2. Test successful data loading
    data_frame = None
    try:
        data_frame = import_data(constants.DATA_PATH)
        logging.info(
            "SUCCESS: test_import loaded data from %s",
            constants.DATA_PATH)
    except FileNotFoundError as err:
        logging.error(
            "ERROR: test_import could not find file at %s: %s",
            constants.DATA_PATH,
            err)
        raise err

    try:
        assert data_frame is not None
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
        logging.info(
            "SUCCESS: test_import verified non-empty dataframe with shape %s",
            data_frame.shape,
        )
    except AssertionError as err:
        logging.error(
            "ERROR: test_import dataframe is empty or malformed: %s", err)
        raise err


def test_eda(perform_eda):
    """
    Test the perform_eda function and assert all expected plot images are generated.

    input:
            perform_eda: reference to perform_eda function
    output:
            None
    """
    expected_eda_files = [
        constants.CHURN_DIST_PATH,
        constants.CUSTOMER_AGE_DIST_PATH,
        constants.MARITAL_STATUS_DIST_PATH,
        constants.TOTAL_TRANS_CT_DIST_PATH,
        constants.HEATMAP_PATH,
    ]

    try:
        data_frame = cls.import_data(constants.DATA_PATH)
        perform_eda(data_frame)
        for file_path in expected_eda_files:
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0
        logging.info(
            "SUCCESS: test_eda generated and verified all %d EDA plots",
            len(expected_eda_files))
    except AssertionError as err:
        logging.error(
            "ERROR: test_eda assertion failed - output image missing or empty: %s",
            err)
        raise err
    except Exception as err:
        logging.error(
            "ERROR: test_eda encountered an unexpected error: %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    """
    Test the encoder_helper function and assert encoded columns exist with valid values.

    input:
            encoder_helper: reference to encoder_helper function
    output:
            None
    """
    try:
        data_frame = cls.import_data(constants.DATA_PATH)
        encoded_df = encoder_helper(
            data_frame,
            constants.CATEGORY_COLUMNS,
            constants.RESPONSE,
        )

        for category in constants.CATEGORY_COLUMNS:
            encoded_col = f"{category}_{constants.RESPONSE}"
            assert encoded_col in encoded_df.columns
            assert not encoded_df[encoded_col].isnull().all()
            assert encoded_df[encoded_col].shape[0] == data_frame.shape[0]

        logging.info(
            "SUCCESS: test_encoder_helper successfully encoded all %d categorical features",
            len(constants.CATEGORY_COLUMNS),
        )
    except AssertionError as err:
        logging.error("ERROR: test_encoder_helper assertion failed: %s", err)
        raise err
    except Exception as err:
        logging.error(
            "ERROR: test_encoder_helper encountered an unexpected error: %s",
            err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering to verify train/test split shapes and non-empty outputs.

    input:
            perform_feature_engineering: reference to perform_feature_engineering function
    output:
            None
    """
    try:
        data_frame = cls.import_data(constants.DATA_PATH)
        data_frame = cls.encoder_helper(
            data_frame,
            constants.CATEGORY_COLUMNS,
            constants.RESPONSE,
        )
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            data_frame,
            constants.RESPONSE,
        )

        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        assert x_train.shape[1] == len(constants.KEEP_COLUMNS)
        assert x_test.shape[1] == len(constants.KEEP_COLUMNS)

        logging.info(
            "SUCCESS: test_perform_feature_engineering verified train (%s) and test (%s) splits",
            x_train.shape,
            x_test.shape,
        )
    except AssertionError as err:
        logging.error(
            "ERROR: test_perform_feature_engineering assertion failed: %s",
            err)
        raise err
    except Exception as err:
        logging.error(
            "ERROR: test_perform_feature_engineering encountered an error: %s",
            err)
        raise err


def test_train_models(train_models):
    """
    Test model training, serialized model files, and evaluation result images.

    input:
            train_models: reference to train_models function
    output:
            None
    """
    expected_artifacts = [
        constants.RFC_MODEL_PATH,
        constants.LOGISTIC_MODEL_PATH,
        constants.ROC_CURVE_PATH,
        constants.RFC_RESULTS_PATH,
        constants.LOGISTIC_RESULTS_PATH,
        constants.FEATURE_IMPORTANCE_PATH,
    ]

    try:
        data_frame = cls.import_data(constants.DATA_PATH)
        data_frame = cls.encoder_helper(
            data_frame,
            constants.CATEGORY_COLUMNS,
            constants.RESPONSE,
        )
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            data_frame,
            constants.RESPONSE,
        )
        train_models(x_train, x_test, y_train, y_test)

        for artifact_path in expected_artifacts:
            assert os.path.exists(artifact_path)
            assert os.path.getsize(artifact_path) > 0

        logging.info(
            "SUCCESS: test_train_models generated and verified all %d model artifacts and plots",
            len(expected_artifacts),
        )
    except AssertionError as err:
        logging.error("ERROR: test_train_models assertion failed: %s", err)
        raise err
    except Exception as err:
        logging.error(
            "ERROR: test_train_models encountered an unexpected error: %s",
            err)
        raise err


if __name__ == "__main__":
    cls.create_output_directories()

    logging.info("START: Commencing customer churn test suite execution")
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
    logging.info("COMPLETED: All tests finished successfully")

    print("Tests completed successfully. Check logs/churn_library.log for details.")
