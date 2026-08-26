"""
Feature engineering for financial crime detection.

This module defines the feature engineering pipeline that transforms raw
transaction data into engineered features. This is used consistently across
training and inference to ensure data consistency.
"""

from pathlib import Path
import pickle
import uuid

from loguru import logger
import pandas as pd

from financial_crime.config import (
    ACCOUNT,
    ACCOUNT_1,
    ACCOUNT_PAIR,
    ACCOUNT_SAME,
    ACCOUNT_TRANSACTED_WITH_ACCOUNT_1_BEFORE,
    AMOUNT_PAID,
    AMOUNT_PAID_USD,
    AMOUNT_RECEIVED,
    AMOUNT_RECEIVED_USD,
    BANK_SAME,
    CURRENCY_MAP,
    EVENT_TIMESTAMP,
    FROM_BANK,
    LABELER,
    PAIR_TRANSACTION_COUNT,
    PAYMENT_CURRENCY,
    PAYMENT_FORMAT,
    PAYMENT_FORMATS,
    RECEIVING_CURRENCY,
    TIME_WINDOWS,
    TIMESTAMP,
    TO_BANK,
    TRANSACTION_ID,
)


class FeatureEngineer:
    """
    Handles feature engineering transformations on raw transaction data.

    Transformations are deterministic for a batch, but historical features are calculated
    in timestamp order so each transaction only sees earlier transactions in that batch.

    Transformations include:
    - Currency conversion to USD
    - Binary feature creation (account/bank matching)
    - Historical account-pair transaction indicator
    - Column dropping
    """

    def __init__(self):
        """Initialize the feature engineer."""
        self._is_fitted = False

    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        """Fit the feature engineer.

        Args:
            X: Raw training features DataFrame (used for API compatibility)

        Returns:
            self for method chaining

        Raises:
            ValueError: If X is None or empty
        """
        if X is None or X.empty:
            raise ValueError("Cannot fit FeatureEngineer on None or empty data")
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations.

        Args:
            X: Raw features DataFrame to transform

        Returns:
            Engineered features as DataFrame

        Raises:
            ValueError: If transform is called without fitting first
        """
        if not self._is_fitted:
            raise ValueError(
                "FeatureEngineer must be fitted before transforming. Call fit() first."
            )

        X = X.copy()

        # Add ID column using random UUIDs for unique transaction identification
        logger.debug("Adding ID column for unique transaction identification...")
        X[TRANSACTION_ID] = [str(uuid.uuid4()) for _ in range(len(X))]

        # Add datetime column for Feature Store
        logger.debug("Adding event_timestamp column for Feature Store...")
        X[EVENT_TIMESTAMP] = pd.to_datetime(X[TIMESTAMP])

        # Add labeler column for Feature Store
        logger.debug("Adding labeler column for Feature Store...")
        X[LABELER] = "mle_team"

        # Currency conversion (standardize scale across all currencies)
        logger.debug("Converting all currencies to USD...")
        X[AMOUNT_RECEIVED_USD] = X.apply(
            lambda row: row[AMOUNT_RECEIVED] * CURRENCY_MAP[row[RECEIVING_CURRENCY]], axis=1
        )
        X[AMOUNT_PAID_USD] = X.apply(
            lambda row: row[AMOUNT_PAID] * CURRENCY_MAP[row[PAYMENT_CURRENCY]], axis=1
        )

        # Binary feature creation
        logger.debug("Calculating account and bank match indicators...")
        X[ACCOUNT_SAME] = (X[ACCOUNT] == X[ACCOUNT_1]).astype(int)
        X[BANK_SAME] = (X[FROM_BANK] == X[TO_BANK]).astype(int)

        logger.debug("Calculating historical account-pair transaction indicators...")
        # Sort first to ensure we calculate historical metrics based on previous transactions only
        sort_order = X[EVENT_TIMESTAMP].sort_values().index
        X = X.loc[sort_order].copy()
        # Calculate whether Account has sent money to Account.1 before
        X[ACCOUNT_PAIR] = X[ACCOUNT].astype(str) + "::" + X[ACCOUNT_1].astype(str)
        X[PAIR_TRANSACTION_COUNT] = X.groupby(ACCOUNT_PAIR, sort=False).cumcount() + 1
        X[ACCOUNT_TRANSACTED_WITH_ACCOUNT_1_BEFORE] = (X[PAIR_TRANSACTION_COUNT] > 1).astype(int)

        # Calculate how often Account makes a given type of Payment (e.g. cash, credit card, etc.)
        for payment_format in PAYMENT_FORMATS:
            # 1. Create a temporary numeric series (1 for match, 0 for mismatch)
            # This prevents the DataError by ensuring the rolling data is numeric
            is_matching_type = (X[PAYMENT_FORMAT] == payment_format).astype(int)

            # 2. Re-attach temporarily to a dummy dataframe alongside the grouping keys
            # This ensures groupby and rolling 'on' can still find their columns
            temp_df = X[[ACCOUNT, EVENT_TIMESTAMP]].copy()
            temp_df["is_match"] = is_matching_type

            # 3. Group and run a standard rolling sum
            grouped = temp_df.groupby(ACCOUNT)

            for window_size, window_label in TIME_WINDOWS.items():
                time_suffix = window_label.replace("Tx_Last_", "")
                new_col_name = f"{PAYMENT_FORMAT}_{payment_format}_Last_{time_suffix}"

                # Standard .sum() works seamlessly now that data is numeric
                X[new_col_name] = (
                    grouped.rolling(window_size, on=EVENT_TIMESTAMP)["is_match"].sum().values
                )
        # Calculate the "All Time" number of transactions by Payment Format
        for payment_format in PAYMENT_FORMATS:
            # 1. Create a boolean series (True/1 where it matches, False/0 where it doesn't)
            is_matching_type = (X[PAYMENT_FORMAT] == payment_format).astype(int)
            # 2. Group the 1s and 0s by Account and calculate the cumulative sum
            new_col_name = f"{PAYMENT_FORMAT}_{payment_format}_Tx_All_Time"
            X[new_col_name] = is_matching_type.groupby(X[ACCOUNT]).cumsum()

        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Raw training features DataFrame

        Returns:
            Engineered features as DataFrame
        """
        return self.fit(X).transform(X)

    def save(self, path: Path) -> None:
        """Save feature engineer to disk.

        Args:
            path: Path to save feature engineer pickle file
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: Path) -> "FeatureEngineer":
        """Load feature engineer from disk.

        Args:
            path: Path to feature engineer pickle file

        Returns:
            Loaded FeatureEngineer instance
        """
        with open(path, "rb") as file:
            return pickle.load(file)
