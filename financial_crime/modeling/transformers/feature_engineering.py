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

from financial_crime.config import CURRENCY_MAP


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
        X["ID"] = [str(uuid.uuid4()) for _ in range(len(X))]

        # Add datetime column for Feature Store
        logger.debug("Adding event_timestamp column for Feature Store...")
        X["event_timestamp"] = pd.to_datetime(X["Timestamp"])

        # Add labeler column for Feature Store
        logger.debug("Adding labeler column for Feature Store...")
        X["labeler"] = "mle_team"

        # Currency conversion
        logger.debug("Converting all currencies to USD...")
        X["Amount_Received_USD"] = X.apply(
            lambda row: row["Amount Received"] * CURRENCY_MAP[row["Receiving Currency"]], axis=1
        )
        X["Amount_Paid_USD"] = X.apply(
            lambda row: row["Amount Paid"] * CURRENCY_MAP[row["Payment Currency"]], axis=1
        )

        # Binary feature creation
        logger.debug("Calculating account and bank match indicators...")
        X["Account_Same"] = (X["Account"] == X["Account.1"]).astype(int)
        X["Bank_Same"] = (X["From Bank"] == X["To Bank"]).astype(int)

        logger.debug("Calculating historical account-pair transaction indicators...")
        sort_order = X["event_timestamp"].sort_values().index
        X = X.loc[sort_order].copy()

        # Calculate whether Account has sent money to Account.1 before
        X["account_pair"] = X["Account"].astype(str) + "::" + X["Account.1"].astype(str)
        X["pair_transaction_count"] = X.groupby("account_pair", sort=False).cumcount() + 1
        X["Account_Transacted_With_Account1_Before"] = (X["pair_transaction_count"] > 1).astype(
            int
        )

        # Calculate how often Account makes a given type of Payment (e.g. cash, credit card, etc.)
        payment_cols = [col for col in X.columns if col.startswith('Payment Format_')]
        windows = {
            '10s': 'Tx_Last_10_Sec',
            '30s': 'Tx_Last_30_Sec',
            '1min': 'Tx_Last_1_Min',
            '5min': 'Tx_Last_5_Min',
            '1h': 'Tx_Last_1_Hour',
            '1D': 'Tx_Last_1_Day',
            '10D': 'Tx_Last_10_Days'
        }
        grouped = X.groupby('Account')
        for window_size, window_label in windows.items():
            for pay_col in payment_cols:
                # Create a clean column name (e.g., 'Payment Format_Bitcoin_last_5_min')
                new_col_name = f"{pay_col}_Last_{window_label.replace('Tx_Last_', '')}"
                
                # Calculate the rolling sum of 1s and 0s
                X[new_col_name] = (
                    grouped.rolling(window_size, on='event_timestamp')
                    [pay_col]
                    .sum()  # Use .sum() because 1 + 1 + 0 = 2 transactions of this type
                    .values
                )
        # Calculate the "All Time" number of transactions by Payment Format
        for pay_col in payment_cols:
            X[f"{pay_col}_Tx_All_Time"] = (
                X.groupby('Account')[pay_col]
                .cumsum()
            )

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
