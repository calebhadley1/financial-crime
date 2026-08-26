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
        payment_formats = [
            "ACH",
            "Bitcoin",
            "Cash",
            "Cheque",
            "Credit Card",
            "Reinvestment",
            "Wire",
        ]
        windows = {
            "10s": "Tx_Last_10_Sec",
            "30s": "Tx_Last_30_Sec",
            "1min": "Tx_Last_1_Min",
            "5min": "Tx_Last_5_Min",
            "1h": "Tx_Last_1_Hour",
            "1D": "Tx_Last_1_Day",
            "10D": "Tx_Last_10_Days",
        }
        for payment_format in payment_formats:
            # 1. Create a temporary numeric series (1 for match, 0 for mismatch)
            # This prevents the DataError by ensuring the rolling data is numeric
            is_matching_type = (X["Payment Format"] == payment_format).astype(int)

            # 2. Re-attach temporarily to a dummy dataframe alongside the grouping keys
            # This ensures groupby and rolling 'on' can still find their columns
            temp_df = X[["Account", "event_timestamp"]].copy()
            temp_df["is_match"] = is_matching_type

            # 3. Group and run a standard rolling sum
            grouped = temp_df.groupby("Account")

            for window_size, window_label in windows.items():
                time_suffix = window_label.replace("Tx_Last_", "")
                new_col_name = f"Payment Format_{payment_format}_Last_{time_suffix}"

                # Standard .sum() works seamlessly now that data is numeric
                X[new_col_name] = (
                    grouped.rolling(window_size, on="event_timestamp")["is_match"].sum().values
                )
        # Calculate the "All Time" number of transactions by Payment Format
        for payment_format in payment_formats:
            # 1. Create a boolean series (True/1 where it matches, False/0 where it doesn't)
            is_matching_type = (X["Payment Format"] == payment_format).astype(int)
            # 2. Group the 1s and 0s by Account and calculate the cumulative sum
            new_col_name = f"Payment Format_{payment_format}_Tx_All_Time"
            X[new_col_name] = is_matching_type.groupby(X["Account"]).cumsum()

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
