"""
Feature engineering for financial crime detection.

This module defines the feature engineering pipeline that transforms raw
transaction data into engineered features. This is used consistently across
training and inference to ensure data consistency.
"""

from pathlib import Path
import pickle

import pandas as pd
from loguru import logger

from financial_crime.config import CURRENCY_MAP


class FeatureEngineer:
    """
    Handles feature engineering transformations on raw transaction data.
    
    NOTE: This transformer is STATELESS - fit() is a no-op for sklearn API compatibility.
    All transformations are deterministic (currency rates, boolean matching) and do not
    require learning from training data. You can safely fit on any data or skip fitting.
    
    Transformations include:
    - Currency conversion to USD
    - Binary feature creation (account/bank matching)
    - Column dropping
    """
    
    def __init__(self):
        """Initialize the feature engineer (no state to initialize)."""
        pass
    
    def fit(self, X: pd.DataFrame) -> "FeatureEngineer":
        """Fit the feature engineer (no-op for this stateless transformer).
        
        Args:
            X: Raw training features DataFrame (unused - for API compatibility)
            
        Returns:
            self for method chaining
        """
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations.
        
        Args:
            X: Raw features DataFrame to transform
            
        Returns:
            Engineered features as DataFrame
        """
        X = X.copy()
        
        # Currency conversion
        logger.debug("Converting all currencies to USD...")
        X["Amount_Received_USD"] = X.apply(
            lambda row: row["Amount Received"] * CURRENCY_MAP[row["Receiving Currency"]], 
            axis=1
        )
        X["Amount_Paid_USD"] = X.apply(
            lambda row: row["Amount Paid"] * CURRENCY_MAP[row["Payment Currency"]], 
            axis=1
        )
        
        # Binary feature creation
        logger.debug("Calculating account and bank match indicators...")
        X["Account_Same"] = (X["Account"] == X["Account.1"]).astype(int)
        X["Bank_Same"] = (X["From Bank"] == X["To Bank"]).astype(int)
        
        # Drop columns that cannot be used for modeling
        cols_to_drop = [
            "Timestamp",
            "From Bank",
            "Account",
            "To Bank",
            "Account.1"
        ]
        logger.debug(f"Dropping columns: {cols_to_drop}")
        X = X.drop(cols_to_drop, axis=1)
        
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
