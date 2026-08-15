"""
Preprocessing pipeline for financial crime feature engineering.

This module defines and manages the preprocessing pipeline (encoding and scaling)
used consistently across training and inference.
"""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from financial_crime.config import CATEGORICAL_FEATURES


class FeaturePreprocessor:
    """
    Handles feature preprocessing: categorical encoding and numerical scaling.
    
    This class wraps the sklearn preprocessing pipeline to provide a clear contract
    for what transformations are applied during training and inference.
    """
    
    def __init__(self):
        """Initialize the preprocessor with column transformation rules."""
        self.preprocessor = make_column_transformer(
            (OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
            remainder="passthrough"
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit the preprocessor on training data.
        
        Args:
            X: Training features DataFrame
            
        Returns:
            self for method chaining
        """
        # Validate input schema
        validate_feature_schema(X)
        
        self.preprocessor.fit(X)
        self.scaler.fit(self.preprocessor.transform(X))
        self._is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to features.
        
        Args:
            X: Features DataFrame to transform
            
        Returns:
            Transformed features as DataFrame
            
        Raises:
            ValueError: If preprocessor hasn't been fitted yet
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming. Call fit() first.")
        
        X_encoded = self.preprocessor.transform(X)
        X_scaled = self.scaler.transform(X_encoded)
        return pd.DataFrame(X_scaled)
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            X: Training features DataFrame
            
        Returns:
            Transformed features as DataFrame
        """
        return self.fit(X).transform(X)
    
    def save(self, path: Path) -> None:
        """Save preprocessor to disk.
        
        Args:
            path: Path to save preprocessor pickle file
        """
        with open(path, "wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(path: Path) -> "FeaturePreprocessor":
        """Load preprocessor from disk.
        
        Args:
            path: Path to preprocessor pickle file
            
        Returns:
            Loaded FeaturePreprocessor instance
        """
        with open(path, "rb") as file:
            return pickle.load(file)


def validate_feature_schema(X: pd.DataFrame) -> None:
    """Validate that features match expected schema.
    
    Args:
        X: Features DataFrame to validate
        
    Raises:
        ValueError: If features don't match expected schema
    """
    # Check for required categorical features
    missing_features = set(CATEGORICAL_FEATURES) - set(X.columns)
    if missing_features:
        raise ValueError(
            f"Missing required categorical features: {missing_features}. "
            f"Available features: {list(X.columns)}"
        )
    
    # Check data types for categorical features
    for col in CATEGORICAL_FEATURES:
        if X[col].dtype == "object" or X[col].dtype.name.startswith("string"):
            continue
        # Allow numeric types for categorical features (will be encoded)
