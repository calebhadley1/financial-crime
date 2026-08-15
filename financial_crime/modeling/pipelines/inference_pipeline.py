"""
Complete ML pipeline for financial crime prediction.

This module provides a unified interface for the complete ML pipeline:
feature engineering → preprocessing → model prediction.
Used for both training and inference to ensure consistency.
"""

from pathlib import Path
import pickle
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer
from financial_crime.modeling.transformers.feature_preprocessing import FeaturePreprocessor


class InferencePipeline:
    """
    Complete ML inference pipeline combining all transformation steps.
    
    This pipeline ensures that raw data goes through the same sequence of
    transformations during both training and inference, guaranteeing consistency.
    """
    
    def __init__(
        self,
        feature_engineer: FeatureEngineer,
        preprocessor: FeaturePreprocessor,
        model
    ):
        """Initialize the pipeline with all components.
        
        Args:
            feature_engineer: FeatureEngineer instance (fitted)
            preprocessor: FeaturePreprocessor instance (fitted)
            model: Trained sklearn model
        """
        self.feature_engineer = feature_engineer
        self.preprocessor = preprocessor
        self.model = model
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions on raw data.
        
        Applies the complete transformation pipeline:
        1. Feature engineering (currency conversion, feature creation)
        2. Preprocessing (categorical encoding, scaling)
        3. Model prediction
        
        Args:
            X: Raw features DataFrame
            
        Returns:
            Model predictions as numpy array
        """
        logger.debug(f"Predicting on {len(X)} samples")
        
        # Step 1: Feature engineering
        logger.debug("Step 1: Applying feature engineering...")
        X_engineered = self.feature_engineer.transform(X)
        
        # Step 2: Preprocessing
        logger.debug("Step 2: Applying preprocessing...")
        X_preprocessed = self.preprocessor.transform(X_engineered)
        
        # Step 3: Model prediction
        logger.debug("Step 3: Running model inference...")
        predictions = self.model.predict(X_preprocessed)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions on raw data.
        
        Only works if the underlying model supports predict_proba.
        
        Args:
            X: Raw features DataFrame
            
        Returns:
            Model probability predictions as numpy array
            
        Raises:
            AttributeError: If model doesn't support predict_proba
        """
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(
                f"Model {type(self.model).__name__} does not support predict_proba"
            )
        
        logger.debug(f"Predicting probabilities on {len(X)} samples")
        
        # Apply transformations
        X_engineered = self.feature_engineer.transform(X)
        X_preprocessed = self.preprocessor.transform(X_engineered)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_preprocessed)
        
        return probabilities
    
    def save(self, path: Path) -> None:
        """Save the complete pipeline to disk.
        
        Args:
            path: Directory path to save pipeline components
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving pipeline to {path}")
        
        # Save each component
        self.feature_engineer.save(path / "feature_engineer.pkl")
        self.preprocessor.save(path / "preprocessor.pkl")
        
        with open(path / "model.pkl", "wb") as file:
            pickle.dump(self.model, file)
        
        logger.success("Pipeline saved successfully")
    
    @staticmethod
    def load(path: Path) -> "InferencePipeline":
        """Load a complete pipeline from disk.
        
        Args:
            path: Directory path containing pipeline components
            
        Returns:
            Loaded InferencePipeline instance
        """
        path = Path(path)
        
        logger.info(f"Loading pipeline from {path}")
        
        # Load each component
        feature_engineer = FeatureEngineer.load(path / "feature_engineer.pkl")
        preprocessor = FeaturePreprocessor.load(path / "preprocessor.pkl")
        
        with open(path / "model.pkl", "rb") as file:
            model = pickle.load(file)
        
        logger.success("Pipeline loaded successfully")
        
        return InferencePipeline(feature_engineer, preprocessor, model)
