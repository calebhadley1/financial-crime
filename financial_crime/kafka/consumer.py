from feast import FeatureStore
from kafka import KafkaConsumer, JsonSerializer
import pandas as pd
from tqdm import tqdm
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer
from loguru import logger
from financial_crime.config import MODELS_DIR, PROCESSED_DATA_DIR

consumer = KafkaConsumer('transaction-fraud-detection-topic', value_deserializer=JsonSerializer())

for msg in tqdm(consumer):
    # Ingest message into Feast

    # 1. Perform Feature Engineering using the persisted FeatureEngineer
    logger.info("Loading FeatureEngineer")
    feature_engineer = FeatureEngineer.load(MODELS_DIR / "pipeline_dir" / "feature_engineer.pkl")

    logger.info("Applying FE to holdout set...")
    df = pd.DataFrame([msg.value])
    df_engineered = feature_engineer.transform(df)

    logger.info("Pushing Data into the FS...")
    feature_store = FeatureStore('financial_crime/feature_store/feature_repo')
    feature_store.push("transaction_push_source", df_engineered)