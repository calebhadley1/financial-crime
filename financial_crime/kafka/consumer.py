import os
from feast import FeatureStore
from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm

from financial_crime.config import MODELS_DIR
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer
from kafka import JsonSerializer, KafkaConsumer

# Get Kafka broker from environment variable
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
api_url = os.getenv('API_URL', 'http://localhost:8000')

consumer = KafkaConsumer(
    "transaction-fraud-detection-topic",
    bootstrap_servers=kafka_bootstrap_servers,
    value_deserializer=JsonSerializer()
)

for msg in tqdm(consumer):
    # Ingest message into Feast

    # 1. Perform Feature Engineering using the persisted FeatureEngineer
    logger.info("Loading FeatureEngineer")
    feature_engineer = FeatureEngineer.load(MODELS_DIR / "pipeline_dir" / "feature_engineer.pkl")

    logger.info("Applying FE to holdout set...")
    df = pd.DataFrame([msg.value])
    df_engineered = feature_engineer.transform(df)

    # 2. Load data into the online Feature Store
    logger.info("Pushing Data into the FS...")
    feature_store = FeatureStore("financial_crime/feature_store/feature_repo")
    feature_store.push("transaction_push_source", df_engineered)

    # 3. Coerce message and request the API to make a prediction
    logger.info("Requesting API fraud detection...")
    json = {
        "raw_features": df_engineered.to_dict(orient='records')
    }
    requests.post(f'{api_url}/predict', json=json)