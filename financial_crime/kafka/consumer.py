from datetime import UTC, datetime
import os

from feast import FeatureStore
from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm

from financial_crime.config import MODELS_DIR
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer
from kafka import JsonSerializer, KafkaConsumer


def process_message(message, feature_store, account_pair_service, feature_engineer, api_url):
    """Engineer, score, and publish one Kafka transaction."""
    # Ingest message into Feast

    # 1. Perform Feature Engineering using the persisted FeatureEngineer
    logger.info("Applying FE to holdout set...")
    df = pd.DataFrame([message])
    df_engineered = feature_engineer.transform(df)

    # TODO add the other new feature engineering
    pair = df_engineered.iloc[0]["account_pair"]
    pair_history = feature_store.get_online_features(
        features=account_pair_service,
        entity_rows=[{"account_pair": pair}],
    ).to_df()
    raw_count = pair_history.iloc[0]["pair_transaction_count"]
    previous_count = 0 if pd.isna(raw_count) else int(raw_count)
    df_engineered["Account_Transacted_With_Account1_Before"] = int(previous_count > 0)
    df_engineered["pair_transaction_count"] = previous_count + 1

    # 2. Load data into the online Feature Store
    logger.info("Pushing Data into the FS...")
    feature_store.push("transaction_push_source", df_engineered)

    # 3. Coerce message and request the API to make a prediction
    dt_cols = df_engineered.select_dtypes(include=["datetime64", "datetimetz"]).columns
    df_engineered[dt_cols] = df_engineered[dt_cols].astype(str)
    logger.info("Requesting API fraud detection...")
    json = df_engineered[["ID", "event_timestamp"]].to_dict(orient="records")
    requests.post(f"{api_url}/predict", json=json)
    return df_engineered


def main():
    """Consume transactions and maintain online account-pair history."""
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    api_url = os.getenv("API_URL", "http://localhost:8000")

    consumer = KafkaConsumer(
        "transaction-fraud-detection-topic",
        bootstrap_servers=kafka_bootstrap_servers,
        value_deserializer=JsonSerializer(),
    )
    feature_store = FeatureStore("financial_crime/feature_store/feature_repo")
    account_pair_service = feature_store.get_feature_service("account_pair_v1")
    feature_store.materialize_incremental(
        end_date=datetime.now(UTC), feature_views=["account_pair_history"]
    )
    feature_engineer = FeatureEngineer.load(MODELS_DIR / "pipeline" / "feature_engineer.pkl")

    for msg in tqdm(consumer):
        process_message(msg.value, feature_store, account_pair_service, feature_engineer, api_url)


if __name__ == "__main__":
    main()
