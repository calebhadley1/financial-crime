import os
from time import sleep

from loguru import logger
import pandas as pd
from tqdm import tqdm

from financial_crime.config import PROCESSED_DATA_DIR
from kafka import JsonSerializer, KafkaProducer

# We will simulate real-time transactions by reading a static Parquet
# file and sending each row to the kafka topic. In a real-world scenario,
# you would have a stream of transactions coming in from a source like a payment gateway or a banking system.
holdout_set_path = PROCESSED_DATA_DIR / "dataset_50k.csv"
logger.info(f"Reading holdout set from {holdout_set_path}")
df = pd.read_csv(holdout_set_path)

# Get Kafka broker from environment variable
kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092").split(",")

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=kafka_bootstrap_servers, value_serializer=JsonSerializer()
)

# Stream rows to Kafka topic
topic_name = "transaction-fraud-detection-topic"
logger.info(f"Sending rows to Kafka topic: {topic_name}")

for record in tqdm(df.to_dict(orient="records"), desc="Sending rows to Kafka"):
    producer.send(topic_name, value=record)
    # Sleep so we don't rip through all rows too quickly. ~13 min at 1 tps for 50k dataset
    sleep(5)

producer.flush()
producer.close()
logger.info(f"Sent {len(df)} rows to Kafka.")
