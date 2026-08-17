import json
from time import sleep
from kafka import KafkaProducer
import pandas as pd
from financial_crime.config import PROCESSED_DATA_DIR
from loguru import logger
from tqdm import tqdm

# We will simulate real-time transactions by reading a static Parquet
# file and sending each row to the kafka topic. In a real-world scenario, 
# you would have a stream of transactions coming in from a source like a payment gateway or a banking system.
holdout_set_path = PROCESSED_DATA_DIR / "dataset_50k.parquet"
logger.info(f"Reading holdout set from {holdout_set_path}")
df = pd.read_parquet(holdout_set_path)

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=["localhost:9092"],
    # value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# Stream rows to Kafka topic
topic_name = "transaction-fraud-detection-topic"
logger.info(f"Sending rows to Kafka topic: {topic_name}")

for record in tqdm(df.to_dict(orient="records"), desc="Sending rows to Kafka"):
  producer.send(topic_name, value=record)
  # Sleep so we don't rip through all rows too quickly. ~13 min at 1 tps for 50k dataset
  sleep(seconds=1)

producer.flush()
producer.close()
logger.info(f"Sent {len(df)} rows to Kafka.")
