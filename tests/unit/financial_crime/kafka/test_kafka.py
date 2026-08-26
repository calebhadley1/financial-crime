import os
import uuid

import pytest

from kafka import KafkaConsumer, KafkaProducer

pytestmark = pytest.mark.integration


def test_kafka_round_trip():
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if not bootstrap_servers:
        pytest.skip("Set KAFKA_BOOTSTRAP_SERVERS to run the kafka-python broker test")

    topic = f"financial-crime-test-{uuid.uuid4()}"
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        consumer_timeout_ms=10000,
        group_id=f"financial-crime-test-{uuid.uuid4()}",
    )
    try:
        producer.send(topic, b"transaction").get(timeout=10)
        producer.flush()
        message = next(consumer)
        assert message.value == b"transaction"
    finally:
        consumer.close()
        producer.close()
