from datetime import timedelta

from feast import (
    ConflictPolicy,
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    LabelView,
    Project,
    PushSource,
    ValueType,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.types import Float32, Int32, Int64, String, UnixTimestamp

# Define a project for the feature repo
project = Project(name="feature_store", description="A project for driver statistics")

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
transaction = Entity(name="transaction", join_keys=["ID"], value_type=ValueType.STRING)
account_pair = Entity(name="account_pair", join_keys=["account_pair"], value_type=ValueType.STRING)

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
# TODO: We will use real time as well later via PushSource, but for now we will use a static dataset
# TODO: Understand how we can have a batch and real time source for historical data and real time data
transaction_source = FileSource(
    name="transaction_source",
    path="../../../data/processed/features.parquet",
    timestamp_field="event_timestamp",
)

# For real time streaming, we will use a PushSource. This allows us to push data into Feast from an external source,
# such as a Kafka topic or a REST API.
transaction_push_source = PushSource(
    name="transaction_push_source",
    batch_source=transaction_source,
)

# Here we define a Feature View that will allow us to serve this
# data to our model online.
transaction_fv = FeatureView(
    name="transaction",
    entities=[transaction],
    ttl=timedelta(),  # Live forever since this is a static dataset
    schema=[
        Field(name="ID", dtype=String),
        Field(name="event_timestamp", dtype=UnixTimestamp),
        Field(name="Timestamp", dtype=String),
        Field(name="To Bank", dtype=String),
        Field(name="From Bank", dtype=String),
        Field(name="Account", dtype=String),
        Field(name="Account.1", dtype=String),
        Field(name="Amount Received", dtype=Float32),
        Field(name="Receiving Currency", dtype=String),
        Field(name="Amount Paid", dtype=Float32),
        Field(name="Payment Currency", dtype=String),
        Field(name="Payment Format", dtype=String),
        Field(name="Amount_Received_USD", dtype=Float32),
        Field(name="Amount_Paid_USD", dtype=Float32),
        Field(name="Account_Same", dtype=Int64),
        Field(name="Bank_Same", dtype=Int64),
        Field(name="Account_Transacted_With_Account1_Before", dtype=Int64),
        Field(name="account_pair", dtype=String),
        Field(name="pair_transaction_count", dtype=Int64),
    ],
    online=True,
    source=transaction_push_source,
)

# --- Label Views ---
# Label views manage mutable human labels for training data, RLHF, and evaluation.
# They use PushSources so labels can be submitted from the UI or external tools.

transaction_fraud_labels_source = PushSource(
    name="transaction_fraud_labels_push",
    batch_source=FileSource(
        name="transaction_fraud_labels_batch",
        path="../../../data/processed/labels.parquet",
        timestamp_field="event_timestamp",
    ),
)

transaction_fraud_labels = LabelView(
    name="transaction_fraud_labels",
    entities=[transaction],
    schema=[
        Field(name="Is Laundering", dtype=Int32),
        Field(name="labeler", dtype=String),
    ],
    source=transaction_fraud_labels_source,
    labeler_field="labeler",
    conflict_policy=ConflictPolicy.LAST_WRITE_WINS,
    description="Human fraud labels for transactional data - used for model training and evaluation",
    tags={
        "feast.io/labeling-method": "table",
        "feast.io/field-role:is_default": "expectation",  # Ground truth labels
    },
)

# This groups features into a model version
transaction_v1 = FeatureService(
    name="transaction_v1",
    features=[
        transaction_fv,  # Selects all features from the feature view
        transaction_fraud_labels,  # Include labels for training
    ],
    logging_config=LoggingConfig(destination=FileLoggingDestination(path="data")),
)

account_pair_history = FeatureView(
    name="account_pair_history",
    entities=[account_pair],
    ttl=timedelta(),
    schema=[
        Field(name="account_pair", dtype=String),
        Field(name="pair_transaction_count", dtype=Int64),
    ],
    online=True,
    source=transaction_push_source,
)

account_pair_v1 = FeatureService(
    name="account_pair_v1",
    features=[account_pair_history],
)
