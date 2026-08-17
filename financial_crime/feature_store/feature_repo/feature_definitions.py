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
from feast.types import Float32, Int32, Int64, String

# Define a project for the feature repo
project = Project(name="feature_store", description="A project for driver statistics")

# Define an entity for the driver. You can think of an entity as a primary key used to
# fetch features.
transaction = Entity(name="transaction", join_keys=["ID"], value_type=ValueType.STRING)

# Read data from parquet files. Parquet is convenient for local development mode. For
# production, you can use your favorite DWH, such as BigQuery. See Feast documentation
# for more info.
# TODO: We will use real time as well later via PushSource, but for now we will use a static dataset
# TODO: Understand how we can have a batch and real time source for historical data and real time data
transaction_source = FileSource(
    name="transaction_transactions_source",
    path="../../../data/processed/features.csv",
    timestamp_field="Timestamp",
)

# Here we define a Feature View that will allow us to serve this
# data to our model online.
transaction_fv = FeatureView(
    name="transaction",
    entities=[transaction],
    ttl=timedelta(), # Live forever since this is a static dataset
    schema=[
        Field(name="ID", dtype=String),
        Field(name="Timestamp", dtype=String),
        Field(name="To Bank", dtype=String),
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
    ],
    online=False,
    source=transaction_source
)


# This groups features into a model version
transaction_v1 = FeatureService(
    name="transaction_v1",
    features=[
        transaction_fv, # Selects all features from the feature view
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path="data")
    ),
)

# --- Label Views ---
# Label views manage mutable human labels for training data, RLHF, and evaluation.
# They use PushSources so labels can be submitted from the UI or external tools.

transaction_fraud_labels_source = PushSource(
    name="transaction_fraud_labels_push",
    batch_source=FileSource(
        name="transaction_fraud_labels_batch",
        path="../../../data/processed/labels.csv",
        timestamp_field="Timestamp",
    ),
)

transaction_fraud_labels = LabelView(
    name="transaction_fraud_labels",
    entities=[transaction],
    schema=[
        Field(name="Is_Laundering", dtype=Int32),
        Field(name="labeler", dtype=String),
    ],
    source=transaction_fraud_labels_source,
    labeler_field="labeler",
    conflict_policy=ConflictPolicy.LAST_WRITE_WINS,
    description="Human fraud labels for transactional data - used for model training and evaluation",
    tags={
        "feast.io/labeling-method": "table",
        "feast.io/field-role:is_default": "expectation", # Ground truth labels
    },
)
