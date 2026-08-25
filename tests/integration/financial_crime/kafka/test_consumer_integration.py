from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import subprocess

from feast import FeatureStore
import pandas as pd
from tests.factory.feature_factory import EngineeredFeatureFactory

from financial_crime.kafka import consumer
from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer


def make_feature_repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "financial_crime" / "feature_store" / "feature_repo"
    repo_path.mkdir(parents=True)
    source_repo = Path("financial_crime/feature_store/feature_repo")
    shutil.copy2(source_repo / "feature_definitions.py", repo_path)
    shutil.copy2(source_repo / "feature_store.yaml", repo_path)

    historical_row = EngineeredFeatureFactory.build(
        ID="historical-id",
        event_timestamp=datetime.now(UTC) - timedelta(days=1),
        account_pair="8000EBD30::8000EBD31",
        pair_transaction_count=1,
        Account_Transacted_With_Account1_Before=0,
    )
    features = pd.DataFrame([historical_row.model_dump(mode="python", by_alias=True)])
    processed_data_path = tmp_path / "data" / "processed"
    processed_data_path.mkdir(parents=True)
    features.to_parquet(processed_data_path / "features.parquet", index=False)

    subprocess.run(["feast", "apply"], cwd=repo_path, check=True)
    return repo_path


def test_consumer_uses_materialized_pair_history_for_new_transaction(tmp_path, monkeypatch):
    feature_repo_path = make_feature_repo(tmp_path)
    feature_store = FeatureStore(str(feature_repo_path))
    feature_store.materialize_incremental(
        end_date=datetime.now(UTC), feature_views=["account_pair_history"]
    )
    account_pair_service = feature_store.get_feature_service("account_pair_v1")
    feature_engineer = FeatureEngineer().fit(
        pd.DataFrame(
            {
                "Timestamp": ["2022/01/02 00:00"],
                "Account": ["8000EBD30"],
                "Account.1": ["8000EBD31"],
                "From Bank": ["10"],
                "To Bank": ["10"],
                "Amount Received": [3697.34],
                "Receiving Currency": ["US Dollar"],
                "Amount Paid": [3697.34],
                "Payment Currency": ["US Dollar"],
                "Payment Format": ["Wire"],
            }
        )
    )
    monkeypatch.setattr(consumer.requests, "post", lambda *args, **kwargs: None)

    engineered = consumer.process_message(
        {
            "Timestamp": "2022/01/02 00:00",
            "Account": "8000EBD30",
            "Account.1": "8000EBD31",
            "From Bank": "10",
            "To Bank": "10",
            "Amount Received": 3697.34,
            "Receiving Currency": "US Dollar",
            "Amount Paid": 3697.34,
            "Payment Currency": "US Dollar",
            "Payment Format": "Wire",
        },
        feature_store,
        account_pair_service,
        feature_engineer,
        "http://api.test",
    )

    assert engineered["Account_Transacted_With_Account1_Before"].iloc[0] == 1
    assert engineered["pair_transaction_count"].iloc[0] == 2