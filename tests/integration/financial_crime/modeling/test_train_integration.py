from datetime import UTC, datetime, timedelta
from pathlib import Path
import shutil
import subprocess

import pandas as pd
from tests.factory.feature_factory import EngineeredFeatureFactory

from financial_crime.modeling import train
from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline


def make_training_data():
    features = [
        EngineeredFeatureFactory.build(
            ID=str(value),
            event_timestamp=datetime(2022, 1, 1, tzinfo=UTC) + timedelta(days=value),
        )
        for value in range(24)
    ]
    data = pd.DataFrame(
        [feature.model_dump(mode="python", by_alias=True) for feature in features]
    )
    data["Is Laundering"] = [0, 1] * 12
    return data


def make_feature_repo(tmp_path, training_data):
    repo_path = tmp_path / "financial_crime" / "feature_store" / "feature_repo"
    repo_path.mkdir(parents=True)
    source_repo = Path("financial_crime/feature_store/feature_repo")
    shutil.copy2(source_repo / "feature_definitions.py", repo_path)
    shutil.copy2(source_repo / "feature_store.yaml", repo_path)

    processed_data_path = tmp_path / "data" / "processed"
    processed_data_path.mkdir(parents=True)
    training_data.drop(columns=["Is Laundering", "labeler"]).to_parquet(
        processed_data_path / "features.parquet", index=False
    )
    training_data[["ID", "event_timestamp", "Is Laundering", "labeler"]].to_parquet(
        processed_data_path / "labels.parquet", index=False
    )

    subprocess.run(["feast", "apply"], cwd=repo_path, check=True)
    return repo_path, processed_data_path / "features.parquet"


def test_main_trains_and_persists_usable_pipeline(tmp_path, monkeypatch):
    training_data = make_training_data()
    pipeline_dir = tmp_path / "pipeline"
    feature_repo_path, features_path = make_feature_repo(tmp_path, training_data)

    monkeypatch.setattr(train, "MODEL_N_ESTIMATORS", 2)
    monkeypatch.setattr(train, "MODEL_MAX_DEPTH", 1)
    monkeypatch.setattr(train, "TEST_SIZE", 0.25)
    monkeypatch.setattr(train, "SAMPLING_STRATEGY", 1.0)

    train.main(
        features_path=features_path,
        feature_repo_path=feature_repo_path,
        pipeline_dir=pipeline_dir,
    )

    assert (pipeline_dir / "preprocessor.pkl").exists()
    assert (pipeline_dir / "model.pkl").exists()

    pipeline = InferencePipeline.load(pipeline_dir)
    predictions = pipeline.predict(training_data.drop(columns=["Is Laundering"]))

    assert predictions.shape == (len(training_data),)