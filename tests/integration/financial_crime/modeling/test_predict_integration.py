from pathlib import Path
import shutil
import subprocess

from feast import FeatureStore
import numpy as np
import pandas as pd
from tests.factory.feature_factory import EngineeredFeatureFactory

from financial_crime import features
from financial_crime.modeling import predict, train


def make_raw_data() -> pd.DataFrame:
    engineered_features = [EngineeredFeatureFactory.build(ID=str(value)) for value in range(24)]
    data = pd.DataFrame(
        [feature.model_dump(mode="python", by_alias=True) for feature in engineered_features]
    )
    data["Is Laundering"] = [0, 1] * 12
    return data.drop(
        columns=[
            "ID",
            "event_timestamp",
            "labeler",
            "Amount_Received_USD",
            "Amount_Paid_USD",
            "Account_Same",
            "Bank_Same",
        ]
    )


def make_feature_repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "financial_crime" / "feature_store" / "feature_repo"
    repo_path.mkdir(parents=True)
    source_repo = Path("financial_crime/feature_store/feature_repo")
    shutil.copy2(source_repo / "feature_definitions.py", repo_path)
    shutil.copy2(source_repo / "feature_store.yaml", repo_path)

    subprocess.run(["feast", "apply"], cwd=repo_path, check=True)
    return repo_path


def test_main_predicts_from_trained_pipeline(tmp_path, monkeypatch):
    raw_data = make_raw_data()
    input_path = tmp_path / "data" / "processed" / "dataset.csv"
    features_path = tmp_path / "data" / "processed" / "features.parquet"
    labels_path = tmp_path / "data" / "processed" / "labels.parquet"
    pipeline_dir = tmp_path / "pipeline"
    predictions_path = tmp_path / "predictions.csv"
    input_path.parent.mkdir(parents=True)
    pipeline_dir.mkdir()
    raw_data.to_csv(input_path, index=False)

    features.main(
        input_path=input_path,
        output_features_path=features_path,
        output_labels_path=labels_path,
        output_feature_engineer_path=pipeline_dir / "feature_engineer.pkl",
    )
    feature_repo_path = make_feature_repo(tmp_path)

    monkeypatch.setattr(train, "MODEL_N_ESTIMATORS", 2)
    monkeypatch.setattr(train, "MODEL_MAX_DEPTH", 1)
    monkeypatch.setattr(train, "TEST_SIZE", 0.25)
    monkeypatch.setattr(train, "SAMPLING_STRATEGY", 1.0)
    train.main(
        features_path=features_path,
        feature_repo_path=feature_repo_path,
        pipeline_dir=pipeline_dir,
    )

    monkeypatch.setattr(predict, "FeatureStore", lambda _: FeatureStore(str(feature_repo_path)))
    predict.main(
        input_path=input_path,
        pipeline_dir=pipeline_dir,
        predictions_path=predictions_path,
    )

    predictions = np.loadtxt(predictions_path, delimiter=",")

    assert predictions.shape == (len(raw_data),)
    assert set(predictions).issubset({0, 1})