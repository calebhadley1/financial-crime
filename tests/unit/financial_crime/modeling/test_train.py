import pandas as pd

import financial_crime.modeling.train as train


class FakeHistoricalFeatures:
    def to_df(self):
        return pd.DataFrame({"ID": ["a"], "event_timestamp": ["2022-01-01"], "feature": [1], "Is Laundering": [0]})


class FakeStore:
    def get_feature_service(self, name):
        return "service"

    def get_historical_features(self, features, entity_df):
        return FakeHistoricalFeatures()


class FakePipeline:
    def train(self, X, y, model):
        assert X.columns.tolist() == ["ID", "event_timestamp", "feature"]
        assert y.columns.tolist() == ["Is Laundering"]
        return self

    def save(self, path):
        self.path = path


def test_main_retrieves_features_trains_and_saves(tmp_path, monkeypatch):
    features_path = tmp_path / "features.parquet"
    pd.DataFrame({"ID": ["a"], "event_timestamp": ["2022-01-01"]}).to_parquet(features_path)
    pipeline = FakePipeline()
    monkeypatch.setattr(train, "FeatureStore", lambda path: FakeStore())
    monkeypatch.setattr(train, "TrainingPipeline", lambda **kwargs: pipeline)

    train.main(features_path=features_path, pipeline_dir=tmp_path / "pipeline")

    assert pipeline.path == tmp_path / "pipeline"