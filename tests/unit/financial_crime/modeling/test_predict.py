import numpy as np
import pandas as pd

import financial_crime.modeling.predict as predict


class FakeFeatureEngineer:
    @staticmethod
    def load(path):
        return FakeFeatureEngineer()

    def fit_transform(self, dataframe):
        return dataframe.assign(ID=["a"], event_timestamp=["2022-01-01"])


class FakeOnlineFeatures:
    def to_df(self):
        return pd.DataFrame({"Is Laundering": [1], "feature": [2]})


class FakeStore:
    def push(self, source, dataframe):
        assert source == "transaction_push_source"

    def get_feature_service(self, name):
        return "service"

    def get_online_features(self, features, entity_rows):
        return FakeOnlineFeatures()


class FakePipeline:
    @staticmethod
    def load(path):
        return FakePipeline()

    def predict_proba(self, dataframe):
        return np.array(
            [[0.1, 0.9]]
        )


def test_main_pushes_features_predicts_and_writes(monkeypatch, tmp_path):
    input_path = tmp_path / "input.csv"
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame({"value": [1]}).to_csv(input_path, index=False)
    saved = {}
    monkeypatch.setattr(predict, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(predict, "FeatureStore", lambda path: FakeStore())
    monkeypatch.setattr(predict, "InferencePipeline", FakePipeline)
    monkeypatch.setattr(predict.np, "savetxt", lambda path, values, delimiter: saved.update(path=path, values=values, delimiter=delimiter))

    predict.main(input_path=input_path, pipeline_dir=tmp_path / "pipeline", predictions_path=predictions_path)

    assert saved == {"path": predictions_path, "values": [1], "delimiter": ","}