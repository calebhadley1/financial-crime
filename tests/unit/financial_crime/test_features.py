import pandas as pd

import financial_crime.features as features


class FakeFeatureEngineer:
    def fit_transform(self, dataframe):
        return dataframe.assign(extra=1)

    def save(self, path):
        path.write_text("saved")


def test_main_writes_features_labels_and_engineer(tmp_path, monkeypatch):
    input_path = tmp_path / "input.csv"
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    engineer_path = tmp_path / "feature_engineer.pkl"
    pd.DataFrame({"ID": ["a"], "event_timestamp": ["2022-01-01"], "Is Laundering": [1], "labeler": ["team"]}).to_csv(input_path, index=False)
    written = {}

    def fake_to_parquet(self, path, index):
        written[path] = (self.copy(), index)

    monkeypatch.setattr(features, "FeatureEngineer", FakeFeatureEngineer)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    features.main(input_path, features_path, labels_path, engineer_path)

    assert written[features_path][0].columns.tolist() == ["ID", "event_timestamp", "extra"]
    assert written[labels_path][0].columns.tolist() == ["ID", "event_timestamp", "Is Laundering", "labeler"]
    assert written[features_path][1] is False
    assert engineer_path.read_text() == "saved"