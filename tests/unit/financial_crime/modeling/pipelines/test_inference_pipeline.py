import numpy as np
import pandas as pd

from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline


class FakePreprocessor:
    def transform(self, dataframe):
        return dataframe[["value"]]

    def save(self, path):
        path.write_bytes(b"preprocessor")


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, dataframe):
        return np.array([1] * len(dataframe))

    def predict_proba(self, dataframe):
        return np.tile([[0.2, 0.8]], (len(dataframe), 1))


def test_predict_and_predict_proba_delegate_to_components():
    pipeline = InferencePipeline(FakePreprocessor(), FakeModel())
    data = pd.DataFrame({"value": [3, 4]})

    np.testing.assert_array_equal(pipeline.predict(data), [1, 1])
    np.testing.assert_allclose(pipeline.predict_proba(data), [[0.2, 0.8], [0.2, 0.8]])


def test_predict_proba_requires_model_support():
    pipeline = InferencePipeline(FakePreprocessor(), object())

    try:
        pipeline.predict_proba(pd.DataFrame({"value": [1]}))
    except AttributeError as error:
        assert "does not support predict_proba" in str(error)
    else:
        raise AssertionError("predict_proba should reject models without probability support")


def test_save_writes_components(tmp_path):
    InferencePipeline(FakePreprocessor(), FakeModel()).save(tmp_path / "pipeline")

    assert (tmp_path / "pipeline" / "preprocessor.pkl").exists()
    assert (tmp_path / "pipeline" / "model.pkl").exists()