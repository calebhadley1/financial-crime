import pandas as pd
import pandas.testing as tm

from financial_crime.api.clients.inference_client import InferenceClient


def test_inference_client_predict_returns_prediction_and_probability():
    class FakePipeline:
        model = type("Model", (), {"classes_": [0, 1]})()

        def predict_proba(self, dataframe):
            assert list(dataframe["feature"]) == [1, 2]
            return [[0.8, 0.2], [0.1, 0.9]]

    client = InferenceClient(pipeline=FakePipeline())
    features_df = pd.DataFrame({"feature": [1, 2]})

    result = client.predict(features_df)

    expected = pd.DataFrame({"prediction": [0, 1], "probability": [0.8, 0.9]})
    tm.assert_frame_equal(result, expected)
