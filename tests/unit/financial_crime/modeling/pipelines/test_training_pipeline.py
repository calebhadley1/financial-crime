import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from financial_crime.modeling.pipelines.inference_pipeline import InferencePipeline
from financial_crime.modeling.pipelines.training_pipeline import TrainingPipeline


def make_training_data():
    return pd.DataFrame(
        {
            "Receiving Currency": ["US Dollar", "Euro"] * 6,
            "Payment Currency": ["US Dollar", "Euro"] * 6,
            "Payment Format": ["Cash", "Wire"] * 6,
            "feature": list(range(12)),
            "ID": [str(value) for value in range(12)],
            "event_timestamp": pd.date_range("2022-01-01", periods=12),
        }
    ), pd.DataFrame({"Is Laundering": [0, 1] * 6})


def test_split_and_resample_preserve_expected_shapes():
    features, labels = make_training_data()
    pipeline = TrainingPipeline(test_size=0.25, sampling_strategy=1.0, random_state=42)

    X_train, X_test, y_train, y_test = pipeline.split_data(features, labels)
    X_resampled, y_resampled = pipeline.handle_class_imbalance(X_train, y_train)

    assert len(X_train) == len(y_train) == 9
    assert len(X_test) == len(y_test) == 3
    assert len(X_resampled) == len(y_resampled)
    assert y_resampled.tolist().count(0) == y_resampled.tolist().count(1)


def test_train_returns_usable_inference_pipeline():
    features, labels = make_training_data()
    pipeline = TrainingPipeline(test_size=0.25, sampling_strategy=1.0, random_state=42)
    model = GradientBoostingClassifier(random_state=42, n_estimators=2, max_depth=1)

    result = pipeline.train(features, labels, model)

    assert isinstance(result, InferencePipeline)
    assert result.predict(features).shape == (12,)