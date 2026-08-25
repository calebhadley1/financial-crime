import pandas as pd

from financial_crime.modeling.transformers.feature_engineering import FeatureEngineer


def test_transform_calculates_prior_account_pair_transactions_in_time_order():
    features = pd.DataFrame(
        {
            "Timestamp": ["2022/01/03 00:00", "2022/01/01 00:00", "2022/01/02 00:00"],
            "Account": ["A", "A", "A"],
            "Account.1": ["B", "B", "C"],
            "From Bank": ["1", "1", "1"],
            "To Bank": ["2", "2", "3"],
            "Amount Received": [10.0, 10.0, 10.0],
            "Receiving Currency": ["US Dollar"] * 3,
            "Amount Paid": [10.0, 10.0, 10.0],
            "Payment Currency": ["US Dollar"] * 3,
        }
    )

    transformed = FeatureEngineer().fit_transform(features)

    assert transformed["Timestamp"].tolist() == [
        "2022/01/01 00:00",
        "2022/01/02 00:00",
        "2022/01/03 00:00",
    ]
    assert transformed["account_pair"].tolist() == ["A::B", "A::C", "A::B"]
    assert transformed["pair_transaction_count"].tolist() == [1, 1, 2]
    assert transformed["Account_Transacted_With_Account1_Before"].tolist() == [0, 0, 1]
