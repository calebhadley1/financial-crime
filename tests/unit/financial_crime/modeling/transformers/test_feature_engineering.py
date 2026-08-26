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
            "Payment Format": ["Bitcoin", "ACH", "ACH"],
        }
    )

    transformed = FeatureEngineer().fit_transform(features)

    assert list(transformed.columns) == (
        list(features.columns)
        + [
            "ID",
            "event_timestamp",
            "labeler",
            "Amount_Received_USD",
            "Amount_Paid_USD",
            "Account_Same",
            "Bank_Same",
            "account_pair",
            "pair_transaction_count",
            "Account_Transacted_With_Account1_Before",
            "Payment Format_ACH_Last_10_Sec",
            "Payment Format_ACH_Last_30_Sec",
            "Payment Format_ACH_Last_1_Min",
            "Payment Format_ACH_Last_5_Min",
            "Payment Format_ACH_Last_1_Hour",
            "Payment Format_ACH_Last_1_Day",
            "Payment Format_ACH_Last_10_Days",
            "Payment Format_Bitcoin_Last_10_Sec",
            "Payment Format_Bitcoin_Last_30_Sec",
            "Payment Format_Bitcoin_Last_1_Min",
            "Payment Format_Bitcoin_Last_5_Min",
            "Payment Format_Bitcoin_Last_1_Hour",
            "Payment Format_Bitcoin_Last_1_Day",
            "Payment Format_Bitcoin_Last_10_Days",
            "Payment Format_Cash_Last_10_Sec",
            "Payment Format_Cash_Last_30_Sec",
            "Payment Format_Cash_Last_1_Min",
            "Payment Format_Cash_Last_5_Min",
            "Payment Format_Cash_Last_1_Hour",
            "Payment Format_Cash_Last_1_Day",
            "Payment Format_Cash_Last_10_Days",
            "Payment Format_Cheque_Last_10_Sec",
            "Payment Format_Cheque_Last_30_Sec",
            "Payment Format_Cheque_Last_1_Min",
            "Payment Format_Cheque_Last_5_Min",
            "Payment Format_Cheque_Last_1_Hour",
            "Payment Format_Cheque_Last_1_Day",
            "Payment Format_Cheque_Last_10_Days",
            "Payment Format_Credit Card_Last_10_Sec",
            "Payment Format_Credit Card_Last_30_Sec",
            "Payment Format_Credit Card_Last_1_Min",
            "Payment Format_Credit Card_Last_5_Min",
            "Payment Format_Credit Card_Last_1_Hour",
            "Payment Format_Credit Card_Last_1_Day",
            "Payment Format_Credit Card_Last_10_Days",
            "Payment Format_Reinvestment_Last_10_Sec",
            "Payment Format_Reinvestment_Last_30_Sec",
            "Payment Format_Reinvestment_Last_1_Min",
            "Payment Format_Reinvestment_Last_5_Min",
            "Payment Format_Reinvestment_Last_1_Hour",
            "Payment Format_Reinvestment_Last_1_Day",
            "Payment Format_Reinvestment_Last_10_Days",
            "Payment Format_Wire_Last_10_Sec",
            "Payment Format_Wire_Last_30_Sec",
            "Payment Format_Wire_Last_1_Min",
            "Payment Format_Wire_Last_5_Min",
            "Payment Format_Wire_Last_1_Hour",
            "Payment Format_Wire_Last_1_Day",
            "Payment Format_Wire_Last_10_Days",
            "Payment Format_ACH_Tx_All_Time",
            "Payment Format_Bitcoin_Tx_All_Time",
            "Payment Format_Cash_Tx_All_Time",
            "Payment Format_Cheque_Tx_All_Time",
            "Payment Format_Credit Card_Tx_All_Time",
            "Payment Format_Reinvestment_Tx_All_Time",
            "Payment Format_Wire_Tx_All_Time",
        ]
    )

    assert transformed["Timestamp"].tolist() == [
        "2022/01/01 00:00",
        "2022/01/02 00:00",
        "2022/01/03 00:00",
    ]
    assert transformed["account_pair"].tolist() == ["A::B", "A::C", "A::B"]
    assert transformed["pair_transaction_count"].tolist() == [1, 1, 2]
    assert transformed["Account_Transacted_With_Account1_Before"].tolist() == [0, 0, 1]
