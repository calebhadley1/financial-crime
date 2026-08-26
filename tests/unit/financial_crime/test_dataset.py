import pandas as pd

from financial_crime.dataset import main


def test_main_writes_remainder_and_holdout(tmp_path):
    input_path = tmp_path / "input.csv"
    remainder_path = tmp_path / "remainder.csv"
    holdout_path = tmp_path / "holdout.csv"
    pd.DataFrame({"value": range(6)}).to_csv(input_path, index=False)

    main(input_path, holdout_path, remainder_path)

    assert pd.read_csv(remainder_path)["value"].tolist() == []
    assert pd.read_csv(holdout_path)["value"].tolist() == list(range(6))


def test_main_keeps_remainder_when_input_exceeds_holdout_size(tmp_path):
    input_path = tmp_path / "input.csv"
    remainder_path = tmp_path / "remainder.csv"
    holdout_path = tmp_path / "holdout.csv"
    pd.DataFrame({"value": range(50002)}).to_csv(input_path, index=False)

    main(input_path, holdout_path, remainder_path)

    assert len(pd.read_csv(remainder_path)) == 2
    assert len(pd.read_csv(holdout_path)) == 50000
