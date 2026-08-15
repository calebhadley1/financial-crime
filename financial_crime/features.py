from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import typer
from financial_crime.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    output_labels_path: Path = PROCESSED_DATA_DIR / "labels.csv"
):
    """
    This feature engineering notebook was created through research from my master's program. 
    The original notebook can be found at `papers/bu_omds/1_ai_for_leaders/milestone_3/notebooks/ibm_eda.ipynb`. 
    I have further reduced it to only the required code in `notebooks/0.01-cjjh-ibm-1.ipynb` and converted that into a script here.
    """
    logger.info("Generating features from dataset...")

    df = pd.read_csv(input_path)
    
    non_matching_currency_rows = df[df["Receiving Currency"] != df["Payment Currency"]]
    logger.info(f"Number of rows with non-matching currencies: {len(non_matching_currency_rows)}")

    # We want to maintain the Payment/Receiving Currency, since it indicates if a transaction was sent/received in the same country. We should scale all the currencys to a single currency like Dollar to make them comparable.
    logger.info("Converting all currencies to USD...")
    # 1 of XYZ currency to USD as of 9/1/2022 via https://www.exchangerates.org.uk/historical/.../01_09_2022
    usd_to_usd = 1
    euro_to_usd = 0.9945
    btc_to_usd = 20050.50  # Opening BTC price
    yuan_to_usd = 0.1446
    yen_to_usd = 0.0071
    pound_to_usd = 1.154
    brazil_to_usd = 0.1907
    aus_to_usd = 0.6789
    rupee_to_usd = 0.0125  # Assuming Indian Rupee
    ruble_to_usd = 0.0166
    canadian_to_usd = 0.7601
    peso_to_usd = 0.0495
    swiss_to_usd = 1.0184
    shekel_to_usd = 0.2943
    riyal_to_usd = 0.266
    currency_map = {
        "US Dollar": usd_to_usd,
        "Euro": euro_to_usd,
        "Bitcoin": btc_to_usd,
        "Yuan": yuan_to_usd,
        "Yen": yen_to_usd,
        "UK Pound": pound_to_usd,
        "Brazil Real": brazil_to_usd,
        "Australian Dollar": aus_to_usd,
        "Rupee": rupee_to_usd,
        "Ruble": ruble_to_usd,
        "Canadian Dollar": canadian_to_usd,
        "Mexican Peso": peso_to_usd,
        "Swiss Franc": swiss_to_usd,
        "Shekel": shekel_to_usd,
        "Saudi Riyal": riyal_to_usd,
    }

    df["Amount_Received_USD"] = df.apply(
        lambda row: row["Amount Received"] * currency_map[row["Receiving Currency"]], axis=1
    )
    df["Amount_Paid_USD"] = df.apply(
        lambda row: row["Amount Paid"] * currency_map[row["Payment Currency"]], axis=1
    )

    logger.info("Calculating account and bank match indicators...")
    df["Account_Same"] = (df["Account"] == df["Account.1"]).astype(int)
    df["Bank_Same"] = (df["From Bank"] == df["To Bank"]).astype(int)

    cols_to_drop = [
        "Timestamp",
        "From Bank",
        "Account",
        "To Bank",
        "Account.1"
    ]
    logger.info(f"Dropping columns that cannot be used for modeling: {cols_to_drop}")
    df = df.drop(cols_to_drop, axis=1)

    logger.info(f"Saving features dataset to {output_features_path}...")
    df.drop("Is Laundering", axis=1).to_csv(output_features_path, index=False)

    logger.info(f"Saving labels dataset to {output_labels_path}...")
    df[["Is Laundering"]].to_csv(output_labels_path, index=False)
    
    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()
