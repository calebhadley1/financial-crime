from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ML Pipeline location
PIPELINE_DIR = MODELS_DIR / "pipeline"

# Feature Engineering Constants
# Currency conversion rates as of 9/1/2022 via https://www.exchangerates.org.uk/historical/.../01_09_2022
CURRENCY_MAP = {
    "US Dollar": 1.0,
    "Euro": 0.9945,
    "Bitcoin": 20050.50,
    "Yuan": 0.1446,
    "Yen": 0.0071,
    "UK Pound": 1.154,
    "Brazil Real": 0.1907,
    "Australian Dollar": 0.6789,
    "Rupee": 0.0125,
    "Ruble": 0.0166,
    "Canadian Dollar": 0.7601,
    "Mexican Peso": 0.0495,
    "Swiss Franc": 1.0184,
    "Shekel": 0.2943,
    "Saudi Riyal": 0.266,
}
# Unique list of formats extracted from "Payment Format" column
PAYMENT_FORMATS = [
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
]
# Windows to calculate transaction frequencies for
TIME_WINDOWS = {
    "10s": "Tx_Last_10_Sec",
    "30s": "Tx_Last_30_Sec",
    "1min": "Tx_Last_1_Min",
    "5min": "Tx_Last_5_Min",
    "1h": "Tx_Last_1_Hour",
    "1D": "Tx_Last_1_Day",
    "10D": "Tx_Last_10_Days",
}
# Original Column names
TIMESTAMP = "Timestamp"
FROM_BANK = "From Bank"
ACCOUNT = "Account"
TO_BANK = "To Bank"
ACCOUNT_1 = "Account.1"
AMOUNT_RECEIVED = "Amount Received"
RECEIVING_CURRENCY = "Receiving Currency"
AMOUNT_PAID = "Amount Paid"
PAYMENT_CURRENCY = "Payment Currency"
PAYMENT_FORMAT = "Payment Format"
IS_LAUNDERING = "Is Laundering"

# Engineered Column Names
TRANSACTION_ID = "ID"
EVENT_TIMESTAMP = "event_timestamp"
LABELER = "labeler"
AMOUNT_RECEIVED_USD = "Amount_Received_USD"
AMOUNT_PAID_USD = "Amount_Paid_USD"
ACCOUNT_SAME = "Account_Same"
BANK_SAME = "Bank_Same"
ACCOUNT_PAIR = "account_pair"
PAIR_TRANSACTION_COUNT = "pair_transaction_count"
ACCOUNT_TRANSACTED_WITH_ACCOUNT_1_BEFORE = "Account_Transacted_With_Account1_Before"

# Preprocessing Constants
CATEGORICAL_FEATURES = ["Receiving Currency", "Payment Currency", "Payment Format"]

# Model Training Constants
RANDOM_STATE = 42
MODEL_N_ESTIMATORS = 65
MODEL_MAX_DEPTH = 4
SAMPLING_STRATEGY = 0.5
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.536

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
