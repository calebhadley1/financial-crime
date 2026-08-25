
from datetime import datetime
from uuid import uuid4

import factory
from financial_crime.schemas.engineered_feature import EngineeredFeature

class EngineeredFeatureFactory(factory.Factory):
    class Meta:
        model = EngineeredFeature

    ID: str = str(uuid4())
    event_timestamp: datetime = datetime.now()
    Timestamp: str =  "2022/09/01 00:20"
    From_Bank: str = "10"
    To_Bank: str = "10"
    Account: str = "8000EBD30"
    Account1 : str = "8000EBD31"
    Amount_Received: float = 3697.34
    Receiving_Currency: str = "US Dollar"
    Amount_Paid: float = 3697.34
    Payment_Currency: str = "US Dollar"
    Payment_Format: str = "Reinvestment"
    labeler: str = "mle-team"
    Amount_Received_USD: float = 3697.34
    Amount_Paid_USD: float = 3697.34
    Account_Same: int = 1
    Bank_Same: int = 1
    Account_Transacted_With_Account1_Before: int = 0
    account_pair: str = "8000EBD30::8000EBD31"
    pair_transaction_count: int = 1
