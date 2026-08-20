from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EngineeredFeature(BaseModel):
    model_config = ConfigDict(
        serialize_by_alias=True,  # Dict keys default to aliases during dump
        populate_by_name=True,  # Allows loading data using either field name or alias
    )

    ID: str
    event_timestamp: datetime
    Timestamp: str
    From_Bank: str = Field(alias="From Bank")
    To_Bank: str = Field(alias="To Bank")
    Account: str
    Account1: str = Field(alias="Account.1")
    Amount_Received: float = Field(alias="Amount Received")
    Receiving_Currency: str = Field(alias="Receiving Currency")
    Amount_Paid: float = Field(alias="Amount Paid")
    Payment_Currency: str = Field(alias="Payment Currency")
    Payment_Format: str = Field(alias="Payment Format")
    labeler: str
    Amount_Received_USD: float
    Amount_Paid_USD: float
    Account_Same: int
    Bank_Same: int
