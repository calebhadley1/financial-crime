from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """
    All requests must have a corresponding entry in the feature store
    """

    ID: UUID
    event_timestamp: datetime


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1]
    probability: float