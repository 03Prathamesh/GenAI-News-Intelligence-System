from pydantic import BaseModel
from typing import List


class NewsRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    real_prob: float
    fake_prob: float
    category: str
    explanation: List[str]
    warning: str
