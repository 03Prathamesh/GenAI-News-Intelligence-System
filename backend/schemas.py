from pydantic import BaseModel
from typing import List, Dict, Any


# ---------------- REQUEST ----------------
class NewsRequest(BaseModel):
    text: str


# ---------------- RESPONSE ----------------
class PredictionResponse(BaseModel):
    prediction: str
    final_prediction: str
    verification_status: str
    confidence: float
    category: str
    explanation: List[str]

    # 🔍 Intelligence layers
    realtime_verification: Dict[str, Any]
    rag_verification: Dict[str, Any]
    ai_explanation: str

    warning: str