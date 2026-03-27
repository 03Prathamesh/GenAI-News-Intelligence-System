from typing import List
import logging

from .fake_news_service import predict_news
from .realtime_service import realtime_fact_check
from .rag_factcheck import rag_verify
from .openai_explainer import openai_explain

logger = logging.getLogger(__name__)
logger.info("🚀 Services initialized")

__all__: List[str] = [
    "predict_news",
    "realtime_fact_check",
    "rag_verify",
    "openai_explain",
]