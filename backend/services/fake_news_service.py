from typing import Dict, List, Any
import os
import sys
import pickle
import re
import logging

from services.openai_explainer import openai_explain
from services.realtime_service import realtime_fact_check
from services.rag_factcheck import rag_verify

# ---------------- LOGGING ----------------
logger = logging.getLogger(__name__)

# ---------------- PATH SETUP ----------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.preprocess import clean_text

# ---------------- MODEL PATH ----------------
MODEL_PATH: str = os.path.join(ROOT_DIR, "models", "fake_news_model.pkl")

# ---------------- LOAD MODEL ----------------
model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    logger.info("✅ Pipeline model loaded successfully")

except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    raise RuntimeError("Model loading failed")


# ---------------- CATEGORY KEYWORDS ----------------
POLITICAL: List[str] = ["election", "government", "minister", "vote", "parliament", "party", "fraud"]
HEALTH: List[str] = ["cure", "doctor", "medicine", "disease", "vaccine", "health"]
FINANCE: List[str] = ["investment", "money", "profit", "loan", "scheme", "crypto", "fraud"]

# ---------------- CATEGORY DETECTION ----------------
def detect_category(text: str) -> str:
    t: str = text.lower()

    if any(w in t for w in POLITICAL):
        return "Political"
    if any(w in t for w in HEALTH):
        return "Health"
    if any(w in t for w in FINANCE):
        return "Finance"

    return "General"


# ---------------- EXPLANATION RULES ----------------
def generate_explanation(text: str, label: str) -> List[str]:
    reasons: List[str] = []
    t: str = text.lower()

    if re.search(r"anonymous|unnamed|leaked|secret|unverified", t):
        reasons.append("Anonymous or unverified sources")

    if re.search(r"guaranteed|miracle|shocking|breaking|urgent", t):
        reasons.append("Sensational or exaggerated language")

    if re.search(r"share this|viral|forward|before deleted", t):
        reasons.append("Viral or social-media style wording")

    if re.search(r"profit|get rich|millions|billions", t):
        reasons.append("Focus on financial gain")

    if not reasons:
        reasons.append(
            "Neutral and factual language detected"
            if label == "REAL"
            else "Patterns consistent with known fake news"
        )

    return reasons


# ---------------- HYBRID DECISION ----------------
def combine_results(ml_label: str, rag_verdict: str) -> Dict[str, str]:
    rag_verdict = rag_verdict.upper()

    if ml_label == "REAL" and rag_verdict == "SUPPORTED":
        return {"final_prediction": "REAL", "status": "✅ Verified"}

    if ml_label == "FAKE" and rag_verdict == "CONTRADICTED":
        return {"final_prediction": "FAKE", "status": "🚨 Fake Confirmed"}

    if (ml_label == "REAL" and rag_verdict == "CONTRADICTED") or \
       (ml_label == "FAKE" and rag_verdict == "SUPPORTED"):
        return {"final_prediction": "SUSPICIOUS", "status": "⚠️ Conflict detected"}

    if rag_verdict == "INCONCLUSIVE":
        return {"final_prediction": ml_label, "status": "🤔 Not fully verified"}

    return {"final_prediction": ml_label, "status": "Unknown"}


# ---------------- MAIN FUNCTION ----------------
def predict_news(text: str) -> Dict[str, Any]:
    try:
        if not text or not text.strip():
            raise ValueError("Empty input")

        # -------- CLEAN TEXT --------
        cleaned: str = clean_text(text)

        # -------- ML PREDICTION --------
        pred = model.predict([cleaned])[0]

        # -------- CONFIDENCE --------
        confidence: float = 90.0
        try:
            score = model.decision_function([cleaned])[0]
            confidence = round(min(abs(score) * 10, 100), 2)
        except Exception:
            pass

        label: str = "REAL" if pred == 1 else "FAKE"

        # -------- CATEGORY --------
        category: str = detect_category(text)

        # -------- RULE EXPLANATION --------
        reasons: List[str] = generate_explanation(text, label)

        # -------- REALTIME FACT CHECK --------
        realtime_data: Dict[str, Any] = realtime_fact_check(text)

        # -------- RAG FACT CHECK --------
        rag_result: Dict[str, Any] = rag_verify(text)

        # -------- HYBRID DECISION --------
        combined = combine_results(label, rag_result.get("verdict", "INCONCLUSIVE"))

        # -------- AI EXPLANATION --------
        ai_explanation: str = openai_explain(
            prediction=label,
            confidence=confidence,
            reasons=reasons,
            category=category,
            verification=rag_result.get("verdict", "")
        )

        # -------- FINAL RESPONSE --------
        return {
            "prediction": label,
            "final_prediction": combined["final_prediction"],
            "verification_status": combined["status"],
            "confidence": confidence,
            "category": category,
            "explanation": reasons,

            # 🔍 INTELLIGENCE LAYERS
            "realtime_verification": realtime_data,
            "rag_verification": rag_result,
            "ai_explanation": ai_explanation,

            "warning": "⚠️ Verify with trusted sources"
            if combined["final_prediction"] != "REAL"
            else "✅ Appears credible"
        }

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")

        return {
            "prediction": "ERROR",
            "final_prediction": "ERROR",
            "verification_status": "System failure",
            "confidence": 0.0,
            "category": "Unknown",
            "explanation": ["Error processing input"],
            "realtime_verification": {},
            "rag_verification": {},
            "ai_explanation": "System error occurred",
            "warning": "⚠️ Please try again"
        }