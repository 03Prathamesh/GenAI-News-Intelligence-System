from typing import Dict, List, Any
import os
import sys
import pickle
import re
import logging

from services.openai_explainer import openai_explain
from services.realtime_service import realtime_fact_check
from services.rag_factcheck import rag_verify

logger = logging.getLogger(__name__)

# ---------------- PATH SETUP ----------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.preprocess import clean_text

# ---------------- MODEL PATH ----------------
MODEL_PATH = os.path.join(ROOT_DIR, "models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.pkl")

model = None
vectorizer = None

# ---------------- SAFE LOAD ----------------
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model missing: {MODEL_PATH}")

    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer missing: {VECTORIZER_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info("✅ Model + Vectorizer loaded successfully")

except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None
    vectorizer = None   # ❗ DO NOT crash app


# ---------------- CATEGORY ----------------
POLITICAL = ["election", "government", "minister", "vote"]
HEALTH = ["doctor", "vaccine", "health"]
FINANCE = ["money", "loan", "crypto"]

def detect_category(text: str) -> str:
    t = text.lower()

    if any(w in t for w in POLITICAL):
        return "Political"
    if any(w in t for w in HEALTH):
        return "Health"
    if any(w in t for w in FINANCE):
        return "Finance"

    return "General"


# ---------------- EXPLANATION ----------------
def generate_explanation(text: str, label: str) -> List[str]:
    reasons = []
    t = text.lower()

    if re.search(r"breaking|shocking|urgent", t):
        reasons.append("Sensational language detected")

    if re.search(r"share this|viral", t):
        reasons.append("Viral-style wording")

    if not reasons:
        reasons.append("Neutral language")

    return reasons


# ---------------- HYBRID DECISION ----------------
def combine_results(ml_label: str, rag_verdict: str):
    rag_verdict = rag_verdict.upper()

    if ml_label == "REAL" and rag_verdict == "SUPPORTED":
        return "REAL", "✅ Verified"

    if ml_label == "FAKE" and rag_verdict == "CONTRADICTED":
        return "FAKE", "🚨 Fake Confirmed"

    if rag_verdict == "INCONCLUSIVE":
        return ml_label, "🤔 Not verified"

    return "SUSPICIOUS", "⚠️ Conflict"


# ---------------- MAIN FUNCTION ----------------
def predict_news(text: str) -> Dict[str, Any]:
    try:
        if not text.strip():
            raise ValueError("Empty input")

        if model is None or vectorizer is None:
            return {
                "prediction": "ERROR",
                "confidence": 0,
                "category": "Unknown",
                "explanation": ["Model not loaded"],
                "warning": "⚠️ Backend issue"
            }

        # -------- CLEAN --------
        cleaned = clean_text(text)

        # -------- VECTORIZE --------
        vec = vectorizer.transform([cleaned])

        # -------- ML --------
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        label = "REAL" if pred == 1 else "FAKE"
        confidence = round(max(prob) * 100, 2)

        # -------- OTHER --------
        category = detect_category(text)
        reasons = generate_explanation(text, label)

        realtime_data = realtime_fact_check(text)
        rag_result = rag_verify(text)

        final_pred, status = combine_results(label, rag_result.get("verdict", ""))

        ai_explanation = openai_explain(
            prediction=label,
            confidence=confidence,
            reasons=reasons,
            category=category,
            verification=rag_result.get("verdict", "")
        )

        return {
            "prediction": label,
            "final_prediction": final_pred,
            "verification_status": status,
            "confidence": confidence,
            "real_prob": round(prob[1]*100,2),
            "fake_prob": round(prob[0]*100,2),
            "category": category,
            "explanation": reasons,
            "rag_verification": rag_result,
            "realtime_verification": realtime_data,
            "ai_explanation": ai_explanation,
            "warning": "⚠️ Verify sources" if final_pred != "REAL" else "✅ Credible"
        }

    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")

        return {
            "prediction": "ERROR",
            "confidence": 0,
            "category": "Unknown",
            "explanation": ["System error"],
            "warning": "⚠️ Try again"
        }