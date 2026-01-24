from typing import Dict, List, Any
import os
import sys
import pickle
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

from src.preprocess import clean_text

MODEL_PATH = os.path.join(ROOT_DIR, "models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(ROOT_DIR, "models", "tfidf_vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

POLITICAL = ["election", "government", "minister", "vote", "parliament", "party"]
HEALTH = ["cure", "doctor", "medicine", "disease", "vaccine", "health"]
FINANCE = ["investment", "money", "profit", "loan", "scheme", "crypto"]


def detect_category(text: str) -> str:
    t = text.lower()
    if any(w in t for w in POLITICAL):
        return "Political"
    if any(w in t for w in HEALTH):
        return "Health"
    if any(w in t for w in FINANCE):
        return "Finance"
    return "General"


def generate_explanation(text: str, label: str) -> List[str]:
    reasons: List[str] = []
    t = text.lower()

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


def predict_news(text: str) -> Dict[str, Any]:
    cleaned: str = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred: int = int(model.predict(vec)[0])
    prob = model.predict_proba(vec)[0]

    label: str = "REAL" if pred == 1 else "FAKE"

    return {
        "prediction": label,
        "confidence": round(float(max(prob)) * 100, 2),
        "real_prob": round(float(prob[1]) * 100, 2),
        "fake_prob": round(float(prob[0]) * 100, 2),
        "category": detect_category(text),
        "explanation": generate_explanation(text, label),
        "warning": "⚠️ Verify with trusted sources" if label == "FAKE" else "✅ Appears credible"
    }
