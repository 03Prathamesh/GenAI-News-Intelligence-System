import os
import pickle
import numpy as np
import re
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

POLITICAL = ["election", "government", "minister", "vote", "parliament", "party"]
HEALTH = ["cure", "doctor", "medicine", "disease", "vaccine", "health"]
FINANCE = ["investment", "money", "profit", "loan", "scheme", "crypto"]

def detect_category(text):
    t = text.lower()
    if any(w in t for w in POLITICAL):
        return "Political"
    if any(w in t for w in HEALTH):
        return "Health"
    if any(w in t for w in FINANCE):
        return "Finance"
    return "General"

def generate_explanation(text, prediction):
    reasons = []
    text_lower = text.lower()
    
    if re.search(r"anonymous|unnamed|leaked|secret|unverified|source", text_lower):
        reasons.append("Anonymous or unverified sources")
    if re.search(r"guaranteed|miracle|shocking|exposed|breaking|urgent", text_lower):
        reasons.append("Sensational or exaggerated language")
    if re.search(r"share this|forward|before deleted|viral|spread", text_lower):
        reasons.append("Viral social media style language")
    if re.search(r"click here|subscribe|buy now|limited offer", text_lower):
        reasons.append("Contains promotional or clickbait phrases")
    if re.search(r"\$\$+|millions?|billions?|profit|get rich", text_lower):
        reasons.append("Focuses on financial gain or large sums")
    if re.search(r"must read|important|alert|warning", text_lower):
        reasons.append("Uses urgent or fear-based language")
    
    if prediction == "FAKE":
        if not reasons:
            reasons.append("Patterns consistent with known fake news")
        return reasons
    else:
        if not reasons:
            reasons.append("Uses neutral and factual language")
        else:
            return ["Balanced reporting with verified information"]
        return reasons

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]

    label = "REAL" if pred == 1 else "FAKE"
    confidence = round(float(max(prob)) * 100, 2)

    return {
        "prediction": label,
        "confidence": confidence,
        "real_prob": round(float(prob[1]) * 100, 2),
        "fake_prob": round(float(prob[0]) * 100, 2),
        "category": detect_category(text),
        "explanation": generate_explanation(text, label),
        "warning": "⚠️ Verify with trusted sources" if label == "FAKE" else "✅ Appears credible"
    }

def analyze_multiple_articles(texts):
    results = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            result = predict_news(text)
            results.append(result)
    return results

def get_detailed_metrics(text):
    result = predict_news(text)
    metrics = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "category": result["category"],
        "reasons": result["explanation"],
        "risk_level": "HIGH" if result["prediction"] == "FAKE" and result["confidence"] > 70 else "MEDIUM" if result["prediction"] == "FAKE" else "LOW",
        "recommendation": "Verify with official sources" if result["prediction"] == "FAKE" else "Credible information"
    }
    return metrics