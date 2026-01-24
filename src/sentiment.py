from typing import Dict
from textblob import TextBlob  # type: ignore


def analyze_sentiment(text: str) -> Dict[str, float | str]:
    try:
        blob = TextBlob(text)

        # TextBlob has no type stubs → explicitly ignore typing here
        polarity = float(blob.sentiment.polarity)  # type: ignore
        subjectivity = float(blob.sentiment.subjectivity)  # type: ignore

        if polarity > 0.1:
            sentiment_label = "POSITIVE"
        elif polarity < -0.1:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"

        return {
            "sentiment": sentiment_label,
            "compound": polarity,
            "positive": max(0.0, polarity),
            "negative": abs(min(0.0, polarity)),
            "neutral": max(0.0, 1.0 - abs(polarity)),
        }

    except Exception:
        return {
            "sentiment": "NEUTRAL",
            "compound": 0.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
        }
