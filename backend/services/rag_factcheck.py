from typing import List, Dict, Any, Optional
import os
import requests
import logging
import json
from openai import OpenAI

logger = logging.getLogger(__name__)

# ================= OPENAI CLIENT =================
api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None

# ================= TRUSTED SOURCES =================
TRUSTED_SOURCES: List[str] = [
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "nytimes.com",
    "cnn.com",
    "ndtv.com",
    "indiatoday.in",
]

# ================= SEARCH NEWS =================
def search_news(query: str) -> List[str]:
    api_key: Optional[str] = os.getenv("BING_API_KEY")

    if not api_key:
        logger.warning("⚠️ Missing BING_API_KEY")
        return []

    endpoint: str = "https://api.bing.microsoft.com/v7.0/search"

    headers: Dict[str, str] = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    params: Dict[str, Any] = {
        "q": query,
        "count": 8,
        "mkt": "en-US",
    }

    try:
        response = requests.get(
            endpoint,
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data: Dict[str, Any] = response.json()

    except Exception as e:
        logger.error(f"❌ Bing search failed: {str(e)}")
        return []

    snippets: List[str] = []

    for item in data.get("webPages", {}).get("value", []):
        url: str = item.get("url", "")

        if any(src in url for src in TRUSTED_SOURCES):
            snippet: str = item.get("snippet", "")
            if snippet:
                snippets.append(snippet)

    return snippets[:5]


# ================= RAG VERIFICATION =================
def rag_verify(text: str) -> Dict[str, Any]:
    snippets: List[str] = search_news(text)

    if not snippets:
        return {
            "verdict": "UNVERIFIED",
            "confidence": 20,
            "explanation": "No trusted sources found.",
            "sources_used": 0
        }

    # ✅ Handle missing OpenAI key safely
    if not client:
        logger.warning("⚠️ Missing OPENAI_API_KEY")
        return {
            "verdict": "UNVERIFIED",
            "confidence": 30,
            "explanation": "AI verification unavailable (missing API key).",
            "sources_used": len(snippets)
        }

    prompt: str = f"""
You are a fact-checking assistant.

Claim:
{text}

Trusted news snippets:
{chr(10).join(snippets)}

Task:
1. Classify the claim as:
   - Supported
   - Contradicted
   - Inconclusive
2. Give a short explanation (2–3 lines)
3. Be cautious

Respond ONLY in JSON:
{{
  "verdict": "...",
  "explanation": "...",
  "confidence": number
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        message = response.choices[0].message
        content: str = message.content if message and message.content else ""

        if not content.strip():
            raise ValueError("Empty response from OpenAI")

        # ✅ Safe JSON parsing
        try:
            parsed: Dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            logger.warning("⚠️ JSON parsing failed")

            return {
                "verdict": "INCONCLUSIVE",
                "confidence": 50,
                "explanation": content.strip(),
                "sources_used": len(snippets)
            }

        return {
            "verdict": parsed.get("verdict", "UNKNOWN"),
            "confidence": parsed.get("confidence", 50),
            "explanation": parsed.get("explanation", ""),
            "sources_used": len(snippets)
        }

    except Exception as e:
        logger.error(f"❌ OpenAI error: {str(e)}")

        return {
            "verdict": "ERROR",
            "confidence": 0,
            "explanation": "Failed to generate AI explanation.",
            "sources_used": len(snippets)
        }