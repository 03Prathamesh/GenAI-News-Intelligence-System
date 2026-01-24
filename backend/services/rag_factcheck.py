from typing import List, Dict, Any
import os
import requests
from openai import OpenAI

# ================= OPENAI CLIENT =================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
    endpoint: str = "https://api.bing.microsoft.com/v7.0/search"

    headers: Dict[str, str] = {
        "Ocp-Apim-Subscription-Key": os.getenv("BING_API_KEY", "")
    }

    params: Dict[str, Any] = {
        "q": query,
        "count": 5,
        "mkt": "en-US",
    }

    response = requests.get(
        endpoint,
        headers=headers,
        params=params,
        timeout=10
    )
    response.raise_for_status()

    data: Dict[str, Any] = response.json()

    snippets: List[str] = []

    for item in data.get("webPages", {}).get("value", []):
        url: str = item.get("url", "")
        if any(src in url for src in TRUSTED_SOURCES):
            snippet: str = item.get("snippet", "")
            if snippet:
                snippets.append(snippet)

    return snippets

# ================= RAG VERIFICATION =================
def rag_verify(text: str) -> str:
    snippets: List[str] = search_news(text)

    if not snippets:
        return "No trusted sources found for verification."

    prompt: str = f"""
Claim:
{text}

Trusted news snippets:
{chr(10).join(snippets)}

Classify the claim as:
Supported / Contradicted / Inconclusive.

Explain briefly and cautiously.
Do NOT claim absolute truth.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    return content.strip() if content else "No explanation generated."
