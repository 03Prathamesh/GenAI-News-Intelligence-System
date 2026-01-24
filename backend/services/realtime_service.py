from typing import Dict, Any, List
import requests
import os

TRUSTED_SOURCES: List[str] = [
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "nytimes.com",
    "cnn.com",
]


def realtime_fact_check(query: str) -> Dict[str, Any]:
    api_key: str | None = os.getenv("BING_API_KEY")
    if not api_key:
        return {"verified": False, "results": []}

    endpoint: str = "https://api.bing.microsoft.com/v7.0/search"

    headers: Dict[str, str] = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    params: Dict[str, Any] = {
        "q": query,
        "count": 5,
        "mkt": "en-US"
    }

    response = requests.get(
        endpoint,
        headers=headers,
        params=params,
        timeout=10
    )
    response.raise_for_status()
    data: Dict[str, Any] = response.json()

    results: List[Dict[str, Any]] = []
    verified: bool = False

    for item in data.get("webPages", {}).get("value", []):
        url: str = item.get("url", "")
        domain: str = url.split("/")[2] if "://" in url else ""

        is_trusted: bool = any(src in domain for src in TRUSTED_SOURCES)
        if is_trusted:
            verified = True

        results.append({
            "title": item.get("name"),
            "url": url,
            "trusted": is_trusted
        })

    return {
        "verified": verified,
        "results": results
    }
