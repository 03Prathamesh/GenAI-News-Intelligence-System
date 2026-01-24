from typing import Dict, List, Any
import os
import requests


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


def realtime_fact_check(query: str, api_key: str | None = None) -> Dict[str, Any]:
    """
    Performs assisted real-time fact checking using Bing Search API.
    """

    if api_key is None:
        api_key = os.getenv("BING_API_KEY")

    if not api_key:
        return {
            "verified": False,
            "results": [],
        }

    endpoint: str = "https://api.bing.microsoft.com/v7.0/search"

    headers: Dict[str, str] = {
        "Ocp-Apim-Subscription-Key": api_key
    }

    params: Dict[str, str | int] = {
        "q": query,
        "count": 5,
        "mkt": "en-US",
        "safeSearch": "Strict",
    }

    response: requests.Response = requests.get(
        endpoint,
        headers=headers,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    data: Dict[str, Any] = response.json()

    results: List[Dict[str, Any]] = []
    verified: bool = False

    pages: List[Dict[str, Any]] = data.get("webPages", {}).get("value", [])

    for item in pages:
        url: str = str(item.get("url", ""))
        title: str = str(item.get("name", ""))

        domain: str = ""
        if "://" in url:
            domain = url.split("/")[2]

        is_trusted: bool = any(src in domain for src in TRUSTED_SOURCES)

        if is_trusted:
            verified = True

        results.append({
            "title": title,
            "url": url,
            "trusted": is_trusted,
        })

    return {
        "verified": verified,
        "results": results,
    }
