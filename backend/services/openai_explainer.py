from typing import List, Optional
import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# ================= OPENAI CLIENT =================
api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None


# ================= EXPLAIN FUNCTION =================
def openai_explain(
    prediction: str,
    confidence: float,
    reasons: List[str],
    category: Optional[str] = None,
    verification: Optional[str] = None
) -> str:
    """
    Generate a human-friendly explanation using OpenAI.
    """

    # ✅ Safe check
    if not client:
        logger.warning("⚠️ OPENAI_API_KEY missing")
        return "AI explanation unavailable (API key not configured)."

    # ---------------- PROMPT ----------------
    prompt: str = f"""
You are an AI assistant explaining a news verification result.

Prediction: {prediction}
Confidence: {confidence}%
Category: {category if category else "Unknown"}

Key Indicators:
- {'; '.join(reasons)}

Additional Verification:
{verification if verification else "No external verification available"}

Task:
Explain this result in simple, clear language for a normal user.

Guidelines:
- Be neutral and cautious
- Do NOT claim absolute truth
- Keep it concise (3–5 lines)
- Mention uncertainty if needed
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        message = response.choices[0].message
        content: str = message.content if message and message.content else ""

        return content.strip() if content else "No explanation generated."

    except Exception as e:
        logger.error(f"❌ OpenAI error: {str(e)}")
        return "AI explanation failed. Please try again later."