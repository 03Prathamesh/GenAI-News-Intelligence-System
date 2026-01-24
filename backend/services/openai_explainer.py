import os
from typing import List
from openai import OpenAI

# Create client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def openai_explain(
    prediction: str,
    confidence: float,
    reasons: List[str]
) -> str:
    """
    Generate a human-friendly explanation using OpenAI.
    """

    prompt = f"""
A machine learning system predicted a news article as "{prediction}"
with a confidence of {confidence}%.

Reasons:
- {'; '.join(reasons)}

Explain this clearly for a normal user.
Do NOT claim absolute truth.
Keep it neutral and informative.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )

    # Safe extraction (fixes Pylance + runtime safety)
    message = response.choices[0].message
    return message.content.strip() if message and message.content else "No explanation generated."
