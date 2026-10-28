"""Optional OpenAI-compatible grounded answer generation.

The app works without a model key. Configure LLM_API_KEY and LLM_MODEL (and
optionally LLM_BASE_URL) to enable a hosted OpenAI-compatible chat-completions API.
"""
from __future__ import annotations

import os

import requests


def generate_grounded_answer(question: str, evidence: list[dict], history: list[dict] | None = None) -> str | None:
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not api_key or not model or not evidence:
        return None

    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    evidence_blocks = []
    for idx, source in enumerate(evidence[:5], start=1):
        title = str(source.get("title", "עמוד באתר הפניקס"))[:180]
        url = str(source.get("url", ""))[:500]
        excerpt = str(source.get("text", source.get("snippet", "")))[:1800]
        evidence_blocks.append(f"[{idx}] כותרת: {title}\nכתובת: {url}\nקטע: {excerpt}")

    system_prompt = (
        "אתה עוזר מידע של חברת הפניקס. ענה בעברית ברורה ותמציתית. "
        "השתמש אך ורק בעובדות שמופיעות בקטעי המקור שסופקו. אל תוסיף ידע חיצוני, "
        "אל תנחש ואל תמציא פרטי כיסוי, סכומים, מועדים או תנאים. אם המקורות אינם "
        "מספיקים כדי לענות, אמור במפורש שלא נמצא מספיק מידע באתר והצע לבדוק מול שירות הלקוחות. "
        "אל תיתן ייעוץ אישי או קביעה לגבי פוליסה אישית. אל תיצור כתובות URL חדשות."
    )
    history_lines = []
    for item in (history or [])[-8:]:
        role = str(item.get("role", ""))
        content = str(item.get("content", item.get("text", ""))).strip()
        if role in ("user", "assistant") and content:
            label = "המשתמש" if role == "user" else "העוזר"
            history_lines.append(f"{label}: {content[:800]}")
    history_text = "\n".join(history_lines)
    user_prompt = (
        f"היסטוריית שיחה רלוונטית:\n{history_text or 'אין'}\n\n"
        f"שאלת המשתמש הנוכחית: {question}\n\nקטעי מקור:\n"
        + "\n\n".join(evidence_blocks)
    )

    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=(5, 18),
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload["choices"][0]["message"]["content"]
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        # A provider outage must not break the existing grounded fallback path.
        return None
    return None
