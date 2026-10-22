import os
from pathlib import Path
import json
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Phoenix Assistant API")

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in CORS_ORIGINS.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = Path(__file__).parent / "data" / "pages.jsonl"

STOPWORDS = {
    "מה", "איך", "האם", "למה", "כמה", "מתי",
    "של", "על", "את", "עם", "או", "גם", "אם",
    "אני", "לי", "זה", "זו", "הוא", "היא",
    "אפשר", "יש", "אין", "האם", "באתר",
    "the", "and", "for", "what", "how", "is",
}


class ChatRequest(BaseModel):
    question: str


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[א-תA-Za-z0-9]+", text.lower())
    return [
        word for word in words
        if len(word) > 1 and word not in STOPWORDS
    ]


def load_pages() -> list[dict]:
    pages = []

    if not DATA_FILE.exists():
        return pages

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                page = json.loads(line)
                if page.get("text") and page.get("url"):
                    pages.append(page)
            except json.JSONDecodeError:
                continue

    return pages


def term_matches(word: str, term: str) -> bool:
    # התאמה בסיסית גם כשיש תחיליות בעברית
    return (
        word == term
        or word.endswith(term)
        or term.endswith(word)
    )



def score_page(page: dict, terms: list[str]) -> int:
    title_words = tokenize(page.get("title", ""))
    body_words = tokenize(page.get("text", ""))

    score = 0

    for term in set(terms):
        # התאמה בכותרת מקבלת ניקוד גבוה יותר
        in_title = any(
            term_matches(word, term)
            for word in title_words
        )

        in_body = any(
            term_matches(word, term)
            for word in body_words
        )

        if in_title:
            score += 8
        elif in_body:
            score += 1

    return score


def make_snippet(text: str, terms: list[str]) -> str:
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(part.strip()) > 25
    ]

    ranked = []
    for sentence in sentences:
        words = tokenize(sentence)
        matches = sum(
            1 for term in set(terms)
            if any(term_matches(word, term) for word in words)
        )
        if matches:
            ranked.append((matches, sentence))

    ranked.sort(key=lambda item: item[0], reverse=True)

    if ranked:
        return " ".join(
            sentence for _, sentence in ranked[:2]
        )[:700]

    # אם לא נמצאו משפטים מתאימים, מחפשים אזור בטקסט
    lower_text = text.lower()
    positions = [
        lower_text.find(term)
        for term in terms
        if lower_text.find(term) >= 0
    ]

    if positions:
        pos = min(positions)
        start = max(0, pos - 200)
        return text[start:start + 650].strip()

    return text[:500].strip()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pages_loaded": len(load_pages()),
    }


@app.post("/chat")
def chat(request: ChatRequest):
    terms = tokenize(request.question)

    if not terms:
        return {
            "answer": "כתבי שאלה עם כמה מילות מפתח כדי שאוכל לחפש באתר.",
            "sources": [],
        }

    pages = load_pages()

    if not pages:
        return {
            "answer": (
                "לא הצלחתי לטעון את קובץ הדפים. "
                "בדקי שקובץ pages.jsonl קיים בתיקיית data."
            ),
            "sources": [],
        }

    ranked = sorted(
        (
            (score_page(page, terms), page)
            for page in pages
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    if not ranked or ranked[0][0] < 2:
        return {
            "answer": (
                "לא מצאתי בדפים שנאספו מידע מספיק כדי לענות "
                "בביטחון. אפשר לנסח את השאלה אחרת?"
            ),
            "sources": [],
        }

    best_score = ranked[0][0]
    min_relevant_score = max(3, best_score * 0.75)

    relevant = [
        (score, page)
        for score, page in ranked[:3]
        if score > min_relevant_score
    ]

    # אם הסינון לא השאיר מקורות, נשמור לפחות את המקור המוביל.
    if not relevant:
        relevant = [ranked[0]]

    sources = []

    for score, page in relevant:
        title = page.get("title") or "עמוד באתר הפניקס"
        snippet = make_snippet(page["text"], terms)

        sources.append({
            "title": title,
            "url": page["url"],
            "snippet": snippet,
        })

    source_text = " ".join(
        source["snippet"] for source in sources
    ).lower()

    question = request.question.lower()

    # תשובה מבוססת-כללים, רק כאשר המקור תומך בה.
    if (
        "ביטוח" in question
        and "חובה" in question
        and "נזקי גוף" in source_text
    ):
        details = []

        if "פגיעה פיזית" in source_text:
            details.append("פגיעה גופנית")

        if "כאב וסבל" in source_text:
            details.append("כאב וסבל")

        if "אובדן כושר עבודה" in source_text:
            if "הכנסה" in source_text:
                details.append("אובדן כושר עבודה והכנסה")
            else:
                details.append("אובדן כושר עבודה")

        answer = (
            "ביטוח חובה לרכב נועד לכסות נזקי גוף במקרה של תאונת דרכים."
        )

        if details:
            answer += (
                " לפי המידע שנמצא באתר, הפגיעות עשויות לכלול "
                + ", ".join(details)
                + "."
            )
    else:
        answer = (
            "מצאתי מקורות שעשויים להיות רלוונטיים, אבל עדיין אין "
            "לי אפשרות לסכם באופן אמין את כל סוגי השאלות. "
            "כדי לא לנחש, מוצגים למטה קטעי המידע והמקורות שמצאתי."
        )

    return {
        "answer": answer,
        "sources": sources,
    }
