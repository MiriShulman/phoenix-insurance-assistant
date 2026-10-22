import json
import os
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Phoenix Assistant API")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_FILE = Path(__file__).parent / "data" / "pages.jsonl"

STOPWORDS = {
    "מה", "איך", "האם", "למה", "כמה", "מתי", "מי", "של", "על", "את",
    "עם", "או", "גם", "אם", "אני", "לי", "זה", "זו", "הוא", "היא",
    "אפשר", "יש", "אין", "באתר", "בין", "לבין", "ההבדל", "כולל", "כוללת",
    "לרכב", "רכבכם", "שלכם", "שלכן", "שלי", "שלנו", "האם", "מהו", "מהי",
    "the", "and", "for", "what", "how", "is", "in", "to", "of",
}


class ChatRequest(BaseModel):
    question: str


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    return re.sub(r"\s+", " ", text.replace("׳", "'").replace("’", "'")).strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[א-תA-Za-z0-9]+", normalize_text(text))
    return [word for word in words if len(word) > 1 and word not in STOPWORDS]


CONTACT_TITLE_CUES = (
    "צור קשר", "יצירת קשר", "צרו קשר", "מוקדי שירות", "מוקד שירות",
    "שירות לקוחות", "פרטי קשר", "פנו אלינו", "צור קשר עם הפניקס",
)
CONTACT_BODY_CUES = (
    "צור קשר", "יצירת קשר", "צרו קשר", "מוקד שירות", "שירות לקוחות",
    "טלפון", "התקשרו", "כתובת", "דואר אלקטרוני", "אימייל", "מייל",
    "פקס", "וואטסאפ", "ווטסאפ", "פנייה", "פנו אלינו", "שעות פעילות",
)
DOCUMENT_CUES = (
    "מסמכים", "מסמך", "לצרף", "צרפו", "צירוף", "טופס", "תמונות",
    "צילום", "תיעוד", "דוח משטרתי", 'דו"ח משטרה', "רישיון", "חשבונית",
    "אישור", "מסמכים רפואיים", "מסמך התביעה",
)
# Sentences like these are usually about pricing/underwriting, not documents to attach.
# Keep this list specific so that legitimate "אישור אי הגשת תביעה" requirements remain.
DOCUMENT_PRICING_NOISE = (
    "מחושב על בסיס הערכת סיכונים",
    "הערכת סיכונים הלוקחת בחשבון",
    "מאפייני הנהג:",
    "מאפייני הנהגים",
    "פרמיית הביטוח",
    "מחיר הביטוח",
    "מחיר הפוליסה",
    "עלות הביטוח",
    "היסטוריית תביעות קודמות",
    "סוג הרכב אינו משפיע על הפרמיה",
)
CLAIM_CONTEXT_CUES = (
    "תביעה", "תביעת", "תביעות", "הגשת תביעה", "טופס תביעה",
    "מקרה ביטוח", "תאונה", "נזק לרכב", "נזקי רכב",
)
DAMAGE_ACTION_CUES = (
    "לדווח", "דיווח", "הודעה על מקרה ביטוח", "הגשת התביעה",
    "מלאו את מסמך התביעה", "הורידו את הטופס", "צלמו", "צילום הנזקים",
    "תיעוד האירוע", "לאסוף מידע", "איסוף פרטים", "החליפו פרטים",
    "להעבירו בהקדם", "הודיעו לחברת הביטוח", "דווחו לחברת הביטוח",
)
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+972[- ]?|0)(?:2|3|4|8|9|5\d|7\d)[- ]?\d{3}[- ]?\d{4}(?!\d)|(?<!\d)\*\d{3,4}(?!\d)"
)


def classify_intent(question: str) -> str:
    """Classify a small set of common Phoenix questions so retrieval is intent-aware."""
    q = normalize_text(question)

    contact_phrases = (
        "ליצור קשר", "יצירת קשר", "צור קשר", "צרו קשר", "איך ליצור קשר",
        "שירות לקוחות", "מוקד", "מספר טלפון", "טלפון של", "כתובת של",
        "אימייל", "דואר אלקטרוני", "מייל של", "נציג שירות", "פנו אליכם",
    )
    if any(phrase in q for phrase in contact_phrases):
        return "contact"

    asks_documents = any(word in q for word in (
        "מסמכים", "מסמך", "לצרף", "צירוף", "טפסים", "טופס", "מה צריך לצרף",
    ))
    mentions_claim_context = any(word in q for word in (
        "תביעה", "תביעת", "תביעות", "מקרה ביטוח", "תאונה", "רכב",
        # Users often ask about total loss without explicitly saying "תביעה" or "רכב".
        "טוטאלוס", "טוטאל לוס", "טוטאל-לוס", "אבדן", "אובדן מוחלט",
        "לא לתיקון", "אובדן להלכה",
    ))
    if asks_documents and mentions_claim_context:
        return "claim_documents"

    asks_damage_report = any(word in q for word in (
        "לדווח", "מדווח", "דיווח", "הודעה על", "נזק", "נזקים", "פגיעה",
    )) and any(word in q for word in ("רכב", "מכונית", "תאונה", "נזק", "נזקים"))
    if asks_damage_report and "מכסה" not in q and "כיסוי" not in q:
        return "report_damage"

    if any(word in q for word in (
        "תביעה", "תביעת", "תביעות", "מגישים", "להגיש", "הגשת", "תבע",
    )):
        return "claim_process"

    return "general"


def page_has_contact_evidence(page: dict) -> bool:
    title = normalize_text(page.get("title", ""))
    body = normalize_text(clean_page_text(page.get("text", "")))
    if any(cue in title for cue in CONTACT_TITLE_CUES):
        return True
    has_contact_context = any(cue in body for cue in CONTACT_BODY_CUES)
    has_contact_detail = bool(EMAIL_RE.search(body) or PHONE_RE.search(body))
    return has_contact_context and has_contact_detail


TOTAL_LOSS_TITLE_CUES = (
    "טוטאלוס", "טוטאל לוס", "טוטאל-לוס", "אבדן", "אובדן מוחלט", "לא לתיקון", "אובדן להלכה",
)
DOCUMENT_PROCESS_NOISE = (
    "תשלום התביעה",
    "מתבצע תוך 30 יום",
    "תוך 30 יום מרגע",
    "כפוף לאישורה",
)


def page_is_total_loss_document_page(page: dict) -> bool:
    """Identify documents pages that apply specifically to total-loss cases."""
    title = normalize_text(page.get("title", ""))
    return any(cue in title for cue in TOTAL_LOSS_TITLE_CUES)


def question_asks_total_loss(question: str) -> bool:
    q = normalize_text(question)
    return any(cue in q for cue in TOTAL_LOSS_TITLE_CUES)



def question_asks_third_party_claim(question: str) -> bool:
    q = normalize_text(question)
    return any(phrase in q for phrase in (
        "צד שלישי", "צד ג", "הנהג הפוגע", "אישור אי הגשת תביעה",
    ))


def question_asks_injury_claim(question: str) -> bool:
    q = normalize_text(question)
    return any(phrase in q for phrase in (
        "נזקי גוף", "פגיעות גוף", "נפגעים בתאונה", "מסמכים רפואיים",
        "אי כושר עבודה", "אי-כושר עבודה", "הפסדי שכר", "תביעת ביטוח חובה",
    )) or ("ביטוח חובה" in q and any(word in q for word in ("תביעה", "מסמכים", "לצרף")))


def _text_between(text: str, start_marker: str, end_markers: tuple[str, ...]) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    end_positions = [
        position for marker in end_markers
        if (position := text.find(marker, start + len(start_marker))) >= 0
    ]
    end = min(end_positions) if end_positions else len(text)
    return text[start:end].strip()


def _claim_document_sections(page: dict) -> dict[str, str]:
    """Extract the actual document-list sections from the scraped Phoenix article."""
    text = clean_page_text(page.get("text", ""))

    injury_anchor = "נפגעתם בתאונה ואתם מגישים תביעת ביטוח? זה המידע ואלה המסמכים שעליכם לצרף:"
    injury_section = _text_between(
        text,
        injury_anchor,
        ("על חברת הביטוח הנתבעת לחזור אליכם", "תביעת ביטוח בתאונה שהמבוטח אינו אשם בה"),
    )
    injury_list = ""
    if injury_section:
        list_start = injury_section.find("מידע מפורט על נסיבות האירוע")
        list_end = injury_section.find("וכו’", list_start) if list_start >= 0 else -1
        if list_start >= 0 and list_end >= 0:
            injury_list = injury_section[list_start:list_end + len("וכו’")].strip()

    third_party_section = _text_between(
        text,
        "בכדי למנוע מצב בו מבוטח תובע",
        ("אם אתם הצד השלישי שנתבע", "האם יש בכך כדי להשפיע על הפרמיה העתידית שלכם?"),
    )

    filing_match = re.search(
        r"(את הגשת התביעה באופן מעשי תוכלו לעשות.*?תתבקשו להשלים את החסר\.)",
        text,
    )
    filing = filing_match.group(1).strip() if filing_match else ""
    return {
        "injury": injury_list,
        "third_party": third_party_section,
        "filing": filing,
    }


def _total_loss_document_list(page: dict) -> str:
    """Return only the total-loss checklist, not timing/contact information nearby."""
    text = clean_page_text(page.get("text", ""))
    start_marker = "מסמכים נדרשים להעברה לחברת הביטוח"
    if start_marker not in text:
        return ""

    raw = _text_between(
        text,
        start_marker,
        ("ניתן להעביר את המסמכים במייל", "טלפונים חשובים לספק גרירה"),
    )
    if not raw:
        return ""

    # The scraper flattened the checklist into one line. Restore the individual
    # items only when their wording is present in the source text.
    item_specs = (
        ("אישור ניהול חשבון בנק של בעל הרכב או צ'ק מבוטל", "אישור ניהול חשבון בנק של בעל הרכב או צ'ק מבוטל"),
        ("טופס הסכמה למסירת מידע ממשרד הרישוי, חתום", "טופס הסכמה למסירת מידע ממשרד הרישוי חתום"),
        ("טופס הסכמה למסירת מידע ממשרד הרישוי – רישיון נהיגה, חתום", "טופס הסכמה למסירת מידע ממשרד הרישוי"),
        ("צילום רישיון נהיגה של הנהג משני צדדיו", "תצלום רישיון נהיגה של הנהג משני צדדיו"),
        ("צילום רישיון רכב", "תצלום רישיון רכב"),
        ("צילום תעודת זהות של בעל/י הרכב", 'תצלום תעודת זהות של בעל/י הרכב'),
        ("דו״ח שמאי סופי", 'דו"ח שמאי סופי'),
        ("אישור משטרה, אם המשטרה הייתה מעורבת", "אישור משטרה (במידה ומשטרה הייתה מעורבת)"),
        ("מכתב התחייבות בלתי חוזרת מהמשעבד עם סכום השעבוד המדויק, או מכתב הסרת שעבוד", "מכתב התחייבות בלתי חוזרת מהמשעבד"),
    )
    items: list[str] = []
    for display, source_phrase in item_specs:
        if source_phrase in raw and display not in items:
            items.append(display)
    if not items:
        return raw
    return "\n".join(f"• {item}" for item in items)


def make_claim_documents_snippet(page: dict, question: str) -> str:
    """Build a subtype-aware document excerpt so different claim types aren't merged."""
    asks_total_loss = question_asks_total_loss(question)
    asks_third_party = question_asks_third_party_claim(question)
    asks_injury = question_asks_injury_claim(question)

    if page_is_total_loss_document_page(page):
        if not asks_total_loss:
            return ""
        return _total_loss_document_list(page)

    url = normalize_text(page.get("url", ""))
    title = normalize_text(page.get("title", ""))
    if "car_insurance_changes" in url or "ביטוח רכב אחרי תאונה" in title:
        sections = _claim_document_sections(page)
        if asks_third_party:
            if not sections["third_party"]:
                return ""
            return sections["third_party"]
        if asks_injury:
            if not sections["injury"]:
                return ""
            return sections["injury"]

        parts: list[str] = []
        if sections["injury"]:
            parts.append("**תביעת נזקי גוף בעקבות תאונה (ביטוח חובה):** " + sections["injury"])
        if sections["third_party"]:
            parts.append("**תביעת צד שלישי בגין נזקי רכוש:** " + sections["third_party"])
        if sections["filing"]:
            parts.append("**אופן ההגשה:** " + sections["filing"])
        parts.append(
            "המאגר אינו מציג רשימה ייעודית מלאה לתביעה על נזק לרכב המבוטח במסגרת ביטוח מקיף, "
            "למעט מקרה של אובדן מוחלט (טוטאלוס); יש לבדוק את טופס התביעה המתאים."
        )
        return "\n\n".join(parts)

    # Keep the general retrieval path for any other document-focused pages.
    return make_snippet(
        page.get("text", ""),
        tokenize(question),
        [],
        question,
    )


def display_source_title(page: dict, question: str) -> str:
    original = (page.get("title") or "עמוד באתר הפניקס").strip()
    if classify_intent(question) != "claim_documents":
        return original
    if page_is_total_loss_document_page(page):
        return original

    url = normalize_text(page.get("url", ""))
    if "car_insurance_changes" in url or "ביטוח רכב אחרי תאונה" in normalize_text(original):
        if question_asks_third_party_claim(question):
            return "תביעת צד שלישי: מסמכים לצירוף (מתוך מאמר הפניקס על ביטוח רכב אחרי תאונה)"
        if question_asks_injury_claim(question):
            return "תביעת נזקי גוף: מסמכים לצירוף (מתוך מאמר הפניקס על ביטוח רכב אחרי תאונה)"
        return "תביעת רכב אחרי תאונה: מסמכים לפי סוג התביעה (מתוך מאמר הפניקס על התייקרות הביטוח)"
    return original


def page_has_claim_document_evidence(page: dict) -> bool:
    title = normalize_text(page.get("title", ""))
    body = normalize_text(clean_page_text(page.get("text", "")))
    claim_titles = (
        "הגשת תביעת רכב", "תביעת רכב", "תביעות רכב", "טופס תביעה",
        "הודעה על מקרה ביטוח", "מסמכים לצירוף", "מסמכים נדרשים",
    )
    if any(cue in title for cue in claim_titles):
        return True

    # Generic pages can mention "אישור היעדר תביעות" in a pricing paragraph.
    # Require evidence of an actual document list instead of accepting any page
    # containing one generic document word and one claim-related word.
    direct_document_phrases = (
        "אלה המסמכים שעליכם לצרף",
        "זה המידע ואלה המסמכים שעליכם לצרף",
        "המסמכים שיש לצרף",
        "מסמכים שיש לצרף",
        "מסמכים לצירוף",
        "מסמכים נדרשים",
        "מלאו את מסמך התביעה",
        "מסמך התביעה",
        "המסמכים הרלוונטיים",
        "אישור אי הגשת תביעה",
        "דוח שמאי",
        'דו"ח שמאי',
    )
    has_claim_context = any(cue in body for cue in CLAIM_CONTEXT_CUES)
    has_document_cue = any(cue in body for cue in DOCUMENT_CUES)
    has_direct_document_evidence = any(phrase in body for phrase in direct_document_phrases)
    if any(noise in body for noise in DOCUMENT_PRICING_NOISE) and not has_direct_document_evidence:
        return False
    return has_claim_context and has_document_cue and has_direct_document_evidence


def page_has_damage_report_evidence(page: dict) -> bool:
    title = normalize_text(page.get("title", ""))
    body = normalize_text(clean_page_text(page.get("text", "")))
    direct_titles = (
        "מה עושים במקרה של תאונת דרכים", "הגשת תביעת רכב",
        "הודעה על מקרה ביטוח רכב", "טופס תביעה",
    )
    if any(cue in title for cue in direct_titles):
        return True
    has_action = any(cue in body for cue in DAMAGE_ACTION_CUES)
    has_damage_context = any(cue in body for cue in (
        "נזק לרכב", "נזקים שנגרמו", "תאונת דרכים", "מקרה ביטוח רכב",
        "הגשת תביעה", "תביעת רכב",
    ))
    is_obviously_unrelated = any(cue in title for cue in (
        "תאונת שרשרת", "מערכות בטיחות", "סוגי ביטוח", "ביטוח צד ג",
        "ביטוח חובה", "ביטוח מקיף",
    ))
    return has_action and has_damage_context and not is_obviously_unrelated



def _hebrew_variants(word: str) -> set[str]:
    """Small, conservative set of Hebrew word forms for search matching."""
    word = normalize_text(word)
    variants = {word}
    prefixes = ("ו", "ה", "ב", "ל", "כ", "מ", "ש")
    suffixes = ("יהם", "יהן", "ים", "ות", "יים", "ינו", "יכם", "יכן", "יהם", "יה", "יו", "נו", "כם", "כן", "ך", "ה", "ת", "י")

    # Remove one common prefix, then optionally one common suffix.
    for prefix in prefixes:
        if word.startswith(prefix) and len(word) >= 4:
            base = word[1:]
            variants.add(base)
            for suffix in suffixes:
                if base.endswith(suffix) and len(base) - len(suffix) >= 3:
                    variants.add(base[:-len(suffix)])
    # Also try suffix-only variants.
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            variants.add(word[:-len(suffix)])
    return {v for v in variants if len(v) >= 3}


def term_matches(word: str, term: str) -> bool:
    if word == term:
        return True
    if len(word) < 3 or len(term) < 3:
        return False
    return bool(_hebrew_variants(word) & _hebrew_variants(term))


def clean_page_text(text: str) -> str:
    """Remove late related-article/navigation tails from Phoenix's scraped text."""
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return ""

    # These are Phoenix related-content/footer labels, not article prose. A marker
    # is only treated as a cutoff when it occurs late in the page.
    strong_tail_markers = (
        "מה חדש |", "מה חדש ביטוח רכב", "טיפים |", "מושגים |",
        "כתבות נוספות", "מאמרים נוספים", "תגיות:",
    )
    cutoffs: list[int] = []
    min_late_position = max(700, int(len(clean) * 0.38))
    lowered = clean.lower()

    for marker in strong_tail_markers:
        pos = lowered.find(marker.lower())
        if pos >= min_late_position:
            cutoffs.append(pos)

    for marker in ("עוד בנושא", "מאמרים קשורים", "כתבות קשורות", "נושאים קשורים"):
        pos = lowered.find(marker.lower())
        if pos < max(700, int(len(clean) * 0.35)):
            continue
        tail = lowered[pos:]
        navigation_signals = (
            "מה חדש", "טיפים |", "מושגים |", "תגיות:",
            "כתבות נוספות", "מאמרים נוספים", "עוד בנושא",
        )
        signal_count = sum(1 for signal in navigation_signals if signal in tail)
        if signal_count >= 2:
            cutoffs.append(pos)

    if cutoffs:
        clean = clean[:min(cutoffs)].strip(" \t\r\n-–—|•·")

    # Remove common breadcrumb-like prefixes when they are glued to article text.
    clean = re.sub(
        r"^(?:למידע נוסף על ביטוח רכב|לפרטים נוספים בנושא ביטוח רכב|לפרטים נוספים בנושא ביטוח)\s+",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    return clean


def load_pages() -> list[dict]:
    pages: list[dict] = []
    if not DATA_FILE.exists():
        return pages

    with DATA_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                page = json.loads(line)
                if page.get("text") and page.get("url"):
                    page["text"] = clean_page_text(page["text"])
                    pages.append(page)
            except json.JSONDecodeError:
                continue
    return pages

def build_search_terms(question: str) -> tuple[list[str], list[str]]:
    """Build query terms and intent-specific phrases for common questions."""
    q = normalize_text(question)
    direct_terms = list(dict.fromkeys(tokenize(q)))
    extra_phrases: list[str] = []
    intent = classify_intent(q)

    if "חובה" in q:
        extra_phrases.extend([
            "הכל על ביטוח רכב חובה", "ביטוח חובה לרכב", "נזקי גוף",
            "פוליסת ביטוח רכב חובה",
        ])
    if "מקיף" in q:
        extra_phrases.extend(["ביטוח מקיף לרכב", "נזקי רכוש", "הרכב המבוטח"])
    if "צד ג" in q or "צד ג׳" in q or "צד שלישי" in q:
        extra_phrases.extend([
            "ביטוח צד ג לרכב", "צד שלישי", "נזקי רכוש", "רכוש של צד שלישי",
        ])

    asks_claim = any(word in q for word in (
        "תביעה", "תביעת", "תביעות", "מגישים", "להגיש", "הגשת", "תבע",
    ))
    asks_accident = any(word in q for word in ("תאונה", "תאונת", "דרכים", "תאונות"))
    if asks_claim:
        extra_phrases.extend([
            "הגשת תביעת רכב", "טופס תביעה", "הודעה על מקרה ביטוח רכב",
            "תאונה או אירוע ביטוחי", "תביעות רכב",
        ])
    if asks_accident:
        extra_phrases.extend(["מה עושים במקרה של תאונת דרכים", "תאונת דרכים"])

    if intent == "contact":
        extra_phrases.extend([
            "צור קשר", "יצירת קשר", "מוקד שירות לקוחות", "פרטי קשר",
            "מספר טלפון", "שעות פעילות מוקד השירות",
        ])
    elif intent == "claim_documents":
        extra_phrases.extend([
            "מסמכים לצירוף לתביעה", "המסמכים הרלוונטיים", "מסמכים נדרשים",
            "טופס תביעה", "הודעה על מקרה ביטוח רכב", "דוח משטרתי",
        ])
    elif intent == "report_damage":
        extra_phrases.extend([
            "דיווח על נזק לרכב", "הודעה על מקרה ביטוח רכב",
            "מה עושים במקרה של תאונת דרכים", "תיעוד הנזקים",
            "הגשת תביעת רכב", "צלמו את הנזקים",
        ])

    return direct_terms, list(dict.fromkeys(extra_phrases))


def score_page(
    page: dict,
    direct_terms: list[str],
    extra_phrases: list[str],
    question: str,
) -> float:
    """Rank pages by query intent and direct evidence, not generic site-wide words."""
    title = normalize_text(page.get("title", ""))
    body = normalize_text(clean_page_text(page.get("text", "")))
    title_words = tokenize(title)
    body_words = tokenize(body)
    q = normalize_text(question)
    intent = classify_intent(q)
    score = 0.0

    generic_terms = {
        "רכב", "רכבים", "ביטוח", "הביטוח", "חברת", "הפניקס", "נזק", "נזקים",
        "תביעה", "תביעת", "תביעות", "הגשת", "הגשה", "מקרה", "מידע",
    }
    for term in set(direct_terms):
        title_match = any(term_matches(word, term) for word in title_words)
        body_match = any(term_matches(word, term) for word in body_words)
        if term in generic_terms:
            if title_match:
                score += 0.9
            elif body_match:
                score += 0.2
        elif title_match:
            score += 5.5
        elif body_match:
            score += 0.65

    for phrase in extra_phrases:
        normalized_phrase = normalize_text(phrase)
        if normalized_phrase in title:
            score += 12.0
        elif normalized_phrase in body:
            score += 3.0
        else:
            phrase_terms = tokenize(normalized_phrase)
            hits = sum(
                1 for term in phrase_terms
                if any(term_matches(word, term) for word in body_words)
            )
            score += min(hits, 2) * 0.15

    asks_claim = any(word in q for word in (
        "תביעה", "תביעת", "תביעות", "מגישים", "להגיש", "הגשת", "תבע",
    ))
    if asks_claim:
        if any(phrase in title for phrase in (
            "הגשת תביעת רכב", "טופס תביעה", "הודעה על מקרה ביטוח רכב", "תביעת רכב",
        )):
            score += 18.0
        if "מה עושים במקרה של תאונת דרכים" in title:
            score += 12.0
        if "ביטוח רכב אחרי תאונה" in title:
            score += 10.0
        if any(phrase in body for phrase in (
            "את הגשת התביעה באופן מעשי תוכלו לעשות",
            "מלאו את מסמך התביעה אונליין",
            "הורידו את הטופס למילוי ושלחו",
        )):
            score += 30.0
        elif "תביעת ביטוח: מה עושים במקרה של תאונה" in body:
            score += 8.0
        if any(term in title for term in (
            "היעדר תביעות", "העדר תביעות", "הנחת היעדר", "הנחת העדר",
            "הטבה בביטוח", "היסטוריית תביעות", "עבר ביטוחי",
        )):
            score -= 15.0
        if any(term in title for term in ("תאונת שרשרת", "סיבות לתאונות", "סיבות לתאונת")):
            score -= 8.0

    if "חובה" in q and not any(word in q for word in ("הונאה", "הונאות", "מאגר")):
        if "הכל על ביטוח רכב חובה" in title:
            score += 14.0
        elif "ביטוח חובה לרכב" in title or "ביטוח רכב חובה" in title:
            score += 9.0
        if "הונאות" in title or "מאגר המידע" in title:
            score -= 15.0

    if "מקיף" in q and ("ההבדל" in q or "הבדלים" in q):
        if "ביטוח מקיף" in title or "מקיף לרכב" in title:
            score += 10.0
    elif "מקיף" in q and ("ביטוח מקיף" in title or "מקיף לרכב" in title):
        score += 8.0

    if "צד ג" in q or "צד ג׳" in q or "צד שלישי" in q:
        if "צד ג" in title or "צד ג'" in title or "צד שלישי" in title:
            score += 10.0

    if intent == "contact":
        if any(cue in title for cue in CONTACT_TITLE_CUES):
            score += 30.0
        if any(cue in body for cue in CONTACT_BODY_CUES):
            score += 4.0
        if EMAIL_RE.search(body) or PHONE_RE.search(body):
            score += 8.0
        if not page_has_contact_evidence(page):
            score -= 30.0

    elif intent == "claim_documents":
        if any(cue in title for cue in (
            "הגשת תביעת רכב", "תביעת רכב", "תביעות רכב", "טופס תביעה",
            "הודעה על מקרה ביטוח",
        )):
            score += 24.0
        document_hits = sum(1 for cue in DOCUMENT_CUES if cue in body)
        if document_hits:
            score += min(document_hits, 5) * 2.5
        if any(cue in body for cue in CLAIM_CONTEXT_CUES):
            score += 5.0
        if not page_has_claim_document_evidence(page):
            score -= 30.0
        if any(cue in title for cue in (
            "סוגי ביטוח", "מערכות בטיחות", "תאונת שרשרת", "היעדר תביעות",
        )):
            score -= 18.0

    elif intent == "report_damage":
        if any(cue in title for cue in (
            "מה עושים במקרה של תאונת דרכים", "הגשת תביעת רכב",
            "הודעה על מקרה ביטוח רכב", "טופס תביעה",
        )):
            score += 24.0
        if "ביטוח רכב אחרי תאונה" in title and any(cue in body for cue in DAMAGE_ACTION_CUES):
            score += 14.0
        action_hits = sum(1 for cue in DAMAGE_ACTION_CUES if cue in body)
        if action_hits:
            score += min(action_hits, 5) * 2.5
        if not page_has_damage_report_evidence(page):
            score -= 30.0

    return score


def make_snippet(
    text: str,
    terms: list[str],
    extra_phrases: list[str],
    question: str = "",
) -> str:
    """Return only sentences that are useful for the question's specific intent."""
    clean = clean_page_text(text)
    if not clean:
        return ""

    nav_markers = (
        "לביצוע פעולות מהירות", "לרכישת ביטוח", "לקבלת הצעה", "לקבלת הצעת",
        "מידע נוסף", "לפרטים נוספים", "עוד בנושא", "מה חדש", "לכל הטפסים",
        "לכל הפעולות", "בצעו שינויים", "עדכון כרטיס אשראי", "קבלו הצעה",
        "פתיחת תפריט", "איתור מוסך הסדר", "איתור שמאי", "תחנות שירות",
        "כתבות נוספות", "מאמרים נוספים", "מושגים |", "טיפים |", "תגיות:",
    )
    query_terms = list(dict.fromkeys(t for t in terms if len(t) >= 3))
    intent = classify_intent(question)
    q_claim = any(t in query_terms for t in (
        "תביעה", "תביעת", "תביעות", "מגישים", "להגיש", "הגשת", "תבע",
    )) or any("תביע" in normalize_text(p) or "הגשת" in normalize_text(p) for p in extra_phrases)
    q_accident = any(t in query_terms for t in ("תאונה", "תאונת", "דרכים", "תאונות")) or any(
        "תאונת דרכים" in normalize_text(p) for p in extra_phrases
    )
    q_injury = any(word in " ".join(query_terms) for word in ("נפגע", "פצוע", "גוף", "רפואי"))
    q_police = any(word in query_terms for word in ("משטרה", "שוטר", "דוח", 'דו"ח'))

    claim_action_cues = (
        "את הגשת התביעה", "הגשת התביעה", "מסמך התביעה", "טופס התביעה",
        "מלאו את מסמך התביעה", "להודיע לחברת הביטוח", "להעבירו בהקדם האפשרי",
        "איסוף כמה שיותר מידע", "לאסוף כמה שיותר מידע", "המסמכים הרלוונטיים",
        "תביעה דרך אתר", "דרך אתר הפניקס", "דרך סוכן", "קבלת התביעה",
        "טיפול בתביעה", "תביעת הפיצויים", "חשוב לקחת את הפרטים",
        "דו\"ח המשטרה", "הדוח המשטרתי",
    )
    accident_action_cues = (
        "צלמו את האירוע", "צלמו תמונות", "צלמו את הנזקים", "מקום התאונה",
        "פרטי הנהג", "פרטי הנהגים", "מספרי הרכבים", "צילום פוליסת הביטוח",
        "תיעוד האירוע", "תיעוד בכתב", "החליפו פרטים", "רשמו לפניכם",
        "חשוב לקחת את הפרטים", "צילום רישיון", "המסמכים והתמונות",
    )
    pricing_noise = (
        "פרמיה העתידית", "התייקרות הפרמיה", "פרמיית הביטוח הייתה מרקיעה",
        "היעדר תביעות", "העדר תביעות", "ריבוי תביעות קטנות", "היינו מגישים תביעה",
        "מאגר המידע לאיתור הונאות", "ביטוח רכבי יוקרה",
    )
    towing_noise = ("גרר", "גרירה", "מוסך vip", "מוסך הסדר", "ספק הגרירה")
    injury_noise = (
        "נזקי גוף", "מסמכים רפואיים", "אישור אי-כושר עבודה", "הולכי רגל",
        "תביעת ביטוח לנזקי גוף", "נפגעתם בתאונה", "נפגעתם כנוסעים",
        'הדו"ח חיוני גם להליך תביעת הפיצויים', "טיפול רפואי ללא תשלום",
    )
    fault_routing_noise = (
        "במקרה ואינכם אשמים", "ביטוח צד ג' שלו", "ביטוח צד ג’ שלו",
        "חברת הביטוח של הנהג הפוגע",
    )

    candidates: list[tuple[float, int, int, str]] = []
    seen: set[str] = set()
    sentences = re.split(r"(?<=[.!?׃])\s+|[\n•·|]+", clean)
    for index, raw_sentence in enumerate(sentences):
        sentence = re.sub(r"\s+", " ", raw_sentence).strip(" \t\r\n-–—|•·")
        if not (25 <= len(sentence) <= 520):
            continue
        normalized = normalize_text(sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        if sentence.endswith("?") and len(sentence) < 180:
            continue
        if any(marker in normalized for marker in nav_markers):
            continue

        # Intent-specific filters are applied before generic keyword scoring. This is
        # the key guard against returning a true-but-unrelated paragraph.
        if intent == "contact":
            has_contact_cue = any(cue in normalized for cue in CONTACT_BODY_CUES)
            has_contact_detail = bool(EMAIL_RE.search(sentence) or PHONE_RE.search(sentence))
            if not (has_contact_cue or has_contact_detail):
                continue
            score = 5.0 + (5.0 if has_contact_detail else 0.0)
            if any(cue in normalized for cue in ("מוקד", "שירות לקוחות", "יצירת קשר", "צור קשר")):
                score += 3.0
            candidates.append((score, 5, index, sentence))
            continue

        if intent == "claim_documents":
            # Avoid bringing pricing/history paragraphs into a document checklist.
            # The phrase "אישור אי הגשת תביעה" is still allowed when it appears as
            # a genuine third-party claim requirement, not in a pricing paragraph.
            if any(noise in normalized for noise in DOCUMENT_PRICING_NOISE):
                continue
            if any(noise in normalized for noise in DOCUMENT_PROCESS_NOISE):
                continue
            has_document_cue = any(cue in normalized for cue in DOCUMENT_CUES)
            has_claim_context = any(cue in normalized for cue in CLAIM_CONTEXT_CUES)
            direct_document_phrases = (
                "אלה המסמכים שעליכם לצרף", "המסמכים שיש לצרף", "מסמכים שיש לצרף",
                "מסמכים לצירוף", "מסמכים נדרשים", "לצרף לתביעה",
                "מלאו את מסמך התביעה", "מסמך התביעה", "המסמכים הרלוונטיים",
                "אישור אי הגשת תביעה", "דוח שמאי", 'דו"ח שמאי',
            )
            has_direct_document_evidence = any(phrase in normalized for phrase in direct_document_phrases)
            if not has_document_cue or not (has_claim_context or has_direct_document_evidence):
                continue
            score = 5.0
            score += min(sum(1 for cue in DOCUMENT_CUES if cue in normalized), 4) * 2.0
            if has_direct_document_evidence:
                score += 4.0
            if any(cue in normalized for cue in ("טופס התביעה", "טופס תביעה", "מסמכי התביעה", "המסמכים הרלוונטיים")):
                score += 4.0
            candidates.append((score, 5, index, sentence))
            continue

        if intent == "report_damage":
            has_action = any(cue in normalized for cue in DAMAGE_ACTION_CUES + accident_action_cues)
            has_damage_context = any(cue in normalized for cue in (
                "נזק", "נזקים", "תאונה", "רכב", "מקרה ביטוח", "תביעה",
            ))
            if not has_action or not has_damage_context:
                continue
            score = 5.0
            score += min(sum(1 for cue in DAMAGE_ACTION_CUES + accident_action_cues if cue in normalized), 4) * 2.0
            candidates.append((score, 5, index, sentence))
            continue

        if q_claim and any(noise in normalized for noise in pricing_noise):
            continue
        if q_claim and not q_injury and any(noise in normalized for noise in injury_noise):
            continue
        if q_claim and not q_police and "על אף שבתאונת דרכים ללא נפגעים" in normalized:
            continue
        if q_claim and not any(term in query_terms for term in ("אשם", "אשמים", "צד", "צד ג")) and any(noise in normalized for noise in fault_routing_noise):
            continue

        words = tokenize(sentence)
        hit_terms = [
            term for term in set(query_terms)
            if any(term_matches(word, term) for word in words)
        ]
        score = min(len(hit_terms), 4) * 3.0
        priority = 0
        for phrase in extra_phrases:
            normalized_phrase = normalize_text(phrase)
            if len(normalized_phrase) >= 8 and normalized_phrase in normalized:
                score += 4.0
                priority += 2
                break

        has_claim_action = any(cue in normalized for cue in claim_action_cues)
        has_accident_action = any(cue in normalized for cue in accident_action_cues)
        if q_claim and not (has_claim_action or has_accident_action) and not q_police:
            continue
        if q_claim:
            if has_claim_action:
                score += 4.0
                priority += 3
            if "מלאו את מסמך התביעה אונליין" in normalized or "הורידו את הטופס למילוי ושלחו" in normalized:
                score += 4.0
                priority += 4
            if "את הגשת התביעה באופן מעשי תוכלו לעשות" in normalized:
                score += 4.0
                priority += 4
            if "חברת הביטוח תאשר את קבלת התביעה" in normalized:
                score += 2.5
                priority += 2
            if "על חברת הביטוח הנתבעת לחזור אליכם בתוך 30 יום" in normalized:
                score -= 2.0
            if any(cue in normalized for cue in ("תביעה", "תביעת", "מסמכים", "הגשת")):
                score += 1.0
                priority += 1
        if q_accident:
            if any(cue in normalized for cue in ("תאונה", "תאונת", "דרכים")):
                score += 2.0
            if has_accident_action:
                score += 4.0
                priority += 3
        if q_claim and any(noise in normalized for noise in towing_noise) and not any(
            cue in normalized for cue in ("את הגשת התביעה", "מלאו את מסמך התביעה", "המסמכים הרלוונטיים")
        ):
            continue
        if score >= 3.0:
            candidates.append((score, priority, index, sentence))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1], -len(item[3])), reverse=True)
    selected: list[tuple[float, int, int, str]] = []
    for candidate in candidates:
        sentence = candidate[3]
        if any(sentence in old[3] or old[3] in sentence for old in selected):
            continue
        selected.append(candidate)
        if len(selected) == 5:
            break

    selected.sort(key=lambda item: item[2])
    output: list[str] = []
    total = 0
    for _, _, _, sentence in selected:
        addition = len(sentence) + (1 if output else 0)
        if total + addition > 850:
            continue
        output.append(sentence)
        total += addition
    return " ".join(output)


def compose_answer(question: str, pages: list[dict], sources: list[dict]) -> str:
    q = normalize_text(question)
    evidence = normalize_text(" ".join((page.get("text") or "") for page in pages))
    source_text = normalize_text(" ".join(
        f"{source.get('title', '')} {source.get('snippet', '')}" for source in sources
    ))
    displayed_evidence = normalize_text(" ".join(source.get("snippet", "") for source in sources))

    asks_difference = any(word in q for word in ("ההבדל", "הבדלים", "להבדיל", "שונה"))
    asks_claim = any(word in q for word in ("תביעה", "תביעות", "מגישים", "להגיש", "הגשת", "תבע"))
    mentions_compulsory = "חובה" in q
    mentions_comprehensive = "מקיף" in q
    mentions_third_party = "צד ג" in q or "צד שלישי" in q
    intent = classify_intent(q)

    if intent == "contact":
        contact_snippets = [source for source in sources if (source.get("snippet") or "").strip()]
        if not contact_snippets:
            return "לא מצאתי במאגר קטע שמציג פרטי קשר מאומתים של הפניקס. כדאי לבדוק בעמוד 'צור קשר' באתר הרשמי."
        first = contact_snippets[0]
        excerpt = (first.get("snippet") or "").strip()[:650]
        return f"המקור הרלוונטי ביותר שמצאתי הוא '{first.get('title', 'צור קשר באתר הפניקס')}': {excerpt}"

    if intent == "claim_documents":
        if not sources:
            if question_asks_total_loss(question):
                return "לא נמצא במאגר קטע שמפרט את המסמכים לתביעת אובדן מוחלט (טוטאלוס)."
            if question_asks_third_party_claim(question):
                return "לא נמצא במאגר קטע שמפרט את המסמכים לתביעת צד שלישי."
            if question_asks_injury_claim(question):
                return "לא נמצא במאגר קטע שמפרט את המסמכים לתביעת נזקי גוף."
            return (
                "לא נמצא במאגר מקור שמפרט את המסמכים לתביעת רכב. "
                "יש לבדוק את טופס התביעה המתאים לסוג המקרה."
            )

        snippet = (sources[0].get("snippet") or "").strip()
        title = (sources[0].get("title") or "עמוד באתר הפניקס").strip()

        if question_asks_total_loss(question):
            return (
                f"בתביעת אובדן מוחלט (טוטאלוס), עמוד הפניקס '{title}' מפרט את המסמכים הבאים:\n"
                f"{snippet}\n"
                "בדקי את טופס התביעה ואת תנאי הפוליסה כדי לוודא אם נדרשים מסמכים נוספים."
            )

        if question_asks_third_party_claim(question):
            return (
                "בתביעת צד שלישי בגין נזק לרכוש, המקור באתר הפניקס מציין את הדרישות הבאות: "
                f"{snippet} "
                "המסמכים המדויקים עשויים להשתנות לפי נסיבות המקרה, ולכן יש לבדוק את טופס התביעה המתאים."
            )

        if question_asks_injury_claim(question):
            snippet = snippet.rstrip()
            if snippet and snippet[-1] not in ".!?":
                snippet += "."
            return (
                "בתביעת נזקי גוף בעקבות תאונה (ביטוח חובה), המקור באתר הפניקס מפרט: "
                f"{snippet} יש לבדוק את טופס התביעה המתאים למקרה."
            )

        return (
            "המסמכים משתנים לפי סוג התביעה. במאגר לא נמצאה רשימה אחת שחלה על כל תביעת רכב; "
            "מצאתי קטע מתוך מאמר הפניקס על ביטוח רכב אחרי תאונה, ובו רשימות נפרדות לסוגי התביעות הבאים:\n\n"
            f"{snippet}"
        )

    if intent == "report_damage":
        action_snippets = [
            (source.get("snippet") or "").strip()
            for source in sources
            if (source.get("snippet") or "").strip()
        ]
        if not action_snippets:
            return (
                "לא מצאתי במאגר הוראות מספיקות לדיווח על נזק לרכב. "
                "כדאי לבדוק את עמוד ההנחיות לתאונת דרכים או את טופס 'הודעה על מקרה ביטוח רכב' באתר הפניקס."
            )
        excerpt = " ".join(action_snippets[:2])[:850]
        return (
            f"לפי המקורות הרלוונטיים שנמצאו באתר הפניקס: {excerpt} "
            "אם מדובר בתביעה, בדקי גם את טופס 'הודעה על מקרה ביטוח רכב' ואת ההוראות שבו."
        )

    if asks_difference and mentions_compulsory and mentions_comprehensive:
        if "נזקי גוף" in evidence and ("נזקי רכוש" in evidence or "רכוש" in evidence):
            return (
                "ההבדל העיקרי הוא סוג הנזק שמכוסה: ביטוח חובה נועד לכסות נזקי גוף "
                "לנהג, לנוסעים ולנפגעים מחוץ לרכב בתאונת דרכים. ביטוח מקיף מיועד לנזקי רכוש, "
                "לרבות נזק לרכב המבוטח ולרכוש של צד שלישי, בהתאם לתנאי הפוליסה. "
                "ביטוח חובה נדרש לפי חוק; ביטוח מקיף אינו חובה חוקית."
            )

    if asks_difference and mentions_compulsory and mentions_third_party:
        if "נזקי גוף" in evidence and ("נזקי רכוש" in evidence or "רכוש" in evidence):
            return (
                "ביטוח חובה מכסה נזקי גוף שנגרמים בתאונת דרכים. ביטוח צד ג' מיועד לכסות "
                "נזקי רכוש שנגרמו לצד שלישי, בהתאם לתנאי הפוליסה. אלה כיסויים שונים, "
                "וביטוח צד ג' אינו מחליף ביטוח חובה."
            )

    if asks_claim and any(word in q for word in ("תאונה", "דרכים", "רכב")):
        # When the displayed excerpts contain actual filing instructions, answer from
        # those excerpts rather than from unrelated text elsewhere on the page.
        has_filing_steps = any(term in displayed_evidence for term in (
            "את הגשת התביעה באופן מעשי תוכלו לעשות",
            "מלאו את מסמך התביעה אונליין",
            "הורידו את הטופס למילוי ושלחו",
        ))
        if has_filing_steps:
            steps: list[str] = []
            if any(term in displayed_evidence for term in (
                "לאסוף כמה שיותר מידע", "איסוף כמה שיותר מידע", "חשוב לקחת את הפרטים",
            )):
                steps.append("במקום התאונה אספי מידע ופרטי הנהגים והרכבים המעורבים.")
            if any(term in displayed_evidence for term in (
                "צלמו את האירוע", "צלמו תמונות", "צלמו את הנזקים",
            )):
                steps.append("צלמי את האירוע ואת הנזקים ושמרי את התיעוד.")
            if "את הגשת התביעה באופן מעשי תוכלו לעשות" in displayed_evidence or "דרך אתר הפניקס" in displayed_evidence:
                steps.append("אפשר להגיש את התביעה דרך אתר הפניקס או באמצעות סוכן.")
            if "מלאו את מסמך התביעה אונליין" in displayed_evidence:
                if "בהודעת אסמס או בדואר" in displayed_evidence or "במייל" in displayed_evidence:
                    steps.append("מלאי את טופס התביעה המקוון והעלי את המסמכים הרלוונטיים; לחלופין, אפשר להוריד את הטופס ולשלוח אותו עם המסמכים, התמונות והתיעוד במייל, בהודעת SMS או בדואר.")
                else:
                    steps.append("מלאי את טופס התביעה המקוון והעלי את המסמכים הרלוונטיים; לחלופין, אפשר להוריד את הטופס ולשלוח אותו עם התמונות והתיעוד בערוצים המצוינים במקור.")
            if "במקרה ועירבתם משטרה" in displayed_evidence or "הדוח המשטרתי" in displayed_evidence or 'דו"ח המשטרה' in displayed_evidence:
                steps.append("אם המשטרה הייתה מעורבת, המקור מציין לצרף את הדוח המשטרתי.")
            if "חברת הביטוח תאשר את קבלת התביעה" in displayed_evidence:
                steps.append("חברת הביטוח תאשר את קבלת התביעה ותבקש להשלים מסמכים חסרים, אם יהיו.")
            if steps:
                return "לפי דף הפניקס על תביעת ביטוח אחרי תאונה: " + " ".join(steps) + " בדקי את ההוראות בטופס המתאים כדי לוודא אילו מסמכים נדרשים לסוג התביעה שלך."

        has_claim_area = any(term in source_text for term in (
            "הגשת תביעת רכב",
            "תביעות רכב",
        ))
        has_claim_form = any(term in source_text for term in (
            "טופס תביעה",
            "הודעה על מקרה ביטוח רכב",
        ))
        has_accident_guide = (
            "מה עושים במקרה של תאונת דרכים" in source_text
            or "תאונת דרכים" in evidence
        )
        if has_claim_area and has_claim_form and has_accident_guide:
            return (
                "כדי להתחיל, פתחי את המקורות המצורפים של הפניקס בנושא תאונת דרכים והגשת תביעת רכב. "
                "במקורות שנמצאו מופיעים אזור 'הגשת תביעת רכב' וטופס 'הודעה על מקרה ביטוח רכב'; "
                "בחרי את הטופס המתאים לסוג הביטוח ולמקרה ופעלי לפי ההוראות והמסמכים שמפורטים בו. "
                "בדף ההנחיות לתאונת דרכים מצוין שאין לחתום על מסמך המהווה הודאה באשמה. "
                "המקורות שנמצאו כאן אינם מפרטים באופן מלא את כל המסמכים הדרושים, ולכן כדאי לבדוק את הטופס עצמו."
            )
        if (has_claim_area or has_claim_form) and has_accident_guide:
            return (
                "מצאתי באתר הפניקס דף הנחיות למקרה של תאונת דרכים וגם מקור שקשור לתביעות רכב. "
                "פתחי את המקורות המצורפים ובדקי את טופס התביעה המתאים לסוג הביטוח ולמקרה. "
                "לפי דף ההנחיות, אין לחתום על מסמך המהווה הודאה באשמה. המקורות שהתקבלו אינם מפרטים "
                "באופן מלא את כל שלבי ההגשה ואת כל המסמכים הדרושים, ולכן כדאי לבדוק את הטופס עצמו."
            )
        if has_accident_guide:
            practical_steps: list[str] = []
            if any(term in evidence for term in (
                "לאסוף מידע", "איסוף פרטים", "פרטי הנהג", "פרטי הנהגים", "פרטי הרכב", "מספר רישוי",
            )):
                practical_steps.append("לאסוף את פרטי הנהגים והרכבים המעורבים.")
            if any(term in evidence for term in (
                "צלמו את האירוע", "צלמו את הנזקים", "צילום הרכב", "לתעד את האירוע", "תיעוד האירוע",
            )):
                practical_steps.append("לתעד את האירוע ולצלם את הרכבים והנזקים.")

            answer = "לפי דפי ההנחיות של הפניקס שנמצאו, לאחר תאונה כדאי "
            if practical_steps:
                answer += " ".join(practical_steps) + " "
            answer += (
                "בתוצאות הנוכחיות לא נמצא מקור עצמאי שמפרט את כל שלבי הגשת התביעה ואת רשימת המסמכים המלאה. "
                "כדי להשלים את המידע, יש לאתר באתר הפניקס את עמוד 'הגשת תביעת רכב' או את טופס 'הודעה על מקרה ביטוח רכב'. "
            )
            if "אין לחתום" in evidence and "הודאה באשמה" in evidence:
                answer += "בדף ההנחיות מצוין שאין לחתום על מסמך המהווה הודאה באשמה."
            return answer.strip()

    if mentions_third_party and any(word in q for word in ("כולל", "מכסה", "מהו", "מה זה", "ביטוח")):
        if "צד שלישי" in evidence or "צד ג" in evidence or "נזקי רכוש" in evidence:
            return (
                "ביטוח צד ג' לרכב מיועד לכסות נזקי רכוש שנגרמו לצד שלישי על ידי הרכב המבוטח, "
                "בהתאם לתנאי הפוליסה ולגבולות הכיסוי. הוא אינו תחליף לביטוח חובה, ובדרך כלל "
                "אינו מכסה נזק לרכב שלך עצמו; לשם כך נדרש כיסוי מתאים כמו ביטוח מקיף."
            )

    if mentions_compulsory and any(word in q for word in ("כולל", "מכסה", "מהו", "מה זה", "ביטוח")):
        if "נזקי גוף" in evidence:
            return (
                "ביטוח חובה לרכב נועד לכסות נזקי גוף בתאונת דרכים: לנהג, לנוסעים ברכב "
                "ולנפגעים מחוץ לרכב, כגון הולכי רגל. הוא אינו מיועד לכיסוי נזק לרכב או לרכוש; "
                "לנזקי רכוש נדרשים כיסויים אחרים, בהתאם לצורך ולתנאי הפוליסה."
            )

    if mentions_comprehensive and any(word in q for word in ("כולל", "מכסה", "מהו", "מה זה", "ביטוח")):
        if "רכוש" in evidence:
            return (
                "ביטוח מקיף מיועד לכסות נזקי רכוש לרכב המבוטח, ובמקורות הפניקס מצוין גם כיסוי "
                "לנזקי רכוש של צד שלישי. הכיסוי בפועל תלוי בתנאי הפוליסה, בסייגים ובגבולות הכיסוי."
            )

    # Use a relevant excerpt from the best-ranked source, and be explicit when evidence is limited.
    if sources:
        source = sources[0]
        snippet = (source.get("snippet") or "").strip()
        title = (source.get("title") or "עמוד באתר הפניקס").strip()
        if snippet:
            return f"לפי המקור באתר הפניקס, '{title}': {snippet[:430].rstrip()}"
        return f"מצאתי את העמוד '{title}' באתר הפניקס. פתחי את המקור המצורף כדי לקרוא את הפרטים המלאים."

    return "לא מצאתי בדפים שנאספו מידע מספיק כדי לענות בביטחון. אפשר לנסח את השאלה אחרת?"


@app.get("/")
def root():
    return {
        "name": "Phoenix Assistant API",
        "status": "ok",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    pages = load_pages()
    return {"status": "ok", "pages_loaded": len(pages)}



@app.post("/chat")
def chat(request: ChatRequest):
    question = (request.question or "").strip()
    direct_terms, extra_phrases = build_search_terms(question)

    if not direct_terms:
        return {"answer": "כתבי שאלה עם כמה מילות מפתח כדי שאוכל לחפש באתר.", "sources": []}

    pages = load_pages()
    if not pages:
        return {
            "answer": "לא הצלחתי לטעון את קובץ הדפים. בדקי שקובץ pages.jsonl קיים בתיקיית data.",
            "sources": [],
        }

    ranked = sorted(
        (
            (score_page(page, direct_terms, extra_phrases, question), page)
            for page in pages
        ),
        key=lambda item: item[0],
        reverse=True,
    )

    normalized_question = normalize_text(question)
    intent = classify_intent(normalized_question)

    # For these intents, never fall back to a generic page merely because it shares
    # words such as "רכב", "ביטוח" or "חברת הביטוח".
    if intent == "contact":
        contact_pages = [(score, page) for score, page in ranked if page_has_contact_evidence(page)]
        if not contact_pages:
            return {
                "answer": "לא מצאתי במאגר פרטי קשר מאומתים של הפניקס. בדקי בעמוד 'צור קשר' באתר הרשמי.",
                "sources": [],
            }
        ranked = contact_pages

    elif intent == "claim_documents":
        asks_total_loss = question_asks_total_loss(normalized_question)
        asks_third_party = question_asks_third_party_claim(normalized_question)
        asks_injury = question_asks_injury_claim(normalized_question)
        asks_own_comprehensive = (
            "מקיף" in normalized_question
            and not asks_total_loss
            and not asks_third_party
            and not asks_injury
        )

        if asks_own_comprehensive:
            return {
                "answer": (
                    "במאגר שהועלה לא נמצאה רשימת מסמכים ייעודית לתביעה על נזק לרכב המבוטח "
                    "במסגרת ביטוח מקיף (שאינה תביעת טוטאלוס). אפשר להיעזר בטופס 'הודעה על מקרה "
                    "ביטוח רכב' באתר הפניקס ולבדוק את רשימת המסמכים המופיעה בו."
                ),
                "sources": [],
            }

        document_pages = [
            (score, page) for score, page in ranked
            if page_has_claim_document_evidence(page)
            and (
                page_is_total_loss_document_page(page)
                if asks_total_loss
                else not page_is_total_loss_document_page(page)
            )
        ]

        # A question about a particular claim type must retrieve evidence for that
        # type from the body text, not require the webpage's overall title to match.
        if asks_third_party:
            document_pages = [
                (score, page) for score, page in document_pages
                if "אישור אי הגשת תביעה" in normalize_text(page.get("text", ""))
                and "צד שלישי" in normalize_text(page.get("text", ""))
            ]
        elif asks_injury:
            document_pages = [
                (score, page) for score, page in document_pages
                if "מסמכים רפואיים" in normalize_text(page.get("text", ""))
                and "הפסדי שכר" in normalize_text(page.get("text", ""))
            ]

        if not document_pages:
            if asks_total_loss:
                answer = "לא מצאתי במאגר רשימת מסמכים ייעודית לתביעת אובדן מוחלט (טוטאלוס)."
            elif asks_third_party:
                answer = "לא מצאתי במאגר רשימת מסמכים ייעודית לתביעת צד שלישי."
            elif asks_injury:
                answer = "לא מצאתי במאגר רשימת מסמכים ייעודית לתביעת נזקי גוף."
            else:
                answer = "לא מצאתי במאגר מקור ממוקד שמפרט אילו מסמכים צריך לצרף לתביעת רכב."
            return {"answer": answer, "sources": []}
        ranked = document_pages

    elif intent == "report_damage":
        damage_pages = [(score, page) for score, page in ranked if page_has_damage_report_evidence(page)]
        if not damage_pages:
            return {
                "answer": "לא מצאתי במאגר הוראות ממוקדות לדיווח על נזק לרכב.",
                "sources": [],
            }
        ranked = damage_pages

    # If the question asks specifically about third-party insurance, prefer pages
    # whose titles are actually about that product rather than pages mentioning it in menus.
    asks_difference = any(word in normalized_question for word in ("ההבדל", "הבדלים", "להבדיל", "שונה"))
    asks_third_party = "צד ג" in normalized_question or "צד שלישי" in normalized_question
    if asks_third_party and not asks_difference and intent != "claim_documents":
        third_party_pages = [
            (score, page) for score, page in ranked
            if ("צד ג" in normalize_text(page.get("title", ""))
                or "צד שלישי" in normalize_text(page.get("title", "")))
        ]
        if third_party_pages:
            ranked = third_party_pages
        else:
            return {
                "answer": "לא מצאתי בדפים שנאספו מקור ממוקד מספיק על ביטוח צד ג'. אפשר לנסח את השאלה אחרת?",
                "sources": [],
            }

    # Avoid returning fraud-database pages for ordinary questions about compulsory insurance.
    if "חובה" in normalized_question and not any(
        word in normalized_question for word in ("הונאה", "הונאות", "מאגר")
    ):
        non_fraud_pages = [
            (score, page) for score, page in ranked
            if "הונאות" not in normalize_text(page.get("title", ""))
            and "מאגר המידע" not in normalize_text(page.get("title", ""))
        ]
        if non_fraud_pages:
            ranked = non_fraud_pages

    # For claim questions, ignore pages that only mention claims in navigation menus.
    if any(word in normalized_question for word in ("תביעה", "תביעות", "מגישים", "להגיש", "הגשת", "תבע")):
        claim_relevant = []
        for score, page in ranked:
            title = normalize_text(page.get("title", ""))
            # Exclude pages whose subject is discounts/history of claims, even if
            # their titles contain the word "תביעות".
            is_claim_history_or_pricing = any(term in title for term in (
                "היעדר תביעות", "העדר תביעות", "הנחת היעדר", "הנחת העדר",
                "הטבה בביטוח", "היסטוריית תביעות", "עבר ביטוחי",
            ))
            is_unrelated_accident_article = any(term in title for term in (
                "תאונת שרשרת", "סיבות לתאונות", "סיבות לתאונת",
            ))
            is_claim_or_accident_guide = any(term in title for term in (
                "הגשת תביעת רכב", "טופס תביעה", "הודעה על מקרה ביטוח",
                "תביעת רכב", "תביעת ביטוח", "מה עושים במקרה של תאונת דרכים",
                "ביטוח רכב אחרי תאונה", "אחרי תאונת דרכים",
            ))
            if is_claim_or_accident_guide and not is_unrelated_accident_article and not is_claim_history_or_pricing:
                # Keep direct claim/accident pages, not product pages whose menus mention claims.
                claim_relevant.append((score + 5.0, page))
        if claim_relevant:
            ranked = claim_relevant

    if not ranked or ranked[0][0] < 2.5:
        return {
            "answer": "לא מצאתי בדפים שנאספו מידע מספיק כדי לענות בביטחון. אפשר לנסח את השאלה אחרת?",
            "sources": [],
        }

    best_score = ranked[0][0]
    threshold = max(2.5, best_score * 0.24)
    relevant: list[tuple[float, dict]] = []
    seen_urls: set[str] = set()

    for score, page in ranked:
        if score < threshold:
            break
        url = page.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        relevant.append((score, page))
        if len(relevant) == 4:
            break

    if not relevant:
        relevant = [ranked[0]]

    relevant_pages = []
    sources = []
    for _, page in relevant:
        if intent == "claim_documents":
            snippet = make_claim_documents_snippet(page, question)
        else:
            snippet = make_snippet(page.get("text", ""), direct_terms, extra_phrases, question)
        if not snippet:
            continue
        
        snippet = snippet.replace(
            "למסך אישור אי הגשת תביעה",
            "למסמך אישור אי־הגשת תביעה",
        )

        relevant_pages.append(page)
        sources.append({
            "title": display_source_title(page, question),
            "url": page["url"],
            "snippet": snippet,
        })

    # For intent-specific queries, an empty excerpt means we did not retrieve evidence
    # that directly answers the question. Do not return a misleading page link as an answer.
    if not sources and intent in ("contact", "claim_documents", "report_damage"):
        if intent == "contact":
            answer = "לא נמצא במאגר קטע שמציג פרטי קשר מאומתים של הפניקס."
        elif intent == "claim_documents":
            answer = "לא נמצא במאגר קטע שמפרט את המסמכים לצירוף לתביעת רכב. בדקי את טופס התביעה המתאים."
        else:
            answer = "לא נמצא במאגר קטע שמסביר כיצד לדווח על נזק לרכב."
        return {"answer": answer, "sources": []}

    # Keep a direct page link as a fallback for general queries if the corpus has no clean excerpt.
    if not sources and relevant:
        page = relevant[0][1]
        relevant_pages = [page]
        sources = [{
            "title": page.get("title") or "עמוד באתר הפניקס",
            "url": page["url"],
            "snippet": "",
        }]

    answer = compose_answer(question, relevant_pages, sources)
    return {"answer": answer, "sources": sources}
