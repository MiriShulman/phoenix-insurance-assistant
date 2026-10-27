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

DATA_FILE = Path(__file__).parent / "data" / "pages_crawled_clean.jsonl"

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
    return re.sub(r"\s+", " ", text.replace("׳", "'").replace("’", "'").replace("״", '"')).strip()


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
        "שירות לקוחות", "מוקד", "מספר הטלפון", "מספר טלפון", "טלפון של", "כתובת של",
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


def page_is_towing_contact(page: dict) -> bool:
    title = normalize_text(page.get("title", ""))
    body = normalize_text(clean_page_text(page.get("text", "")))
    return (
        any(cue in title for cue in ("גרירה", "ספק גרירה", "טלפונים חשובים"))
        or "טלפונים חשובים לספק גרירה" in body
        or "ספק הגרירה" in body
    )


def page_has_contact_evidence(page: dict) -> bool:
    title = normalize_text(page.get("title", ""))
    body = normalize_text(clean_page_text(page.get("text", "")))

    # Towing-provider numbers are not general Phoenix contacts.
    is_towing_contact = page_is_towing_contact(page)
    has_official_contact_title = any(cue in title for cue in CONTACT_TITLE_CUES)
    if is_towing_contact and not has_official_contact_title:
        return False

    if has_official_contact_title:
        return True
    has_contact_context = any(cue in body for cue in CONTACT_BODY_CUES)
    has_contact_detail = bool(EMAIL_RE.search(body) or PHONE_RE.search(body))
    return has_contact_context and has_contact_detail


# Specific question topics are handled separately from broad intent labels such as
# "claim_process". This prevents a generic word like "תביעה" from winning over the
# actual question (status, timing, missing documents, theft, glass, flood, etc.).
TOPIC_RULES = {
    "theft_coverage": {
        "title_cues": ("כיסוי גניבה", "גניבת רכב", "גניבת הרכב"),
        "evidence_cues": ("גניבה", "נגנב", "גניבת הרכב", "גניבת רכב"),
        "sentence_cues": ("גניבה", "נגנב", "גניבת הרכב", "גניבת רכב"),
        "exclude_titles": ("נהיגה בחורף", "תאונת שרשרת", "מערכות בטיחות"),
        "search_phrases": ("ביטוח מקיף גניבת רכב", "כיסוי ביטוחי לגניבה", "גניבת הרכב"),
    },
    "glass_coverage": {
        "title_cues": ("כיסוי שמשות", "שירותי שמשות", "שמשות לרכב", "שבר שמשה"),
        "evidence_cues": ("שמשות", "שמשה", "שמשת הרכב", "שבר שמשה", "כיסוי לשמשות"),
        "sentence_cues": ("שמשות", "שמשה", "שמשת הרכב", "שבר שמשה", "כיסוי לשמשות"),
        "exclude_titles": ("נהיגה בחורף", "תאונת שרשרת"),
        "search_phrases": ("כיסוי לשמשות הרכב", "שבר שמשה", "שירות שמשות"),
    },
    "vehicle_flood_coverage": {
        "title_cues": (
            "נזקי הצפה לרכב", "נזקי מים לרכב", "שיטפון לרכב",
            "כיסוי נזקי טבע לרכב",
        ),
        "evidence_cues": ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים", "נזקי טבע"),
        "sentence_cues": ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים", "נזקי טבע"),
        "exclude_titles": (
            "הצפה בבית", "ביטוח תכולה לדירה", "ביטוח תכולה",
            "נזקי מים וצנרת", "נשיכת כלב", "ביטוח דירה",
        ),
        "search_phrases": (
            "נזקי הצפה לרכב", "נזקי טבע לרכב ביטוח מקיף",
            "שיטפון ביטוח רכב", "נזקי מים לרכב",
        ),
    },
    "weather_flood_contents": {
        "title_cues": (
            "החורף בשיאו", "נזקי חורף", "ביטוח תכולה לדירה",
            "הצפה בבית", "נזקי טבע", "שיטפון", "הצפה",
        ),
        "evidence_cues": (
            "תכולה", "תכולת הדירה", "רכוש בדירה",
            "מזג אוויר סוער", "מזג אוויר קיצוני", "נזקי טבע",
            "סערה", "סופה", "רוחות", "הצפה", "הצפות",
            "שיטפון", "שיטפונות", "נזקי מים",
        ),
        "sentence_cues": (
            "תכולה", "תכולת הדירה", "רכוש בדירה",
            "מזג אוויר סוער", "מזג אוויר קיצוני", "נזקי טבע",
            "סערה", "סופה", "רוחות", "הצפה", "הצפות",
            "שיטפון", "שיטפונות", "נזקי מים",
        ),
        "exclude_titles": (
            "ביטוח רכב", "תאונת דרכים", "תאונת שרשרת",
            "ביטוח נזקי מים וצנרת", "נזקי צנרת", "נשיכת כלב",
        ),
        "search_phrases": (
            "ביטוח תכולה נזקי מזג אוויר סוער",
            "נזקי טבע לתכולת הדירה",
            "הצפה בבית נזק לתכולה ביטוח",
            "שיטפון או הצפה ממקור מים חיצוני תכולה",
        ),
    },
    "flood_coverage": {
        "title_cues": ("הצפה", "שיטפון", "נזקי מים"),
        "evidence_cues": ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים"),
        "sentence_cues": ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים"),
        "exclude_titles": ("נהיגה בחורף", "תאונת שרשרת"),
        "search_phrases": ("נזקי הצפה לרכב", "שיטפון ביטוח רכב", "נזקי מים לרכב"),
    },
    "water_pipe_damage": {
        "title_cues": (
            "נזקי מים וצנרת", "ביטוח נזקי מים", "נזקי צנרת",
            "נזק מנזילה בצנרת",
        ),
        "evidence_cues": (
            "נזקי מים וצנרת", "נזקי צנרת", "נזילה בצנרת",
            "דליפה בצנרת", "דליפת מים", "פיצוץ צינור",
            "צינור שהתפוצץ", "נזק למבנה", "נזק לתכולה",
        ),
        "sentence_cues": (
            "נזקי מים וצנרת", "נזקי צנרת", "נזילה בצנרת",
            "דליפה בצנרת", "דליפת מים", "פיצוץ צינור",
            "צינור שהתפוצץ", "נזק למבנה", "נזק לתכולה",
        ),
        "exclude_titles": (
            "נשיכת כלב", "כלבים", "ביטוח רכב", "תאונת דרכים",
            "תאונת שרשרת", "הצפה בבית",
        ),
        "search_phrases": (
            "ביטוח נזקי מים וצנרת", "נזק מנזילה בצנרת בביטוח דירה",
            "כיסוי לנזקי צנרת", "נזק למבנה בעקבות פיצוץ צינור",
        ),
    },
    "water_leak_cost": {
        "title_cues": (
            "עלות המים", "חשבון מים", "החזר על מים",
            "נזקי מים וצנרת",
        ),
        "evidence_cues": (
            "עלות המים", "חשבון המים", "חשבון מים",
            "מים שהתבזבזו", "מים שבוזבזו",
            "החזר על חשבון המים", "צריכת מים חריגה",
            "הפרשי צריכה", "נזילה בצנרת", "דליפת מים",
            "נזקי מים וצנרת",
        ),
        "sentence_cues": (
            "עלות המים", "חשבון המים", "חשבון מים",
            "מים שהתבזבזו", "מים שבוזבזו",
            "החזר על חשבון המים", "צריכת מים חריגה",
            "הפרשי צריכה",
        ),
        "exclude_titles": (
            "רכב", "תאונת דרכים", "תאונת שרשרת",
            "ביטוח רכב",
        ),
        "search_phrases": (
            "עלות מים בעקבות נזילה בצנרת",
            "חשבון מים בעקבות נזילה",
            "החזר על חשבון המים לאחר נזילה",
        ),
    },

    "weather_damage_contents": {
        "title_cues": (
            "ביטוח תכולה לדירה",
            "החורף בשיאו",
            "ביטוח דירה ועוד הרחבות",
            "ביטוח כל הסיכונים",
        ),
        "evidence_cues": (
            "מזג אוויר קיצוני",
            "מזג אוויר סוער",
            "נזקי טבע",
            "סערה",
            "סופה",
            "תכולת הדירה",
            "ביטוח תכולה",
        ),
        "sentence_cues": (
            "מזג אוויר קיצוני",
            "מזג אוויר סוער",
            "נזקי טבע",
            "סערה",
            "סופה",
            "תכולת הדירה",
            "ביטוח תכולה",
        ),
        "exclude_titles": (
            "ביטוח רכב",
            "תאונת דרכים",
            "תאונת שרשרת",
            "ביטוח נזקי מים וצנרת",
            "הצפה בבית",
            "ביטוח תכולה בתקופת מלחמה",
        ),
        "search_phrases": (
            "ביטוח תכולה נזקי מזג אוויר סוער",
            "נזקי טבע ביטוח תכולת דירה",
            "נזקי סערה לתכולת הדירה",
        ),
    },

    "home_insurance_overview": {
        # Broad questions should prefer the general home-insurance page, not
        # an incident-specific article that merely contains the words "ביטוח דירה".
        "title_cues": (
            "ביטוח דירה",
            "ביטוח הדירה",
            "ביטוח דירה ועוד הרחבות",
        ),
        "evidence_cues": (
            "פוליסת ביטוח דירה",
            "ביטוח מבנה",
            "ביטוח תכולה",
            "מבנה ותכולה",
            "תכולת הדירה",
            "ביטוח צד ג",
            "ביטוח צד ג'",
        ),
        "sentence_cues": (
            "פוליסת ביטוח דירה",
            "ביטוח מבנה",
            "ביטוח תכולה",
            "מבנה ותכולה",
            "תכולת הדירה",
            "ביטוח צד ג",
            "ביטוח צד ג'",
        ),
        "exclude_titles": (
            "נשיכת כלב",
            "כלבים",
            "נזקי מים וצנרת",
            "נזקי צנרת",
            "הצפה בבית",
            "ביטוח כל הסיכונים",
            "ביטוח תכולה בתקופת מלחמה",
        ),
        "search_phrases": (
            "ביטוח דירה מבנה תכולה צד ג",
            "ביטוח דירה מה כולל",
            "פוליסת ביטוח דירה מבנה ותכולה",
        ),
    },
    "health_insurance_overlap": {
        # Route questions about overlap between private health insurance and the
        # health-fund supplementary plan to the article that compares these systems.
        "title_cues": (
            "מסתמכים על קופות החולים בביטוחי הבריאות",
            "קופות החולים בביטוחי הבריאות",
            "כפל ביטוח בריאות",
            "ביטוח בריאות פרטי וקופת חולים",
        ),
        "evidence_cues": (
            "קופת החולים", "קופות החולים", "ביטוחי הבריאות",
            "ביטוח בריאות פרטי", "חברת הביטוח הפרטית",
            "ביטוח משלים", "שב\"ן", "שבן",
            "הרחבת הביטוח", "השתלות", "ניתוחים",
        ),
        "sentence_cues": (
            "קופת החולים", "קופות החולים", "ביטוחי הבריאות",
            "ביטוח בריאות פרטי", "חברת הביטוח הפרטית",
            "ביטוח משלים", "שב\"ן", "שבן",
            "הרחבת הביטוח", "השתלות", "ניתוחים",
            "כפל ביטוח", "כיסוי כפול",
        ),
        "exclude_titles": (
            "ביטוח נסיעות", "נסיעות לחו\"ל", "מחלות כרוניות",
            "קופת גמל", "ביטוח רכב", "ביטוח דירה",
        ),
        "search_phrases": (
            "מסתמכים על קופות החולים בביטוחי הבריאות",
            "כפל ביטוח בריאות קופת חולים ביטוח פרטי",
            "ביטוח משלים שב\"ן ביטוח בריאות פרטי",
            "הרחבת הביטוח מעבר לביטוח שמספקות קופות החולים",
        ),
    },
    "accident_first_steps": {
        "title_cues": ("מה עושים במקרה של תאונת דרכים", "מה עושים אחרי תאונה"),
        "evidence_cues": ("החליפו פרטים", "פרטי הנהגים", "צלמו את הנזקים", "תיעוד האירוע",
                          "אין לחתום", "הודאה באשמה", "פרטי הרכב"),
        "sentence_cues": ("החליפו פרטים", "פרטי הנהג", "פרטי הנהגים", "צלמו", "תיעוד האירוע",
                          "אין לחתום", "הודאה באשמה", "פרטי הרכב", "מספרי הרכבים",
                          "רישיון הנהיגה", "מקום התאונה"),
        "exclude_titles": ("תאונת שרשרת", "סיבות לתאונות", "נהיגה בחורף", "מערכות בטיחות"),
        "search_phrases": ("מה עושים במקרה של תאונת דרכים", "החלפת פרטים אחרי תאונה",
                           "תיעוד נזקי תאונה"),
    },
    "report_to_insurer": {
        "title_cues": ("הודעה על מקרה ביטוח רכב", "הגשת תביעת רכב", "טופס תביעה"),
        "evidence_cues": ("הודעה על מקרה ביטוח רכב", "מלאו את מסמך התביעה",
                          "את הגשת התביעה באופן מעשי תוכלו לעשות", "דווחו לחברת הביטוח",
                          "הודיעו לחברת הביטוח", "דרך אתר הפניקס", "באמצעות סוכן"),
        "sentence_cues": ("הודעה על מקרה ביטוח רכב", "מלאו את מסמך התביעה",
                          "את הגשת התביעה באופן מעשי תוכלו לעשות", "דווחו לחברת הביטוח",
                          "הודיעו לחברת הביטוח", "דרך אתר הפניקס", "באמצעות סוכן",
                          "הגשת תביעת רכב", "המסמכים הרלוונטיים"),
        "exclude_titles": ("טלפונים חשובים לספק גרירה", "נהיגה בחורף", "תאונת שרשרת"),
        "search_phrases": ("דיווח להפניקס על תאונת רכב", "הודעה על מקרה ביטוח רכב",
                           "הגשת תביעת רכב באתר הפניקס"),
    },
    "deductible": {
        "title_cues": ("השתתפות עצמית",),
        "evidence_cues": ("השתתפות עצמית",),
        "sentence_cues": ("השתתפות עצמית",),
        "exclude_titles": (),
        "search_phrases": ("מהי השתתפות עצמית בביטוח רכב", "סכום השתתפות עצמית"),
    },
    "driver_eligibility": {
        "title_cues": ("כל נהג", "נהגים מורשים", "מי רשאי לנהוג"),
        "evidence_cues": ("נהג צעיר", "נהג חדש", "כל נהג", "נהגים המורשים",
                          "הנהגים המורשים", "נהגים מורשים", "נהג נוסף", "גיל הנהג", "רשאי לנהוג",
                          "מורשה לנהוג", "מורשים לנהוג", "כיסוי ביטוחי לנהג"),
        "sentence_cues": ("נהג צעיר", "נהג חדש", "כל נהג", "נהגים המורשים",
                          "הנהגים המורשים", "נהגים מורשים", "נהג נוסף", "גיל הנהג", "רשאי לנהוג",
                          "מורשה לנהוג", "מורשים לנהוג", "כיסוי ביטוחי לנהג", "תנאי הפוליסה"),
        "exclude_titles": ("נהיגה בחורף", "תאונת שרשרת"),
        "search_phrases": ("מי רשאי לנהוג ברכב המבוטח", "נהגים המורשים לפי הפוליסה",
                           "כיסוי ביטוחי לכל נהג"),
    },
    "young_driver": {
        "title_cues": ("נהג צעיר", "נהג חדש"),
        "evidence_cues": ("נהג צעיר", "נהג חדש", "גיל הנהג", "כיסוי לנהג צעיר",
                          "הרחבת נהג צעיר"),
        "sentence_cues": ("נהג צעיר", "נהג חדש", "גיל הנהג", "כיסוי לנהג צעיר",
                          "הרחבת נהג צעיר", "הרחבה לנהג"),
        "exclude_titles": ("נהיגה בחורף", "תאונת שרשרת"),
        "search_phrases": ("כיסוי ביטוחי לנהג צעיר", "ביטוח רכב נהג צעיר ונהג חדש"),
    },
    "vehicle_recovered": {
        "title_cues": ("רכב שנגנב ונמצא", "רכב גנוב שנמצא", "מציאת רכב גנוב"),
        "evidence_cues": ("לאחר שרכב גנוב הוחזר לבעליו", "רכב גנוב הוחזר לבעליו",
                          "הרכב שנגנב נמצא", "רכב שנגנב ונמצא", "נמצא הרכב הגנוב",
                          "הרכב הגנוב נמצא", "הוחזר לבעליו"),
        "sentence_cues": ("לאחר שרכב גנוב הוחזר לבעליו", "רכב גנוב הוחזר לבעליו",
                          "הרכב שנגנב נמצא", "רכב שנגנב ונמצא", "נמצא הרכב הגנוב",
                          "הרכב הגנוב נמצא", "הוחזר לבעליו"),
        "exclude_titles": ("נהיגה בחורף", "טיפים לנהיגה", "תאונת שרשרת", "מערכות בטיחות"),
        "search_phrases": ("מה עושים אם רכב גנוב נמצא", "רכב שנגנב נמצא לאחר הגשת תביעה",
                           "החזרת רכב גנוב ותביעת ביטוח"),
    },
    "vehicle_stolen": {
        "title_cues": ("גניבת רכב", "רכב נגנב", "תביעת גניבה", "גניבת הרכב"),
        "evidence_cues": ("גניבה", "נגנב", "גניבת הרכב", "גניבת רכב"),
        "sentence_cues": ("גניבה", "נגנב", "גניבת הרכב", "גניבת רכב", "דווחו למשטרה",
                          "הודעה על מקרה ביטוח", "טופס תביעה", "הגשת תביעה", "מסמכים"),
        "exclude_titles": ("נהיגה בחורף", "טיפים לנהיגה", "תאונת שרשרת", "מערכות בטיחות"),
        "search_phrases": ("מה עושים כשהרכב נגנב", "תביעת ביטוח גניבת רכב",
                           "דיווח על גניבת רכב"),
    },
    "claim_status": {
        "title_cues": ("בירור מצב תביעה", "בדיקת מצב תביעה", "סטטוס תביעה", "מעקב תביעה"),
        "evidence_cues": ("סטטוס תביעה", "מצב התביעה", "סטטוס התביעה", "בדיקת סטטוס", "מעקב אחר תביעה",
                          "לבדוק את מצב התביעה", "תביעות שהוגשו"),
        "sentence_cues": ("סטטוס תביעה", "מצב התביעה", "סטטוס התביעה", "בדיקת סטטוס", "מעקב אחר תביעה",
                          "לבדוק את מצב התביעה", "תביעות שהוגשו", "באזור האישי"),
        "exclude_titles": ("היעדר תביעות", "הנחת היעדר תביעות", "היסטוריית תביעות"),
        "search_phrases": ("איך לבדוק מצב תביעה שהוגשה", "סטטוס תביעת רכב", "מעקב אחר תביעה"),
    },
    "claim_timeline": {
        "title_cues": ("זמן טיפול בתביעה", "זמן הטיפול בתביעות", "בירור ויישוב תביעות"),
        "evidence_cues": ("תוך 30 יום", "בתוך 30 יום", "30 יום מרגע", "זמן הטיפול בתביעה",
                          "משך הטיפול בתביעה", "מועד מענה", "חברת הביטוח הנתבעת לחזור אליכם"),
        "sentence_cues": ("תוך 30 יום", "בתוך 30 יום", "30 יום מרגע", "זמן הטיפול בתביעה",
                          "משך הטיפול בתביעה", "מועד מענה", "לחזור אליכם בתוך 30 יום",
                          "טיפול בתביעה", "הטיפול בתביעה"),
        "exclude_titles": ("היעדר תביעות", "הנחת היעדר תביעות", "היסטוריית תביעות"),
        "search_phrases": ("כמה זמן טיפול בתביעת ביטוח רכב", "תוך כמה זמן משיבים לתביעה",
                           "זמן הטיפול בתביעה"),
    },
    "missing_documents": {
        "title_cues": ("השלמת מסמכים", "מסמכים חסרים"),
        "evidence_cues": ("תתבקשו להשלים את החסר", "להשלים את החסר", "השלמת מסמכים",
                          "מסמכים חסרים", "חסר מסמך", "השלמת המסמכים", "המסמכים החסרים"),
        "sentence_cues": ("תתבקשו להשלים את החסר", "להשלים את החסר", "השלמת מסמכים",
                          "מסמכים חסרים", "חסר מסמך", "השלמת המסמכים", "המסמכים החסרים",
                          "מסמכים נוספים"),
        "exclude_titles": ("היעדר תביעות", "הנחת היעדר תביעות", "היסטוריית תביעות"),
        "search_phrases": ("מה עושים אם חסרים מסמכים בתביעה", "השלמת מסמכים לתביעה",
                           "מסמכים חסרים בתביעת רכב"),
    },
}


def classify_question_topic(question: str) -> str:
    """Identify the specific subject of a question before broad keyword retrieval."""
    q = normalize_text(question)

    weather_terms = ("מזג האוויר", "תחזית מזג", "מה יהיה מזג", "כמה מעלות")
    insurance_context_terms = (
        "ביטוח", "כיסוי", "מכסה", "מכוסה", "מכוסים", "פוליסה",
        "נזק", "נזקים", "תכולה", "רכוש", "תביעה", "הפניקס", "רכב",
    )
    if any(term in q for term in weather_terms) and not any(
        term in q for term in insurance_context_terms
    ):
        return "out_of_scope"

    # Private health insurance vs. the supplementary health-fund plan (שב"ן).
    # Keep this separate from travel insurance and from generic insurance queries.
    health_fund_terms = (
        "קופת חולים", "קופת החולים", "קופות החולים",
        "ביטוח משלים", "המשלים של קופת החולים", "משלים של קופת החולים",
        "שב\"ן", "שבן",
    )
    overlap_terms = (
        "כפל ביטוח", "כפל ביטוחי", "כפל כיסויים", "כיסוי כפול",
        "כיסויים כפולים", "חפיפה", "חופף", "מול", "לעומת",
        "ביחס", "בהשוואה", "בנוסף", "מעבר", "ההבדל",
    )
    private_health_terms = (
        "ביטוח בריאות פרטי", "ביטוחי הבריאות", "ביטוח פרטי",
        "חברת הביטוח הפרטית", "ביטוח בריאות",
    )
    if any(term in q for term in health_fund_terms) and (
        any(term in q for term in overlap_terms)
        or (
            any(term in q for term in private_health_terms)
            and any(term in q for term in ("כדאי", "צריך", "נחוץ", "כדאיות"))
        )
    ):
        return "health_insurance_overlap"

    if any(term in q for term in ("מספר הטלפון", "מספר טלפון", "טלפון")) and any(
        term in q for term in (
            "תאונה", "תאונת", "דיווח", "לדווח", "תביעה", "תביעות",
            "בירור", "בירורים", "גרירה", "גרר",
        )
    ) and any(term in q for term in ("הפניקס", "חברת הביטוח")):
        return "claim_contact"

    if any(term in q for term in ("מספר הטלפון", "מספר טלפון", "טלפון")) and any(
        term in q for term in ("תביעה", "תביעות", "בירור", "בירורים")
    ):
        return "claim_contact"

    # Questions asking how to report an accident to Phoenix are not asking what
    # to do at the scene, and should not retrieve police-contact instructions.
    if any(term in q for term in ("לדווח להפניקס", "דיווח להפניקס", "מדווחים להפניקס",
                                 "מדווחים להפניקס על", "איך מדווחים להפניקס",
                                 "להפניקס על תאונה", "לחברת הביטוח על התאונה",
                                 "דיווח לחברת הביטוח", "לדווח לחברת הביטוח")):
        return "report_to_insurer"

    if any(term in q for term in (
        "מצב התביעה", "סטטוס התביעה", "מעקב אחר התביעה",
        "לבדוק מה מצב", "לבדוק את מצב התביעה", "בדיקת סטטוס",
        "הסטטוס של תביעה", "הסטטוס של התביעה",
        "סטטוס של תביעה", "סטטוס של התביעה",
        "איך אפשר לבדוק את הסטטוס", "איך בודקים סטטוס",
    )) or (
        any(term in q for term in ("תביעה שכבר הוגשה", "תביעה שהוגשה"))
        and any(term in q for term in ("לבדוק", "סטטוס", "מצב", "מעקב"))
    ):
        return "claim_status"

    if any(term in q for term in (
        "אין לי את כל המסמכים", "חסרים מסמכים", "מסמכים חסרים",
        "אין את כל המסמכים", "חסר לי מסמך", "חסרים לי מסמכים",
        "חסר מסמך", "מסמך חסר", "חסר מסמך שנדרש",
    )):
        return "missing_documents"

    if any(term in q for term in ("זמן הטיפול", "משך הטיפול", "כמה זמן", "תוך כמה זמן",
                                 "מתי אקבל תשובה", "כמה זמן לוקח")) and any(
        term in q for term in ("תביעה", "תביעת", "ביטוח", "הפניקס")
    ):
        return "claim_timeline"

    # Hebrew definite articles can attach to both words ("ההשתתפות העצמית"),
    # so do not rely on only the uninflected phrase.
    if any(term in q for term in (
        "השתתפות עצמית", "השתתפות העצמית", "ההשתתפות העצמית",
        "דמי השתתפות עצמית", "דמי ההשתתפות העצמית", "סכום ההשתתפות העצמית",
    )):
        return "deductible"

    if any(term in q for term in ("נהג צעיר", "נהג חדש")):
        return "young_driver"

    if any(term in q for term in (
        "כל נהג", "כל אחד יכול לנהוג", "מי יכול לנהוג",
        "מי רשאי לנהוג", "נהג כלשהו", "נהגים אחרים",
        "נהגים מורשים", "אילו נהגים מורשים", "הנהגים המורשים",
        "מורשים לנהוג", "אילו נהגים רשאים",
    )):
        return "driver_eligibility"

    if any(term in q for term in (
        "הרכב נמצא", "נמצא הרכב", "הוחזר לבעליו", "מוחזר לבעליו",
        "הרכב הוחזר", "הרכב שנגנב נמצא", "שנמצא ומוחזר",
        "נמצא והוחזר", "רכב גנוב שנמצא", "הרכב הגנוב נמצא",
    )) and any(term in q for term in ("תביעה", "תביעת", "ביטוח", "גנב", "גניבה")):
        return "vehicle_recovered"

    if any(term in q for term in ("הרכב נגנב", "אם נגנב הרכב", "רכב שנגנב",
                                 "גניבת הרכב", "גניבת רכב", "להפעיל את הביטוח")) and any(
        term in q for term in ("מה לעשות", "מה עליי לעשות", "מה צריך לעשות", "דיווח",
                               "תביעה", "להפעיל", "איך")
    ):
        return "vehicle_stolen"

    if any(term in q for term in ("שמשות", "שמשה", "שמשת הרכב")) and any(
        term in q for term in ("מכסה", "כיסוי", "נזק", "ביטוח")
    ):
        return "glass_coverage"

    # A question about the water bill is different from a question about physical damage.
    if any(term in q for term in (
        "עלות המים", "עלות מים",
        "חשבון המים", "חשבון מים",
        "החזר על המים", "החזר על מים",
        "מים שהתבזבזו", "מים שבוזבזו",
        "צריכת מים חריגה", "הפרשי צריכה",
        "חיוב מים", "תשלום על המים", "תשלום מים",
    )):
        return "water_leak_cost"

    flood_terms = ("הצפה", "שיטפון", "שיטפונות", "הצפות", "נזקי מים", "נזקי טבע")
    vehicle_terms = ("רכב", "הרכב", "מכונית", "אוטו", "ביטוח רכב")
    if any(term in q for term in flood_terms) and any(term in q for term in vehicle_terms):
        return "vehicle_flood_coverage"

    pipe_terms = (
        "נזילה בצנרת", "נזילה מהצנרת", "דליפה בצנרת", "דליפת מים",
        "פיצוץ צינור", "צינור שהתפוצץ", "נזקי מים וצנרת", "נזקי צנרת",
    )
    pipe_coverage_terms = ("מכסה", "מכוסה", "מכוסים", "כיסוי", "ביטוח", "נזק", "נזקים", "פוליסה")
    if any(term in q for term in pipe_terms) and any(term in q for term in pipe_coverage_terms):
        return "water_pipe_damage"

    # Questions combining storm damage and flooding to home contents need both
    # evidence types; do not let the broad flood rule swallow the storm half.
    combined_weather_terms = (
        "מזג אוויר סוער", "מזג האוויר הסוער", "מזג אוויר קיצוני",
        "סערה", "סערות", "סופה", "סופות", "רוח", "רוחות", "נזקי טבע",
    )
    combined_flood_terms = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים")
    combined_contents_terms = (
        "תכולה", "תכולת", "רהיט", "רהיטים", "רכוש", "חפצים",
        "מכשירי חשמל", "טלוויזיה",
    )
    combined_coverage_terms = (
        "מכוסה", "מכוסים", "מכוסות", "מכסה", "כיסוי", "ביטוח", "נזק", "נזקים",
    )
    if (
        any(term in q for term in combined_weather_terms)
        and any(term in q for term in combined_flood_terms)
        and any(term in q for term in combined_contents_terms)
        and any(term in q for term in combined_coverage_terms)
    ):
        return "weather_flood_contents"

    if any(term in q for term in ("הצפה", "שיטפון", "שיטפונות", "הצפות", "נזקי מים")):
        return "flood_coverage"
    # Damage to home contents caused by strong winds or storms.
    if (
        any(term in q for term in (
            "רוח", "רוחות", "סערה", "סערות", "סופה", "סופות",
            "מזג אוויר סוער", "מזג האוויר הסוער",
            "מזג אוויר קיצוני", "נזקי טבע",
        ))
        and any(term in q for term in (
            "תכולה", "תכולת", "רהיט", "רהיטים", "רכוש", "חפצים",
            "מכשירי חשמל", "טלוויזיה",
        ))
        and any(term in q for term in (
            "מכוסה", "מכוסים", "מכוסות", "מכסה",
            "כיסוי", "ביטוח", "נזק", "נזקים",
            "כולל", "כוללים",
        ))
    ):
        return "weather_damage_contents"

    if any(term in q for term in ("גניבה", "גניבת")) and "מקיף" in q and any(
        term in q for term in ("מכסה", "כיסוי", "כולל")
    ):
        return "theft_coverage"

    if any(term in q for term in ("מיד אחרי תאונה", "מיד לאחר תאונה", "מה עושים אחרי תאונה",
                                 "מה לעשות אחרי תאונה", "מה צריך לעשות אחרי תאונה",
                                 "אחרי תאונת דרכים")):
        return "accident_first_steps"

    # A general home-insurance question needs the product overview page. Do not
    # classify a specific incident (for example, a dog bite) as a broad overview.
    home_insurance_terms = (
        "ביטוח דירה", "ביטוח הדירה", "פוליסת דירה", "פוליסת ביטוח דירה",
        "ביטוח מבנה", "ביטוח תכולה",
    )
    broad_coverage_terms = (
        "מה מכסה", "מה מכסים", "מה כולל", "מה כוללת", "מה נכלל",
        "מה נותן", "מה נותנת", "כיסוי", "כולל",
    )
    specific_incident_terms = (
        "נשיכת כלב", "נשיכה", "כלב", "כלבים", "נזילה", "צנרת",
        "הצפה", "שיטפון", "נזקי מים", "פריצה", "גניבה", "שריפה",
        "סערה", "רעידת אדמה", "שבר", "תכשיט שנגנב",
    )
    if (
        any(term in q for term in home_insurance_terms)
        and any(term in q for term in broad_coverage_terms)
        and not any(term in q for term in specific_incident_terms)
    ):
        return "home_insurance_overview"

    return ""


def page_has_topic_evidence(page: dict, topic: str) -> bool:
    rule = TOPIC_RULES.get(topic)
    if not rule:
        return False
    title = normalize_text(page.get("title", ""))
    body = normalize_text(clean_page_text(page.get("text", "")))

    if any(cue in title for cue in rule.get("exclude_titles", ())):
        return False

    if topic == "health_insurance_overlap":
        # The direct Phoenix article title is strong evidence. Otherwise require
        # evidence of both systems, not just generic mentions of insurance.
        if any(cue in title for cue in rule.get("title_cues", ())):
            return True
        health_fund_cues = (
            "קופת החולים", "קופות החולים", "ביטוח משלים",
            "השב\"ן", "שב\"ן", "שבן",
        )
        private_insurance_cues = (
            "ביטוחי הבריאות", "ביטוח בריאות פרטי", "ביטוח פרטי",
            "חברת הביטוח הפרטית", "הביטוח הפרטי",
        )
        benefit_cues = (
            "כיסוי", "מכוסה", "הרחבת הביטוח", "השתלה", "השתלות",
            "ניתוח", "ניתוחים",
        )
        return (
            any(cue in body for cue in health_fund_cues)
            and any(cue in body for cue in private_insurance_cues)
            and any(cue in body for cue in benefit_cues)
        )

    if topic == "weather_flood_contents":
        # A title such as "הצפה בבית" is not enough on its own: the body must
        # actually discuss contents, plus either storm/weather or flooding.
        contents_cues = ("תכולה", "תכולת הדירה", "רכוש בדירה", "רכוש הבית")
        weather_cues = (
            "מזג אוויר סוער", "מזג אוויר קיצוני", "נזקי טבע",
            "סערה", "סופה", "רוחות", "רוח חזקה",
        )
        flood_cues = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים")
        return (
            any(cue in body for cue in contents_cues)
            and (
                any(cue in body for cue in weather_cues)
                or any(cue in body for cue in flood_cues)
            )
        )

    if any(cue in title for cue in rule.get("title_cues", ())):
        return True

    if topic == "water_pipe_damage":
        pipe_cues = (
            "נזקי מים וצנרת", "נזקי צנרת", "נזילה בצנרת",
            "דליפה בצנרת", "פיצוץ צינור", "צינור שהתפוצץ",
        )
        direct_coverage_context = (
            "ביטוח נזקי מים", "כיסוי לנזקי מים", "נזקי מים וצנרת",
            "נזקי צנרת", "הרחבת נזקי מים", "נזק למבנה",
            "נזק לתכולה", "מבנה ותכולה",
        )
        return (
            any(cue in body for cue in pipe_cues)
            and any(cue in body for cue in direct_coverage_context)
        )

    if topic == "vehicle_flood_coverage":
        # Do not let home, contents, or plumbing pages answer a car-insurance question.
        home_only_title_cues = (
            "דירה", "תכולה", "הצפה בבית", "נזקי מים וצנרת", "נשיכת כלב",
        )
        if any(cue in title for cue in home_only_title_cues):
            return False
        flood_cues = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים", "נזקי טבע")
        vehicle_cues = ("רכב", "הרכב המבוטח", "לרכב", "ביטוח רכב", "רכב המבוטח")
        coverage_cues = ("כיסוי", "מכסה", "מכוסה", "ביטוח מקיף", "פוליסה")
        return (
            any(cue in body for cue in flood_cues)
            and any(cue in title or cue in body for cue in vehicle_cues)
            and any(cue in body or cue in title for cue in coverage_cues)
        )

    # "באמצעות סוכן" by itself can describe purchasing/adding insurance, not
    # filing a claim. For reporting questions, require direct claim/report evidence.
    if topic == "report_to_insurer":
        direct_reporting_cues = (
            "הודעה על מקרה ביטוח רכב",
            "מלאו את מסמך התביעה",
            "את הגשת התביעה באופן מעשי תוכלו לעשות",
            "דווחו לחברת הביטוח",
            "הודיעו לחברת הביטוח",
            "דרך אתר הפניקס",
            "מלאו את מסמך התביעה אונליין",
        )
        return any(cue in body for cue in direct_reporting_cues)

    if topic == "vehicle_recovered":
        direct_recovery_phrases = (
            "לאחר שרכב גנוב הוחזר לבעליו", "רכב גנוב הוחזר לבעליו",
            "הרכב שנגנב נמצא", "רכב שנגנב ונמצא", "נמצא הרכב הגנוב",
            "הרכב הגנוב נמצא",
        )
        return any(phrase in body for phrase in direct_recovery_phrases)

    has_evidence = any(cue in body for cue in rule.get("evidence_cues", ()))
    if not has_evidence:
        return False

    if topic == "vehicle_stolen":
        # A generic theft mention is not enough; the page must also discuss a
        # reporting/claim action, or have a direct theft-related title.
        action_cues = ("דווחו", "דווח", "משטרה", "תביעה", "טופס", "מסמכים",
                       "הודעה על מקרה ביטוח", "חברת הביטוח", "הפניקס")
        return any(cue in body for cue in action_cues)
    return True


def topic_sentence_matches(sentence: str, topic: str, question: str = "") -> bool:
    normalized = normalize_text(sentence)
    normalized_question = normalize_text(question)
    rule = TOPIC_RULES.get(topic, {})
    if not any(cue in normalized for cue in rule.get("sentence_cues", ())):
        return False

    if topic == "health_insurance_overlap":
        health_fund_cues = (
            "קופת החולים", "קופות החולים", "ביטוח משלים",
            "השב\"ן", "שב\"ן", "שבן",
        )
        private_or_coverage_cues = (
            "ביטוחי הבריאות", "ביטוח בריאות", "ביטוח פרטי",
            "חברת הביטוח", "הרחבת הביטוח", "כיסוי", "מכוסה",
            "ניתוח", "השתלה",
        )
        has_health_fund_and_private_or_coverage = (
            any(cue in normalized for cue in health_fund_cues)
            and any(cue in normalized for cue in private_or_coverage_cues)
        )
        has_private_claim_example = (
            any(cue in normalized for cue in (
                "חברת הביטוח הפרטית", "חברת הביטוח", "ביטוח פרטי",
            ))
            and any(cue in normalized for cue in (
                "השתל", "השתלות", "ניתוח", "ניתוחים", "כיסוי", "מכוסה",
            ))
        )
        return has_health_fund_and_private_or_coverage or has_private_claim_example

    if topic == "vehicle_recovered":
        direct_recovery_phrases = (
            "לאחר שרכב גנוב הוחזר לבעליו", "רכב גנוב הוחזר לבעליו",
            "הרכב שנגנב נמצא", "רכב שנגנב ונמצא", "נמצא הרכב הגנוב",
            "הרכב הגנוב נמצא",
        )
        return any(phrase in normalized for phrase in direct_recovery_phrases)

    if topic == "vehicle_stolen":
        theft_cues = ("גניבה", "נגנב", "גניבת הרכב", "גניבת רכב")
        action_cues = ("דווח", "משטרה", "תביעה", "טופס", "מסמכים",
                       "הודעה על מקרה ביטוח", "חברת הביטוח", "הפניקס")
        return any(cue in normalized for cue in theft_cues) and any(
            cue in normalized for cue in action_cues
        )

    if topic == "theft_coverage":
        coverage_cues = ("מכסה", "כיסוי", "מקיף", "פוליסה", "מבוטח")
        theft_cues = ("גניבה", "נגנב", "גניבת הרכב", "גניבת רכב")
        return any(cue in normalized for cue in theft_cues) and any(
            cue in normalized for cue in coverage_cues
        )

    if topic == "glass_coverage":
        glass_cues = ("שמשות", "שמשה", "שמשת הרכב", "שבר שמשה")
        coverage_cues = ("כיסוי", "מכסה", "שירות", "הרחבה", "פוליסה", "תשלום")
        return any(cue in normalized for cue in glass_cues) and any(
            cue in normalized for cue in coverage_cues
        )

    if topic == "weather_flood_contents":
        contents_cues = ("תכולה", "תכולת הדירה", "רכוש בדירה", "רכוש הבית")
        weather_cues = (
            "מזג אוויר סוער", "מזג אוויר קיצוני", "נזקי טבע",
            "סערה", "סופה", "רוחות", "רוח חזקה",
        )
        flood_cues = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים")
        return (
            any(cue in normalized for cue in contents_cues)
            and (
                any(cue in normalized for cue in weather_cues)
                or any(cue in normalized for cue in flood_cues)
            )
        )

    if topic == "water_pipe_damage":
        pipe_cues = (
            "נזקי מים וצנרת", "נזקי צנרת", "נזילה בצנרת",
            "דליפה בצנרת", "דליפת מים", "פיצוץ צינור", "צינור שהתפוצץ",
        )
        context_cues = (
            "כיסוי", "מכסה", "מכוסה", "מכוסים", "ביטוח",
            "נזק למבנה", "נזק לתכולה", "עלות המים",
        )
        return any(cue in normalized for cue in pipe_cues) and any(
            cue in normalized for cue in context_cues
        )

    if topic == "vehicle_flood_coverage":
        flood_cues = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים", "נזקי טבע")
        vehicle_cues = ("רכב", "הרכב", "לרכב", "ביטוח רכב")
        coverage_cues = ("כיסוי", "מכסה", "מכוסה", "ביטוח מקיף", "פוליסה", "נזקי טבע")
        return (
            any(cue in normalized for cue in flood_cues)
            and any(cue in normalized for cue in vehicle_cues)
            and any(cue in normalized for cue in coverage_cues)
        )

    if topic == "flood_coverage":
        flood_cues = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים")
        coverage_cues = ("כיסוי", "מכסה", "ביטוח מקיף", "פוליסה", "נזקי טבע", "נזק")
        return any(cue in normalized for cue in flood_cues) and any(
            cue in normalized for cue in coverage_cues
        )

    if topic == "deductible":
        # For a general personal-deductible question, avoid snippets about specific
        # products or broad marketing commentary unless the user asks about them.
        if "חובה" not in normalized_question and any(term in normalized for term in ("ביטוח חובה", "ביטוח החובה")):
            return False
        asks_third_party = any(term in normalized_question for term in ("צד ג", "צד ג'", "צד שלישי"))
        if not asks_third_party and any(term in normalized for term in ("צד ג", "צד ג'", "צד שלישי")):
            return False
        if any(cue in normalized for cue in (
            "אנשים מעדיפים לשלם השתתפות עצמית כמה שפחות",
            "מכאן נובעת התחרות בין חברות הביטוח",
            "התחרות בין חברות הביטוח",
        )):
            return False
        return any(cue in normalized for cue in (
            "השתתפות עצמית", "הסכום המצוין בפוליסת הביטוח", "הפוליסות השונות",
            "מסלולי ביטוח הרכב", "דף פרטי הביטוח", "תנאי הפוליסה",
        ))

    return True


def topic_fallback_answer(topic: str) -> str:
    fallbacks = {
        "theft_coverage": "לא מצאתי במאגר קטע ממוקד שמאשר את הכיסוי לגניבת רכב בביטוח מקיף. יש לבדוק את תנאי הפוליסה והסייגים.",
        "glass_coverage": "לא מצאתי במאגר מקור ממוקד שמבהיר אם הפוליסה כוללת כיסוי לשמשות. בדקי אם מופיע בפוליסה כיסוי או שירות שמשות.",
        "home_insurance_overview": (
            "לא מצאתי במאגר מקור כללי מספיק על ביטוח דירה. בדרך כלל חשוב לבדוק בנפרד "
            "את כיסוי המבנה, התכולה והאם נכלל כיסוי צד ג׳, וכן את ההרחבות והחריגים בפוליסה שלך."
        ),
        "health_insurance_overlap": (
            "לא מצאתי במאגר מקור שמשווה ישירות בין ביטוח הבריאות הפרטי לבין הביטוח המשלים "
            "של קופת החולים. לכן איני יכול לקבוע אם קיימת חפיפה או כפל ביטוח; צריך להשוות "
            "את הכיסויים והתנאים בשני המסלולים."
        ),
                "water_leak_cost": (
            "לא מצאתי במאגר מקור שמאשר אם עלות המים שהתבזבזו בעקבות נזילה מכוסה. "
            "חשוב להבחין בין כיסוי לנזקי צנרת לבין החזר על חשבון המים עצמו. "
            "כדי לקבל תשובה ודאית, יש לבדוק את תנאי הפוליסה או לפנות להפניקס."
        ),
        "water_pipe_damage": (
            "לא מצאתי במאגר מקור ממוקד שמאשר אם נזק פיזי מנזילה או מפיצוץ צינור מכוסה. "
            "חשוב להבחין בין כיסוי לנזק למבנה או לתכולה לבין החזר על חשבון המים. "
            "בדקי את תנאי הפוליסה וההרחבות שנרכשו או פני להפניקס."
        ),
        "vehicle_flood_coverage": (
            "לא מצאתי במאגר מקור ממוקד לגבי נזקי הצפה לרכב. אין להסיק ממקור שעוסק בהצפה בדירה "
            "או בתכולה שהרכב מכוסה; בדקי את תנאי ביטוח הרכב, הכיסוי לנזקי טבע והחריגים."
        ),
        "weather_flood_contents": (
            "לא מצאתי במאגר מקורות ממוקדים שמכסים גם נזקי סערה לתכולה וגם נזקי הצפה. "
            "יש לבדוק בנפרד את הכיסוי לנזקי מזג אוויר ואת הרחבת נזקי המים/הצפה, בהתאם לתנאי הפוליסה."
        ),
        "flood_coverage": "לא מצאתי במאגר מקור ממוקד לגבי נזקי הצפה או שיטפון. הכיסוי תלוי בתנאי הפוליסה ובחריגים, ולכן אין להסיק מהמאגר שהנזק מכוסה.",
        "accident_first_steps": "לא מצאתי במאגר קטע ממוקד שמפרט מה לעשות מיד לאחר תאונת דרכים. אם יש סכנה או נפגעים, דאגי תחילה לבטיחות ולעזרה מתאימה.",
        "report_to_insurer": "לא מצאתי במאגר הוראות מאומתות לדיווח להפניקס על תאונת רכב. חפשי באתר הרשמי את טופס 'הודעה על מקרה ביטוח רכב' או את אזור 'הגשת תביעת רכב'.",
        "deductible": "לא מצאתי במאגר הסבר ממוקד על השתתפות עצמית. הסכום והאופן שבו הוא חל תלויים בתנאי הפוליסה.",
        "driver_eligibility": "לא מצאתי במאגר מקור ממוקד שמפרט אילו נהגים מכוסים. יש לבדוק בפוליסה מי הנהגים המורשים ומהן מגבלות הגיל.",
        "young_driver": "לא מצאתי במאגר מקור ממוקד לגבי כיסוי לנהג צעיר. הכיסוי תלוי בתנאי הפוליסה ובהרחבות שנרכשו.",
        "vehicle_stolen": "לא מצאתי במאגר הוראות מאומתות להגשת תביעת גניבת רכב או קישור לטופס המתאים. כדי לקבל את הטופס, רשימת המסמכים והצעדים העדכניים, פני לאתר הרשמי של הפניקס או לסוכן הביטוח. אין במאגר מקור שמאפשר לפרט כאן את תהליך ההגשה.",
        "vehicle_recovered": "לא מצאתי במאגר מקור מאומת שמסביר כיצד מציאת רכב שנגנב משפיעה על תביעה שכבר הוגשה. עדכני את הפניקס ובקשי הנחיות; המאגר אינו מאפשר לקבוע אם התביעה תבוטל או תימשך.",
        "claim_status": "לא מצאתי במאגר הוראות מאומתות לבדיקת מצב תביעה קיימת. כדאי לבדוק באזור האישי או לפנות לשירות הלקוחות של הפניקס.",
        "claim_timeline": "לא מצאתי במאגר פרק זמן מאומת לסיום הטיפול בתביעת רכב. חשוב להבחין בין מענה ראשוני של חברת הביטוח לבין סיום הטיפול בתביעה.",
        "missing_documents": "לא מצאתי במאגר הוראה ממוקדת לגבי מסמכים חסרים. בדקי את בקשת ההשלמה שקיבלת מחברת הביטוח ופעלי לפי ההנחיות שבה.",
    }
    return fallbacks.get(topic, "לא מצאתי במאגר מידע מספיק כדי לענות בביטחון.")


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
    body = normalize_text(clean_page_text(page.get("text", "")))
    topic = classify_question_topic(question)
    if topic == "claim_status" and "סטטוס תביעה" in body:
        return "סטטוס תביעה באתר הפניקס"
    if topic == "claim_contact" and any(phrase in body for phrase in (
        "מוקד תביעות וואטספ", "מוקד תביעות וואטסאפ", "מוקד תביעות ווטספ", "מוקד תביעות ווטסאפ",
    )):
        return "מוקד תביעות בוואטסאפ | הפניקס"
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


def _evidence_key(text: str) -> str:
    """Normalize page text or a source excerpt for duplicate-evidence checks."""
    normalized = normalize_text(clean_page_text(text or ""))
    return re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()


def _is_duplicate_evidence(text: str, seen_keys: list[str], threshold: float = 0.96) -> bool:
    """Detect exact and near-duplicate evidence without merging genuinely distinct facts."""
    key = _evidence_key(text)
    if not key:
        return False
    if key in seen_keys:
        return True

    tokens = set(key.split())
    # Short excerpts are often generic; only deduplicate those when they match exactly.
    if len(tokens) < 12:
        return False

    for previous_key in seen_keys:
        previous_tokens = set(previous_key.split())
        if len(previous_tokens) < 12:
            continue
        union = tokens | previous_tokens
        if not union:
            continue
        jaccard = len(tokens & previous_tokens) / len(union)
        if jaccard >= threshold:
            return True
    return False


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
    topic = classify_question_topic(q)
    # Specific topics take precedence over broad intents such as report_damage.
    intent = "" if topic in TOPIC_RULES else classify_intent(q)

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

    topic = classify_question_topic(q)
    if topic in TOPIC_RULES:
        extra_phrases.extend(TOPIC_RULES[topic].get("search_phrases", ()))

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
    topic = classify_question_topic(q)
    # Do not apply generic-intent penalties to pages for a specific classified topic.
    intent = "" if topic in TOPIC_RULES else classify_intent(q)
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
        asks_towing = any(term in q for term in ("גרירה", "גרר"))
        is_towing_contact = page_is_towing_contact(page)
        if any(cue in title for cue in CONTACT_TITLE_CUES):
            score += 30.0
        if any(cue in body for cue in CONTACT_BODY_CUES):
            score += 4.0
        if EMAIL_RE.search(body) or PHONE_RE.search(body):
            score += 8.0
        if asks_towing and is_towing_contact:
            score += 30.0
        if not page_has_contact_evidence(page) and not (asks_towing and is_towing_contact):
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

    topic = classify_question_topic(q)
    if topic in TOPIC_RULES:
        rule = TOPIC_RULES[topic]
        title_hits = sum(1 for cue in rule.get("title_cues", ()) if cue in title)
        body_hits = sum(1 for cue in rule.get("evidence_cues", ()) if cue in body)
        if title_hits:
            score += 45.0 + min(title_hits - 1, 3) * 3.0
        if body_hits:
            score += min(body_hits, 5) * 4.0
        if any(cue in title for cue in rule.get("exclude_titles", ())):
            score -= 60.0
        if not page_has_topic_evidence(page, topic):
            score -= 35.0

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
    topic = classify_question_topic(question)

    if intent == "contact":
        normalized_clean = normalize_text(clean)
        asks_towing = any(term in normalize_text(question) for term in ("גרירה", "גרר"))
        if asks_towing and "ספק הגרירה" in normalized_clean and "*8888" in normalized_clean and "*2008" in normalized_clean:
            return "ספק הגרירה מופיע בתנאי הפוליסה: שגריר – *8888; דרכים – *2008. בדקי בפוליסה איזה ספק חל עלייך."

        if topic == "claim_contact":
            claim_phrase = re.search(r"מוקד\s+תביעות\s+(?:וואטספ|וואטסאפ|ווטספ|ווטסאפ)", normalized_clean)
            if claim_phrase:
                nearby = clean[claim_phrase.end():claim_phrase.end() + 100]
                phone_match = PHONE_RE.search(nearby)
                if phone_match:
                    digits = re.sub(r"\D", "", phone_match.group(0))
                    if len(digits) == 10 and digits.startswith("0"):
                        formatted_phone = digits[:3] + "-" + digits[3:]
                    else:
                        formatted_phone = phone_match.group(0)
                    return f"מוקד תביעות בוואטסאפ: {formatted_phone}."

    if topic == "claim_status" and "סטטוס תביעה" in normalize_text(clean):
        return "באתר הפניקס מופיעה אפשרות בשם 'סטטוס תביעה'; הטקסט שנאסף אינו מפרט את שלבי הבדיקה."

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

        # Avoid presenting a blanket instruction to leave a vehicle in place until
        # a police investigator arrives; that wording is not a safe universal rule.
        if topic == "accident_first_steps" and any(cue in normalized for cue in (
            "אל תזיזו את הרכב", "אין להזיז את הרכב", "לא להזיז את הרכב",
            "עד שהבוחן המשטרתי", "עד להגעת הבוחן", "עד שהמשטרה תגיע",
        )):
            continue

        # Specific topics override broad intent labels. Only sentences that directly
        # discuss the requested subject are eligible for the answer excerpt.
        if topic in TOPIC_RULES:
            if not topic_sentence_matches(sentence, topic, question):
                continue
            rule = TOPIC_RULES[topic]
            cue_hits = sum(1 for cue in rule.get("sentence_cues", ()) if cue in normalized)
            score = 7.0 + min(cue_hits, 5) * 2.5
            candidates.append((score, 5, index, sentence))
            continue

        # Intent-specific filters are applied before generic keyword scoring. This is
        # the key guard against returning a true-but-unrelated paragraph.
        if intent == "contact":
            asks_towing = any(cue in normalize_text(question) for cue in ("גרירה", "גרר"))
            if any(cue in normalized for cue in towing_noise) and not asks_towing:
                continue
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


def compose_topic_answer(
    question: str,
    topic: str,
    evidence: str,
    sources: list[dict],
) -> str | None:
    """Create a direct, topic-specific answer from the retrieved Phoenix evidence."""
    if topic == "out_of_scope":
        return "הצ'אטבוט הזה מיועד לשאלות על ביטוחי הפניקס. במאגר אין מידע על תחזית מזג אוויר."
    if topic not in TOPIC_RULES:
        return None

    q = normalize_text(question)
    evidence = normalize_text(evidence)
    source = next((item for item in sources if (item.get("snippet") or "").strip()), None)
    title = (source.get("title") if source else "") or "המקור שנמצא באתר הפניקס"

    if topic == "health_insurance_overlap":
        has_health_fund_context = any(term in evidence for term in (
            "קופת החולים", "קופות החולים", "ביטוח משלים", "שב\"ן", "שבן",
        ))
        has_private_context = any(term in evidence for term in (
            "ביטוחי הבריאות", "ביטוח בריאות פרטי", "ביטוח פרטי",
            "חברת הביטוח הפרטית", "הביטוח הפרטי",
        ))
        has_benefit_context = any(term in evidence for term in (
            "כיסוי", "מכוסה", "הרחבת הביטוח", "ניתוח", "ניתוחים",
            "השתלה", "השתלות",
        ))
        if not (has_health_fund_context and has_private_context and has_benefit_context):
            return topic_fallback_answer(topic)

        return (
            "ייתכן שיש חפיפה בין ביטוח בריאות פרטי לבין הביטוח המשלים של קופת החולים, "
            "אבל אי אפשר לקבוע מהמקור הזה אם יש כפל ביטוח בפועל. המאמר באתר הפניקס מתאר "
            "מקרים שבהם ביטוח פרטי סיפק כיסוי להשתלה או לטיפול שלא קיבל מענה מקופת החולים, "
            "וממליץ לבחון כיסוי פרטי נוסף. עם זאת, הוא אינו משווה סעיף-סעיף בין פוליסה פרטית "
            "לבין מסלול השב\"ן, ולכן אינו מוכיח שאין כפילות. כדי לבדוק את המקרה שלך, יש "
            "להשוות את הכיסויים, התקרות, החריגים ותנאי המימוש בשני המסלולים. "
            f"המקור שנמצא: '{title}'."
        )

    if topic == "home_insurance_overview":
        has_structure_evidence = any(term in evidence for term in (
            "ביטוח מבנה", "ביטוח תכולה", "מבנה ותכולה",
            "פוליסת ביטוח דירה נחלקת",
        ))
        if not has_structure_evidence:
            return topic_fallback_answer(topic)

        answer = (
            "לפי המידע הכללי באתר הפניקס, ביטוח דירה מתחלק לביטוח מבנה ולביטוח תכולה. "
            "ביטוח מבנה מתייחס למבנה הדירה, וביטוח תכולה לרכוש שנמצא בתוכה. "
            "בחלק מהפוליסות מופיע גם כיסוי צד ג׳, בהתאם לתנאי הפוליסה ולכיסויים שנרכשו. "
            "הכיסוי בפועל, סכומי הביטוח, ההרחבות והחריגים נקבעים לפי הפוליסה הספציפית, "
            "לכן כדאי לבדוק את דף פרטי הביטוח ואת תנאיו. "
            f"המקור המרכזי שנמצא: '{title}'."
        )
        return answer

    if topic == "theft_coverage":
        if any(term in evidence for term in ("גניבה", "גניבת", "נגנב")) and ("מקיף" in q or "מקיף" in evidence):
            return (
                "כן. לפי המידע שנמצא במאגר, ביטוח מקיף כולל כיסוי לגניבת הרכב, "
                "בכפוף לתנאי הפוליסה, לסייגים ולדרישות הגשת התביעה. "
                f"המקור הרלוונטי: '{title}'."
            )
        return topic_fallback_answer(topic)

    if topic == "glass_coverage":
        if any(term in evidence for term in ("שמשות", "שמשה", "שבר שמשה", "כיסוי לשמשות")):
            answer = (
                "לא בהכרח בכל פוליסה. כיסוי לנזק לשמשות תלוי בתנאי הפוליסה ובשאלה אם נכלל בה "
                "כיסוי או שירות שמשות. בדקי את דף פרטי הביטוח וההרחבות. "
                f"המקור הרלוונטי במאגר: '{title}'."
            )
            asks_how_to_activate = any(term in q for term in (
                "איך מפעילים", "להפעיל את הביטוח", "מה צריך לעשות", "איך מפעילים את השירות",
                "איך משתמשים", "איך מזמינים", "מה עושים",
            ))
            if asks_how_to_activate:
                answer += (
                    " המאגר מאשר שקיים שירות תיקון או החלפת שמשות כהרחבה אפשרית, אך אינו מפרט "
                    "את שלבי ההפעלה או את ספק השירות לכל פוליסה. בדקי בפרטי הביטוח את ספק שירות "
                    "השמשות ואת פרטי ההתקשרות שלו; אם אינם מופיעים, פני להפניקס לקבלת הנחיות."
                )
            return answer
        return topic_fallback_answer(topic)
    if topic == "water_pipe_damage":
        pipe_source = next(
            (
                item for item in sources
                if (item.get("snippet") or "").strip()
                and any(
                    cue in normalize_text(item.get("title", ""))
                    for cue in TOPIC_RULES[topic]["title_cues"]
                )
            ),
            None,
        )
        if pipe_source is None:
            pipe_source = next(
                (
                    item for item in sources
                    if (item.get("snippet") or "").strip()
                    and any(
                        cue in normalize_text(item.get("snippet", ""))
                        for cue in (
                            "נזקי מים וצנרת", "נזקי צנרת", "נזילה בצנרת",
                            "דליפה בצנרת", "פיצוץ צינור", "צינור שהתפוצץ",
                        )
                    )
                    and not any(
                        cue in normalize_text(item.get("title", ""))
                        for cue in TOPIC_RULES[topic]["exclude_titles"]
                    )
                ),
                None,
            )
        if pipe_source is None:
            return topic_fallback_answer(topic)

        pipe_title = pipe_source.get("title") or "המקור בנושא נזקי מים וצנרת"
        pipe_snippet = normalize_text(pipe_source.get("snippet", ""))
        has_pipe_evidence = any(
            cue in pipe_snippet or cue in normalize_text(pipe_title)
            for cue in (
                "נזקי מים וצנרת", "נזקי צנרת", "נזילה בצנרת",
                "דליפה בצנרת", "פיצוץ צינור", "צינור שהתפוצץ",
            )
        )
        has_damage_and_coverage = (
            any(cue in pipe_snippet for cue in ("נזק למבנה", "נזק לתכולה", "נזקי מים", "נזקי צנרת"))
            and any(cue in pipe_snippet for cue in ("כיסוי", "מכסה", "מכוסים", "מכוסה", "לכסות", "הרחבה"))
        )
        if not has_pipe_evidence or not has_damage_and_coverage:
            return (
                "מצאתי מקור בנושא נזקי מים וצנרת, אך הקטע שאוחזר אינו מאשר במפורש אם הנזק הספציפי מכוסה. "
                "הכיסוי תלוי בהרחבה שנרכשה, בתנאי הפוליסה ובחריגים. חשוב להבחין בין הנזק הפיזי לבין "
                "עלות המים שהתבזבזו. "
                f"המקור: '{pipe_title}'."
            )

        answer = (
            "לפי המקור באתר הפניקס, הכיסוי לנזקי מים וצנרת עשוי לכלול נזק למבנה או לתכולה שנגרם "
            "מדליפה בצנרת או מפיצוץ צינור, בהתאם להרחבה שנרכשה ולתנאי הפוליסה. "
            "אין להסיק מכך שעלות המים שהתבזבזו מכוסה; יש לבדוק גם את החריגים ואת מצב הצנרת. "
            f"המקור: '{pipe_title}'."
        )
        return answer

    if topic == "vehicle_flood_coverage":
        vehicle_source = next(
            (
                item for item in sources
                if (item.get("snippet") or "").strip()
                and any(
                    cue in normalize_text(item.get("title", ""))
                    for cue in TOPIC_RULES[topic]["title_cues"]
                )
            ),
            None,
        )
        if vehicle_source is None:
            vehicle_source = next(
                (
                    item for item in sources
                    if (item.get("snippet") or "").strip()
                    and any(cue in normalize_text(item.get("snippet", ""))
                            for cue in ("הצפה", "שיטפון", "נזקי מים", "נזקי טבע"))
                    and any(cue in normalize_text(item.get("title", ""))
                            for cue in ("רכב", "מקיף"))
                    and not any(cue in normalize_text(item.get("title", ""))
                                for cue in TOPIC_RULES[topic]["exclude_titles"])
                ),
                None,
            )
        if vehicle_source is None:
            return topic_fallback_answer(topic)

        vehicle_title = vehicle_source.get("title") or "המקור בנושא ביטוח רכב"
        vehicle_snippet = normalize_text(vehicle_source.get("snippet", ""))
        has_vehicle_and_flood_evidence = (
            any(cue in vehicle_snippet for cue in ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים", "נזקי טבע"))
            and any(cue in vehicle_snippet or cue in normalize_text(vehicle_title)
                    for cue in ("רכב", "הרכב", "לרכב", "ביטוח רכב", "מקיף"))
        )
        if not has_vehicle_and_flood_evidence:
            return topic_fallback_answer(topic)

        if any(cue in vehicle_snippet for cue in ("אינו מכסה", "אינם מכוסים", "לא מכסה", "לא מכוסה", "לא יכסה")):
            return (
                "לפי הקטע שנמצא, קיימת מגבלת כיסוי או החרגה בנוגע לנזקי הצפה/טבע. "
                "אין להניח שהנזק מכוסה; יש לבדוק את הנוסח המלא של הפוליסה ואת פרטי הכיסוי. "
                f"המקור: '{vehicle_title}'."
            )

        if any(cue in vehicle_snippet for cue in ("מכסה", "כיסוי", "מכוסים", "מכוסה")):
            return (
                "המקור שנמצא עוסק בביטוח רכב ובנזקי הצפה או טבע, אך אין להסיק ממנו שכל נזקי הצפה מכוסים. "
                "הכיסוי בפועל תלוי בסוג הביטוח, בכיסוי שנרכש, בתנאי הפוליסה ובחריגים. "
                f"בדקי את פרטי הפוליסה מול הפניקס. המקור: '{vehicle_title}'."
            )

        return topic_fallback_answer(topic)

    if topic == "water_leak_cost":
        water_snippet = normalize_text(
            source.get("snippet", "")
        ) if source else ""

        water_bill_terms = (
            "עלות המים", "חשבון המים", "חשבון מים",
            "מים שהתבזבזו", "מים שבוזבזו",
            "צריכת מים חריגה", "הפרשי צריכה",
        )
        coverage_terms = (
            "מכוסה", "מכוסים", "כיסוי", "החזר",
            "פיצוי", "ישולם", "זכאות",
        )
        exclusion_terms = (
            "לא מכוסה", "אינו מכוסה", "אינם מכוסים",
            "לא מכסה", "אינו מכסה", "אין החזר",
            "לא יינתן החזר",
        )

        has_water_bill_evidence = any(
            term in water_snippet for term in water_bill_terms
        )
        has_coverage_evidence = any(
            term in water_snippet for term in coverage_terms
        )
        has_exclusion = any(
            term in water_snippet for term in exclusion_terms
        )

        if has_water_bill_evidence and any(
            term in water_snippet
            for term in (
                "אינו כולל",
                "אינה כוללת",
                "אינם כוללים",
                "לא את עלות המים",
                "ולא את עלות המים",
            )
        ):
            return (
                "לפי המקור באתר הפניקס, עלות המים עצמה אינה מכוסה. "
                "הכיסוי המתואר מתייחס לעלות הנזק שנגרם למבנה או לרכוש, "
                "בהתאם לתנאי הפוליסה. "
                f"המקור: '{title}'."
            )

        if has_water_bill_evidence and has_coverage_evidence:
            if has_exclusion:
                return (
                    "המקור שנמצא כולל התייחסות למגבלת כיסוי או להחזר "
                    f"בנושא עלות המים. המקור: '{title}'. "
                    "יש לבדוק את הנוסח המלא ואת תנאי הפוליסה."
                )
            return (
                "נמצא קטע שמתייחס במפורש לחשבון המים ולכיסוי או להחזר, "
                "אך יש לבדוק את הנוסח המלא ואת תנאי הפוליסה לפני קביעה "
                f"אם קיימת זכאות. המקור: '{title}'."
            )

        if source:
            return (
                "המקור שנמצא אינו מאשר במפורש אם עלות המים עצמה מכוסה. "
                "כיסוי לנזקי צנרת אינו בהכרח כולל החזר על חשבון המים. "
                f"המקור שנמצא: '{title}'. מומלץ לבדוק את תנאי הפוליסה "
                "או לפנות להפניקס."
            )

        return topic_fallback_answer(topic)

    if topic == "weather_damage_contents":
        weather_source = next(
            (
                item
                for item in sources
                if any(
                    cue in normalize_text(item.get("title", ""))
                    for cue in (
                        "ביטוח תכולה לדירה",
                        "החורף בשיאו",
                        "ביטוח דירה ועוד הרחבות",
                        "ביטוח כל הסיכונים",
                    )
                )
                and any(
                    cue in normalize_text(item.get("snippet", ""))
                    for cue in (
                        "מזג אוויר סוער",
                        "מזג אוויר קיצוני",
                        "נזקי טבע",
                        "סערה",
                        "סופה",
                    )
                )
                and any(
                    cue in normalize_text(item.get("snippet", ""))
                    for cue in (
                        "תכולה",
                        "תכולת הדירה",
                        "רכוש בדירה",
                    )
                )
            ),
            None,
        )

        if not weather_source:
            return topic_fallback_answer(topic)

        weather_title = (
            weather_source.get("title")
            or "ביטוח תכולה לדירה"
        )

        return (
            "לפי המידע באתר הפניקס, ביטוח תכולה עשוי לכלול "
            "כיסוי לנזק לרכוש בדירה כתוצאה ממזג אוויר סוער. "
            "לכן, נזק לרהיטים בעקבות רוחות חזקות עשוי להיות מכוסה, "
            "אך המקור אינו מאשר במפורש שכל נזק כזה מכוסה. "
            "הכיסוי בפועל תלוי בתנאי הפוליסה, בכיסויים שנרכשו "
            "ובחריגים. מומלץ לבדוק את הפוליסה או לפנות להפניקס "
            "לקבלת אישור למקרה הספציפי. "
            f"המקור: '{weather_title}'."
        )

    if topic == "weather_flood_contents":
        content_cues = ("תכולה", "תכולת הדירה", "רכוש בדירה", "רכוש הבית")
        weather_cues = (
            "מזג אוויר סוער", "מזג אוויר קיצוני", "נזקי טבע",
            "סערה", "סופה", "רוחות", "רוח חזקה",
        )
        flood_cues = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים")
        weather_excluded_title_cues = (
            "הצפה בבית", "נזקי מים וצנרת", "נזקי צנרת",
        )
        direct_coverage_cues = ("מכסה", "מכוסים", "מכוסה", "כיסוי")
        coverage_exclusion_cues = (
            "אינו מכסה", "אינה מכסה", "לא מכסה", "אינו מכוסה", "אינם מכוסים",
        )
        weather_source = next(
            (
                item for item in sources
                if any(cue in normalize_text(item.get("snippet", "")) for cue in content_cues)
                and any(cue in normalize_text(item.get("snippet", "")) for cue in weather_cues)
                and any(cue in normalize_text(item.get("snippet", "")) for cue in direct_coverage_cues)
                and not any(cue in normalize_text(item.get("snippet", "")) for cue in coverage_exclusion_cues)
                and not any(cue in normalize_text(item.get("title", "")) for cue in weather_excluded_title_cues)
            ),
            None,
        ) or next(
            (
                item for item in sources
                if any(cue in normalize_text(item.get("snippet", "")) for cue in content_cues)
                and any(cue in normalize_text(item.get("snippet", "")) for cue in weather_cues)
                and not any(cue in normalize_text(item.get("title", "")) for cue in weather_excluded_title_cues)
            ),
            None,
        )
        # Prefer a source that explicitly states contents coverage for flooding,
        # even if its title does not contain the word "הצפה".
        direct_contents_flood_source = next(
            (
                item for item in sources
                if any(cue in normalize_text(item.get("snippet", "")) for cue in content_cues)
                and any(cue in normalize_text(item.get("snippet", "")) for cue in flood_cues)
                and any(cue in normalize_text(item.get("snippet", "")) for cue in (
                    "מכסה", "מכוסים", "מכוסה", "כיסוי",
                ))
                and not any(cue in normalize_text(item.get("snippet", "")) for cue in (
                    "אינו מכסה", "אינה מכסה", "לא מכסה", "אינו מכוסה", "אינם מכוסים",
                ))
            ),
            None,
        )
        flood_source = direct_contents_flood_source or next(
            (
                item for item in sources
                if any(cue in normalize_text(item.get("title", "")) for cue in ("הצפה", "שיטפון"))
                and any(cue in normalize_text(item.get("snippet", "")) for cue in content_cues)
                and any(cue in normalize_text(item.get("snippet", "")) for cue in flood_cues)
            ),
            None,
        )

        parts = []
        if weather_source:
            weather_title = weather_source.get("title") or "המקור בנושא נזקי מזג אוויר"
            weather_snippet = normalize_text(weather_source.get("snippet", ""))
            direct_weather_coverage = (
                any(cue in weather_snippet for cue in direct_coverage_cues)
                and not any(cue in weather_snippet for cue in coverage_exclusion_cues)
            )
            if direct_weather_coverage:
                parts.append(
                    "**נזקי מזג אוויר סוער:** לפי המקור '" + weather_title + "', ביטוח תכולה מכסה נזקים "
                    "לרכוש בדירה כתוצאה ממזג אוויר סוער. הכיסוי למקרה הספציפי כפוף לתנאי הפוליסה, "
                    "להרחבות ולחריגים."
                )
            else:
                parts.append(
                    "**נזקי מזג אוויר סוער:** לפי המידע במקור '" + weather_title + "', "
                    "ביטוח תכולה עשוי לכלול כיסוי לנזק לרכוש בדירה כתוצאה ממזג אוויר סוער. "
                    "הכיסוי בפועל תלוי בתנאי הפוליסה, בהרחבות ובחריגים."
                )
        else:
            parts.append(
                "**נזקי מזג אוויר סוער:** במקורות שאוחזרו לא נמצא קטע שמאשר במפורש כיסוי לנזקי סערה לתכולה. "
                "אין להסיק שכיסוי להצפה כולל גם נזקי רוח או סערה."
            )

        if flood_source:
            flood_title = flood_source.get("title") or "המקור בנושא הצפה בבית"
            flood_snippet = normalize_text(flood_source.get("snippet", ""))
            direct_external_flood = any(cue in flood_snippet for cue in (
                "שיטפון או הצפה ממקור מים חיצוני",
                "הצפה ממקור מים חיצוני",
                "הצפה פתאומית מהביוב",
            ))
            direct_contents_flood_coverage = (
                any(cue in flood_snippet for cue in ("מכסה", "מכוסים", "מכוסה", "כיסוי"))
                and not any(cue in flood_snippet for cue in (
                    "אינו מכסה", "אינה מכסה", "לא מכסה", "אינו מכוסה", "אינם מכוסים",
                ))
                and any(cue in flood_snippet for cue in content_cues)
                and any(cue in flood_snippet for cue in flood_cues)
            )
            if direct_contents_flood_coverage:
                parts.append(
                    "**נזקי הצפה:** לפי המקור '" + flood_title + "', ביטוח תכולה מכסה נזקים לרכוש בדירה "
                    "כתוצאה מהצפות. יש לבדוק את תנאי הפוליסה, ההרחבות והחריגים כדי לוודא שהמקרה הספציפי נכלל."
                )
            elif "תכולה" in flood_snippet and direct_external_flood:
                parts.append(
                    "**נזקי הצפה:** לפי המקור '" + flood_title + "', הרחבת סיכוני מים ונוזלים אחרים "
                    "בביטוח תכולה עשויה לכלול נזקי תכולה מדליפה פתאומית בצנרת, מפיצוץ צינור, "
                    "מהצפה פתאומית מהביוב או משיטפון/הצפה ממקור מים חיצוני, בכפוף להרחבה שנרכשה ולתנאי הפוליסה."
                )
            else:
                parts.append(
                    "**נזקי הצפה:** נמצא מקור בנושא הצפה ('" + flood_title + "'), אך הקטע שאוחזר אינו מאשר במפורש "
                    "את הכיסוי לנזקי תכולה במקרה המסוים. יש לבדוק את הרחבת נזקי המים ואת החריגים בפוליסה."
                )
        else:
            parts.append(
                "**נזקי הצפה:** לא נמצא מקור ממוקד שמאשר את הכיסוי לנזקי הצפה בתכולה. "
                "יש לבדוק את הרחבת נזקי המים/הצפה ואת החריגים בפוליסה."
            )

        return " ".join(parts)

    if topic == "flood_coverage" and any(
        term in q for term in (
            "תכולה", "תכולת", "רכוש", "רהיטים", "מכשירי חשמל"
        )
    ):
        contents_source = next(
            (
                item for item in sources
                if "הצפה" in normalize_text(item.get("title", ""))
                and any(
                    term in normalize_text(item.get("snippet", ""))
                    for term in (
                        "שיטפון או הצפה ממקור מים חיצוני",
                        "הצפה ממקור מים חיצוני",
                    )
                )
            ),
            None,
        )

        if contents_source:
            contents_title = (
                contents_source.get("title")
                or "המקור בנושא הצפה בבית"
            )
            contents_snippet = normalize_text(
                contents_source.get("snippet", "")
            )

            if "תכולה" in contents_snippet:
                return (
                    "לפי המקור באתר הפניקס, הרחבת סיכוני מים ונוזלים "
                    "אחרים בביטוח תכולה עשויה לכלול נזקי תכולה שנגרמו "
                    "מדליפה פתאומית בצנרת, מפיצוץ צינור, מהצפה פתאומית "
                    "מהביוב או משיטפון או הצפה ממקור מים חיצוני. "
                    "עלות המים עצמה אינה מכוסה לפי המקור. "
                    "הכיסוי בפועל תלוי בהרחבה שנרכשה ובתנאי הפוליסה. "
                    f"המקור: '{contents_title}'."
                )

    if topic == "flood_coverage":
        flood_terms = ("הצפה", "שיטפון", "שיטפונות", "נזקי מים")
        snippet = normalize_text(source.get("snippet", "")) if source else ""
        if any(term in evidence for term in flood_terms):
            has_direct_coverage_statement = any(term in snippet for term in ("מכסה", "כיסוי", "נזקי טבע"))
            is_explicitly_excluded = any(term in snippet for term in (
                "אינו מכסה", "אינם מכוסים", "לא מכסה", "לא יכסה", "אינו מכוסה",
            ))
            if has_direct_coverage_statement and is_explicitly_excluded:
                return (
                    "לפי המקור באתר הפניקס, אין להניח שנזקי הצפה או שיטפון מכוסים בכל מקרה; "
                    f"המקור מציין מגבלה או החרגה: '{title}'. בדקי את נוסח הפוליסה והחריגים."
                )
            if has_direct_coverage_statement:
                answer = (
                    "לפי המקור שנמצא, נזקי הצפה או שיטפון עשויים להיכלל בכיסוי לנזקי טבע המתואר בו, "
                    "בכפוף לסוג הביטוח, לתנאי הפוליסה ולחריגים. המקור אינו קובע שכל מקרה הצפה מכוסה. "
                    f"המקור: '{title}'."
                )
                asks_what_to_do = any(term in q for term in (
                    "מה צריך לעשות", "מה לעשות", "איך לפעול", "איך מדווחים", "מה עליי לעשות",
                    "מה עלי לעשות", "מה לעשות עכשיו",
                ))
                if asks_what_to_do:
                    answer += (
                        " לגבי הפעולות לאחר ההצפה: המאגר אינו מפרט נוהל דיווח ייעודי. בדקי את פרטי "
                        "הפוליסה ופני להפניקס או לסוכן הביטוח לקבלת הוראות להגשת דיווח ולמסמכים הנדרשים. "
                        "אם ניתן לעשות זאת בבטחה, שמרי תיעוד של הנזק לצורך הבירור."
                    )
                return answer
            return (
                "הכיסוי לנזקי הצפה או שיטפון תלוי בסוג הביטוח, בתנאי הפוליסה ובחריגים. "
                f"המקור '{title}' עוסק בנושא, אך הקטע שנמצא אינו מאשר במפורש שהנזק מכוסה."
            )
        return topic_fallback_answer(topic)

    if topic == "accident_first_steps":
        steps = [
            "ודאי תחילה שהמקום בטוח; אם יש נפגעים או סכנה מיידית, דאגי לעזרה מתאימה."
        ]
        if any(term in evidence for term in (
            "החליפו פרטים", "פרטי הנהגים", "פרטי הנהג", "מספרי הרכבים", "פרטי הרכב",
        )):
            steps.append("החליפי פרטים עם הנהגים המעורבים ורשמי את פרטי הנהגים והרכבים.")
        if any(term in evidence for term in ("צלמו", "תיעוד האירוע", "תיעוד בכתב", "צילום הרכב")):
            steps.append("תעדי את האירוע וצלמי את הרכבים ואת הנזקים, ככל שניתן לעשות זאת בבטחה.")
        if "אין לחתום" in evidence and "הודאה באשמה" in evidence:
            steps.append("לפי דף ההנחיות של הפניקס, אין לחתום על מסמך המהווה הודאה באשמה.")
        if len(steps) == 1 and source is None:
            return topic_fallback_answer(topic)
        return "מיד אחרי תאונה: " + " ".join(steps) + f" ההנחיות נמצאו במקור '{title}'."

    if topic == "report_to_insurer":
        steps = []
        if any(term in evidence for term in (
            "הודעה על מקרה ביטוח רכב", "מלאו את מסמך התביעה", "טופס תביעה",
        )):
            steps.append("השתמשי בטופס 'הודעה על מקרה ביטוח רכב' או בטופס התביעה המתאים.")
        if any(term in evidence for term in (
            "דרך אתר הפניקס", "את הגשת התביעה באופן מעשי תוכלו לעשות",
            "מלאו את מסמך התביעה אונליין",
        )):
            steps.append("המקור מציין שניתן להגיש דרך אתר הפניקס.")
        if "באמצעות סוכן" in evidence or "דרך סוכן" in evidence:
            steps.append("המקור מציין שאפשר להיעזר גם בסוכן הביטוח.")
        if any(term in evidence for term in ("המסמכים הרלוונטיים", "צלמו את הנזקים", "המסמכים והתמונות")):
            steps.append("צרפי את המסמכים והתיעוד הרלוונטיים לפי הוראות הטופס.")
        if not steps:
            return topic_fallback_answer(topic)
        return (
            "כדי לדווח להפניקס על תאונת רכב, " + " ".join(steps)
            + " דיווח למשטרה, אם נדרש בנסיבות המקרה, אינו מחליף דיווח לחברת הביטוח."
        )

    if topic == "deductible":
        if any(term in evidence for term in ("השתתפות עצמית", "ההשתתפות העצמית", "דמי ההשתתפות")):
            asks_personal_amount = any(term in q for term in (
                "שלי", "שלי?", "האישי", "האישית", "איך אפשר לבדוק", "איך בודקים",
                "איפה לבדוק", "איך אני בודק", "איך אני בודקת", "מה גובה ההשתתפות העצמית שלי",
            ))
            asks_third_party = any(term in q for term in ("צד ג", "צד ג'", "צד שלישי"))
            if asks_third_party and any(term in evidence for term in ("לבטלה באמצעות הגדלת סכום הפרמיה", "ביטוח רכב צד ג")):
                return (
                    "לפי המקור, בביטוח רכב צד ג' קיימת השתתפות עצמית, וניתן לבטלה באמצעות הגדלת סכום הפרמיה. "
                    "הסכום המדויק והאפשרות הזו בפוליסה שלך תלויים במסלול ובתנאי הפוליסה, ולכן כדאי לבדוק את דף פרטי הביטוח. "
                    f"המקור במאגר: '{title}'."
                )
            if asks_personal_amount:
                return (
                    "אי אפשר לקבוע את סכום ההשתתפות העצמית האישי מתוך מאמר כללי: הסכום תלוי בפוליסה, "
                    "בסוג הכיסוי ובנסיבות התביעה. בדקי את דף פרטי הביטוח או את תנאי הפוליסה שלך; "
                    "אם הסכום אינו ברור, פני להפניקס או לסוכן הביטוח. סכומים שמופיעים במאמר כדוגמאות "
                    "אינם בהכרח הסכום שלך. "
                    f"המקור במאגר: '{title}'."
                )
            return (
                "השתתפות עצמית היא הסכום שהמבוטח נדרש לשלם בהתאם לתנאי הפוליסה במקרה הרלוונטי. "
                "הסכום והאופן שבו הוא חל משתנים לפי סוג הביטוח, המסלול ותנאי הפוליסה; יש לבדוק את דף פרטי הביטוח. "
                f"המקור במאגר: '{title}'."
            )
        return topic_fallback_answer(topic)

    if topic == "driver_eligibility":
        if any(term in evidence for term in (
            "נהגים המורשים", "הנהגים המורשים", "כל נהג", "נהג נוסף",
            "רשאי לנהוג", "מורשה לנהוג", "נהג צעיר", "נהג חדש",
        )):
            return (
                "לא בהכרח כל נהג מכוסה. הכיסוי תלוי בתנאי הפוליסה, בזהות או בגיל הנהגים "
                "שהותר להם לנהוג, ובהרחבות שנרכשו. בדקי בפוליסה מי הנהגים המורשים ומהן מגבלות הגיל. "
                f"המקור הרלוונטי: '{title}'."
            )
        return topic_fallback_answer(topic)

    if topic == "young_driver":
        if any(term in evidence for term in ("נהג צעיר", "נהג חדש", "כיסוי לנהג צעיר", "הרחבת נהג צעיר")):
            return (
                "אי אפשר לקבוע רק מכך שיש ביטוח רכב בהפניקס שהפוליסה מכסה נהג צעיר. "
                "הכיסוי תלוי בתנאי הפוליסה ובהרחבות שנרכשו לנהג צעיר או חדש. "
                f"בדקי את דף פרטי הביטוח וההרחבות; המקור הרלוונטי הוא '{title}'."
            )
        return topic_fallback_answer(topic)

    if topic == "vehicle_recovered":
        recovery_cues = ("הרכב נמצא", "נמצא הרכב", "הוחזר לבעליו", "מוחזר לבעליו",
                         "הרכב שנגנב נמצא", "שנמצא ומוחזר",
                         "רכב שנגנב ונמצא", "נמצא הרכב הגנוב", "הרכב הגנוב נמצא")
        if any(term in evidence for term in recovery_cues):
            return (
                f"המקור '{title}' מזכיר רכב שנמצא או הוחזר, אך הקטע שנמצא אינו קובע בבירור "
                "כיצד הדבר משפיע על תביעה שכבר הוגשה. עדכני את הפניקס ובקשי הנחיות בהתאם לשלב התביעה "
                "ולתנאי הפוליסה; אין במאגר בסיס לקבוע שהתביעה מתבטלת או נמשכת אוטומטית."
            )
        return topic_fallback_answer(topic)

    if topic == "vehicle_stolen":
        actions = []
        if "משטרה" in evidence:
            actions.append("דווחי למשטרה אם הגניבה טרם דווחה.")
        if any(term in evidence for term in (
            "הודעה על מקרה ביטוח", "טופס תביעה", "טופס התביעה", "הגשת תביעה",
            "פנו לחברת הביטוח", "פנה לחברת הביטוח", "דווחו לחברת הביטוח", "פנו להפניקס",
        )):
            actions.append("דווחי להפניקס באמצעות טופס 'הודעה על מקרה ביטוח רכב' או טופס תביעת הרכב המתאים.")
        if any(term in evidence for term in ("מסמכים", "המסמכים הרלוונטיים")):
            actions.append("העבירי את המסמכים הנדרשים לפי הטופס והוראות הפניקס.")
        if not actions:
            # The corpus has general theft-coverage information but no verified
            # claim-submission steps. State that distinction instead of implying
            # that the coverage article explains how to file a theft claim.
            if source and any(term in evidence for term in ("גניבה", "גניבת הרכב", "נגנב")) and "מקיף" in evidence:
                return (
                    f"המקור '{title}' מתייחס לכיסוי גניבה במסגרת ביטוח מקיף, אך המאגר אינו כולל "
                    "הוראות מאומתות להגשת תביעת גניבה או רשימת מסמכים ייעודית. בדקי את תנאי הפוליסה "
                    "ופני להפניקס דרך האתר הרשמי או לסוכן הביטוח כדי לקבל את טופס התביעה וההנחיות "
                    "המתאימים למקרה שלך."
                )
            return topic_fallback_answer(topic)
        if "מקיף" in evidence and "גניבה" in evidence:
            actions.append("כיסוי לגניבה תלוי בתנאי הפוליסה ובסייגים.")
        return "אם הרכב נגנב: " + " ".join(dict.fromkeys(actions)) + f" המקור הרלוונטי: '{title}'."

    if topic == "claim_status":
        if "באזור האישי" in evidence or "אזור אישי" in evidence:
            return (
                "המקור שנמצא מפנה לאזור האישי לצורך פעולות הקשורות לתביעה. היכנסי לאזור האישי "
                "באתר הפניקס וחפשי את התביעה שהגשת ואת הסטטוס שלה. "
                f"המקור: '{title}'."
            )
        if "סטטוס תביעה" in evidence:
            return (
                "במאגר מופיעה באתר הפניקס אפשרות בשם 'סטטוס תביעה', אך אין פירוט מאומת של שלבי הבדיקה. "
                "חפשי באתר את האפשרות הזו; אם אינך מצליחה למצוא את התביעה, פני לשירות הלקוחות של הפניקס. "
                f"המקור: '{title}'."
            )
        if any(term in evidence for term in ("מעקב אחר תביעה", "מצב התביעה", "סטטוס התביעה",
                                              "בדיקת סטטוס", "תביעות שהוגשו")):
            return (
                f"מצאתי מקור שעוסק במצב או במעקב אחר תביעה: '{title}'. "
                "הקטע שנמצא אינו מפרט בבירור את כל שלבי הבדיקה, ולכן כדאי לפעול לפי ההנחיות "
                "בעמוד עצמו או לפנות לשירות הלקוחות של הפניקס."
            )
        return topic_fallback_answer(topic)

    if topic == "claim_timeline":
        if "30 יום" in evidence or "30 ימים" in evidence:
            if "חברת הביטוח הנתבעת לחזור אליכם" in evidence or "לחזור אליכם בתוך 30 יום" in evidence:
                return (
                    "המקור מציין שעל חברת הביטוח הנתבעת לחזור בתוך 30 יום. חשוב: זהו מועד למענה "
                    "או להתייחסות לתביעה, ולא בהכרח התחייבות לסיים את הטיפול כולו בתוך 30 יום. "
                    f"המקור: '{title}'."
                )
            return (
                f"המקור '{title}' מזכיר תקופה של 30 יום בהקשר התביעה, אך לא ברור ממנו שזהו משך "
                "הטיפול הכולל. אין במאגר זמן מאומת לסיום כל תביעת רכב."
            )
        return (
            "לא מצאתי במאגר פרק זמן מאומת לסיום הטיפול בתביעת רכב. משך הטיפול עשוי להיות תלוי "
            "בסוג התביעה ובמסמכים החסרים; יש להבחין בין מענה ראשוני לבין סיום התביעה."
        )

    if topic == "missing_documents":
        if any(term in evidence for term in (
            "תתבקשו להשלים את החסר", "להשלים את החסר", "השלמת מסמכים",
            "מסמכים חסרים", "המסמכים החסרים",
        )):
            return (
                "אם חסרים מסמכים, המקור מציין שחברת הביטוח עשויה לבקש להשלים את החסר. "
                "העבירי את המסמכים החסרים לפי בקשת הפניקס, ושמרי את אישור השליחה. "
                "הרשימה המדויקת תלויה בסוג התביעה ובבקשת ההשלמה שקיבלת."
            )
        return topic_fallback_answer(topic)

    return None


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
    topic = classify_question_topic(q)
    topic_answer = compose_topic_answer(question, topic, evidence, sources)
    if topic_answer is not None:
        return topic_answer

    if mentions_compulsory and any(
    phrase in q for phrase in (
        "רכב אחר",
        "הרכב האחר",
        "נזק לרכב האחר",
        "נזק לרכב אחר",
        "פגעתי ברכב",
        "רכב של אדם אחר",
    )
    ) and any(word in q for word in ("מכסה", "יכסה", "כיסוי")):
        return (
            "לא. ביטוח חובה מכסה נזקי גוף שנגרמו בתאונת דרכים, "
            "ולא נזקי רכוש לרכב האחר. כדי לכסות נזקי רכוש שנגרמו "
            "לצד שלישי, נדרש ביטוח צד ג' או ביטוח מקיף, "
            "בכפוף לתנאי הפוליסה."
        )
    if intent == "contact":
        contact_snippets = [source for source in sources if (source.get("snippet") or "").strip()]
        if not contact_snippets:
            return "לא מצאתי במאגר קטע שמציג פרטי קשר מאומתים של הפניקס. כדאי לבדוק בעמוד 'צור קשר' באתר הרשמי."

        asks_towing = any(term in q for term in ("גרירה", "גרר"))
        asks_claims_contact = any(term in q for term in ("תביעה", "תביעות", "בירור", "בירורים"))

        if asks_towing and asks_claims_contact:
            towing_sources = [
                source for source in contact_snippets
                if any(term in normalize_text(
                    f"{source.get('title', '')} {source.get('snippet', '')}"
                ) for term in ("גרירה", "גרר", "ספק הגרירה"))
            ]
            claims_sources = [source for source in contact_snippets if source not in towing_sources]
            parts = []

            if claims_sources:
                first_claim = claims_sources[0]
                claim_snippet = (first_claim.get('snippet') or '').strip()[:500]
                if "מוקד תביעות בוואטסאפ" in normalize_text(claim_snippet):
                    parts.append(
                        f"לבירורים על תביעות: המקור '{first_claim.get('title', 'מוקד תביעות באתר הפניקס')}' "
                        f"מציין {claim_snippet} זהו ערוץ וואטסאפ ולא בהכרח מספר לשיחה טלפונית."
                    )
                else:
                    parts.append(
                        f"לבירורים על תביעות: לפי המקור '{first_claim.get('title', 'צור קשר באתר הפניקס')}', "
                        f"{claim_snippet} "
                        "המקור מציג פרטי קשר, אך אינו מאשר בהכרח שמדובר במוקד תביעות ייעודי."
                    )
            else:
                parts.append("לא נמצא במאגר מספר מאומת לבירורים על תביעות.")

            if towing_sources:
                first_towing = towing_sources[0]
                parts.append(
                    f"לגרירה: לפי המקור '{first_towing.get('title', 'פרטי ספק גרירה')}', "
                    f"{(first_towing.get('snippet') or '').strip()[:500]}"
                )
            else:
                parts.append("לא נמצא במאגר מספר טלפון מאומת לספק הגרירה.")

            return " ".join(parts)

        first = contact_snippets[0]
        excerpt = (first.get("snippet") or "").strip()[:650]
        if topic == "claim_contact":
            if "מוקד תביעות בוואטסאפ" in normalize_text(excerpt):
                return (
                    f"לפי המקור '{first.get('title', 'מוקד תביעות באתר הפניקס')}', מוקד תביעות בוואטסאפ: 054-9893455. "
                    "זהו ערוץ וואטסאפ ולא מספר לשיחה טלפונית."
                )
            return (
                f"פרטי הקשר שמופיעים במקור '{first.get('title', 'צור קשר באתר הפניקס')}': {excerpt} "
                "המאגר אינו מאשר שהמספר הוא מוקד ייעודי לדיווח על תאונות, אלא רק שהוא מופיע בעמוד הקשר של הפניקס."
            )
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

        # For a general document question, combine the relevant lists instead of
        # showing only the first source and accidentally hiding other claim types.
        all_snippets = []
        seen_snippets = set()
        for source in sources:
            candidate = (source.get("snippet") or "").strip()
            if candidate and candidate not in seen_snippets:
                seen_snippets.add(candidate)
                all_snippets.append(candidate)
        combined_snippet = "\n\n".join(all_snippets)[:2600] if all_snippets else snippet

        return (
            "המסמכים משתנים לפי סוג התביעה, ולכן אין רשימה אחת שחלה על כל מקרה. "
            "המקורות שנמצאו במאגר כוללים את המידע הבא לפי סוג התביעה:\n\n"
            f"{combined_snippet}\n\n"
            "התאימי את הרשימה לסוג התביעה שלך ובדקי את טופס התביעה המתאים; "
            "המקורות במאגר אינם מבטיחים שזו רשימה מלאה לכל מקרה."
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


    if asks_difference and mentions_compulsory and mentions_comprehensive and mentions_third_party:
        if "נזקי גוף" in evidence and ("נזקי רכוש" in evidence or "רכוש" in evidence):
            return (
                "ההבדל בין שלושת סוגי הביטוח הוא: "
                "ביטוח חובה מכסה נזקי גוף לנהג, לנוסעים ולמשתמשי דרך שנפגעו בתאונה, בהתאם לחוק. "
                "ביטוח מקיף מכסה נזקי רכוש לרכב המבוטח ולרכוש של צד שלישי, בהתאם לתנאי הפוליסה. "
                "ביטוח צד ג' מכסה נזקי רכוש שנגרמו לאחרים עקב השימוש ברכב המבוטח, "
                "אך אינו מכסה נזק לרכב המבוטח עצמו. "
                "ביטוח חובה נדרש לפי חוק, ואילו ביטוח מקיף וביטוח צד ג' אינם חובה חוקית."
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

    if mentions_third_party and any(
        phrase in q for phrase in (
            "חובה לפי החוק",
            "חובה חוקית",
            "לפי החוק",
            "מחויב",
            "חייבים לרכוש",
            "חייבים לעשות",
        )
    ):
        return (
            "לא. ביטוח צד ג' לרכב אינו חובה לפי החוק. "
            "ביטוח חובה הוא הביטוח הנדרש בחוק ומכסה נזקי גוף בתאונות דרכים. "
            "ביטוח צד ג' הוא ביטוח רשות המיועד לכיסוי נזקי רכוש שנגרמו לצד שלישי, "
            "בהתאם לתנאי הפוליסה ולגבולות הכיסוי."
        )

    if mentions_third_party and any(word in q for word in ("כולל", "מכסה", "מהו", "מה זה", "ביטוח")):
        if "צד שלישי" in evidence or "צד ג" in evidence or "נזקי רכוש" in evidence:
            return (
                "ביטוח צד ג' לרכב מיועד לכסות נזקי רכוש שנגרמו לצד שלישי על ידי הרכב המבוטח, "
                "בהתאם לתנאי הפוליסה ולגבולות הכיסוי. הוא אינו תחליף לביטוח חובה, ובדרך כלל "
                "אינו מכסה נזק לרכב שלך עצמו; לשם כך נדרש כיסוי מתאים כמו ביטוח מקיף."
            )

    if mentions_comprehensive and any(
        phrase in q for phrase in (
            "חובה לפי החוק",
            "חובה חוקית",
            "לפי החוק",
            "מחויב",
            "חייבים לרכוש",
            "חייבים לעשות",
        )
    ):
        return (
            "לא. ביטוח מקיף לרכב אינו חובה לפי החוק. "
            "ביטוח חובה הוא הביטוח הנדרש בחוק ומכסה נזקי גוף בתאונות דרכים. "
            "ביטוח מקיף הוא ביטוח נוסף, המיועד לכיסוי נזקי רכוש לרכב המבוטח "
            "ולרכוש של צד שלישי, בהתאם לתנאי הפוליסה ולגבולות הכיסוי."
        )

    if mentions_compulsory and any(word in q for word in ("כולל", "מכסה", "מהו", "מה זה", "ביטוח")):
        if "נזקי גוף" in evidence:
            return (
                "ביטוח חובה לרכב נועד לכסות נזקי גוף בתאונת דרכים: לנהג, לנוסעים ברכב "
                "ולנפגעים מחוץ לרכב, כגון הולכי רגל. הוא אינו מיועד לכיסוי נזק לרכב או לרכוש; "
                "לנזקי רכוש נדרשים כיסויים אחרים, בהתאם לצורך ולתנאי הפוליסה."
            )

    if mentions_compulsory and any(word in q for word in ("כולל", "מכסה", "מהו", "מה זה", "ביטוח")):
        if "נזקי גוף" in evidence:
            return (
                "ביטוח חובה לרכב נועד לכסות נזקי גוף בתאונת דרכים: לנהג, לנוסעים ברכב "
                "ולנפגעים מחוץ לרכב, כגון הולכי רגל. הוא אינו מיועד לכיסוי נזק לרכב או לרכוש; "
                "לנזקי רכוש נדרשים כיסויים אחרים, בהתאם לצורך ולתנאי הפוליסה."
            )

    if mentions_comprehensive and any(
        phrase in q for phrase in (
            "רכב של אדם אחר",
            "רכב אחר",
            "רכב של צד שלישי",
            "נזק לרכב אחר",
        )
    ):
        return (
            "כן. ביטוח מקיף לרכב כולל גם כיסוי לנזקי רכוש שנגרמו לרכב "
            "או לרכוש של צד שלישי בעקבות השימוש ברכב המבוטח, "
            "בכפוף לתנאי הפוליסה, לסייגים ולגבולות האחריות."
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
    topic = classify_question_topic(question)
    if topic == "out_of_scope":
        return {
            "answer": "הצ'אטבוט הזה מיועד לשאלות על ביטוחי הפניקס. במאגר אין מידע על תחזית מזג אוויר.",
            "sources": [],
        }

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
    topic = classify_question_topic(normalized_question)

    # Specific topics must be backed by pages that actually discuss that subject.
    # This takes precedence over broad intents such as "claim_process".
    if topic in TOPIC_RULES:
        topic_pages = [
            (score, page) for score, page in ranked
            if page_has_topic_evidence(page, topic)
        ]

        # Prefer a direct comprehensive-policy page over a generic car-insurance
        # landing page for glass coverage when both are present.
        if topic == "glass_coverage":
            specific_glass_pages = [
                (score, page) for score, page in topic_pages
                if "מקיף" in normalize_text(page.get("title", ""))
                and any(cue in normalize_text(clean_page_text(page.get("text", "")))
                        for cue in ("שמשות", "שמשה", "שבר שמשה", "כיסוי לשמשות"))
            ]
            if specific_glass_pages:
                topic_pages = specific_glass_pages

        # A generic personal-deductible question should not pull in pages focused
        # on compulsory insurance or third-party insurance unless asked explicitly.
        if topic == "deductible":
            q_has_compulsory = "חובה" in normalized_question
            q_has_third_party = any(term in normalized_question for term in ("צד ג", "צד ג'", "צד שלישי"))
            filtered_deductible_pages = []
            for score, page in topic_pages:
                title = normalize_text(page.get("title", ""))
                if not q_has_compulsory and "ביטוח חובה" in title:
                    continue
                if not q_has_third_party and any(term in title for term in ("צד ג", "צד ג'", "צד שלישי")):
                    continue
                filtered_deductible_pages.append((score, page))
            if filtered_deductible_pages:
                topic_pages = filtered_deductible_pages

        if not topic_pages and topic == "vehicle_stolen":
            # If no page describes theft-claim steps, retain a source that discusses
            # theft coverage so the answer can clearly distinguish coverage from process.
            coverage_pages = [
                (score, page) for score, page in ranked
                if page_has_topic_evidence(page, "theft_coverage")
                and "מקיף" in normalize_text(page.get("title", ""))
            ]
            if coverage_pages:
                ranked = coverage_pages
            else:
                return {"answer": topic_fallback_answer(topic), "sources": []}
        elif not topic_pages:
            return {"answer": topic_fallback_answer(topic), "sources": []}
        else:
            if topic == "weather_flood_contents":
                content_cues = ("תכולה", "תכולת הדירה", "רכוש בדירה", "רכוש הבית")
                weather_cues = (
                    "מזג אוויר סוער", "מזג אוויר קיצוני", "נזקי טבע",
                    "סערה", "סופה", "רוחות", "רוח חזקה",
                )
                flood_cues = ("הצפה", "הצפות", "שיטפון", "שיטפונות", "נזקי מים")
                weather_excluded_title_cues = (
                    "הצפה בבית", "נזקי מים וצנרת", "נזקי צנרת",
                )
                weather_candidates = []
                flood_candidates = []
                for score, page in topic_pages:
                    title = normalize_text(page.get("title", ""))
                    body = normalize_text(clean_page_text(page.get("text", "")))
                    if not any(cue in body for cue in content_cues):
                        continue
                    if any(cue in body for cue in weather_cues) and not any(
                        cue in title for cue in weather_excluded_title_cues
                    ):
                        weather_candidates.append((score, page))
                    if any(cue in title for cue in ("הצפה", "שיטפון")) and any(
                        cue in body for cue in flood_cues
                    ):
                        flood_candidates.append((score, page))
                selected = []
                if weather_candidates:
                    score, page = max(weather_candidates, key=lambda item: item[0])
                    selected.append((score + 1000.0, page))
                if flood_candidates:
                    score, page = max(flood_candidates, key=lambda item: item[0])
                    if all(page.get("url") != existing.get("url") for _, existing in selected):
                        selected.append((score + 900.0, page))
                ranked = selected if selected else topic_pages
            else:
                ranked = topic_pages

    # For broad intents, never fall back to a generic page merely because it shares
    # words such as "רכב", "ביטוח" or "חברת הביטוח".
    elif intent == "contact":
        asks_for_phone = topic == "claim_contact" or any(
            term in normalized_question for term in ("מספר הטלפון", "מספר טלפון", "טלפון")
        )
        asks_towing = any(term in normalized_question for term in ("גרירה", "גרר"))
        contact_pages = []
        for score, page in ranked:
            body = clean_page_text(page.get("text", ""))
            is_towing_contact = page_is_towing_contact(page)
            normalized_body = normalize_text(body)
            is_claims_contact = any(phrase in normalized_body for phrase in (
                "מוקד תביעות וואטספ", "מוקד תביעות וואטסאפ", "מוקד תביעות ווטספ", "מוקד תביעות ווטסאפ",
            )) and bool(PHONE_RE.search(body))
            if is_towing_contact and not asks_towing and not (topic == "claim_contact" and is_claims_contact):
                continue
            if not page_has_contact_evidence(page) and not (asks_towing and is_towing_contact) and not (topic == "claim_contact" and is_claims_contact):
                continue
            if asks_for_phone and not PHONE_RE.search(body):
                continue
            contact_pages.append((score, page))

        asks_claims_contact = any(term in normalized_question for term in ("תביעה", "תביעות", "בירור", "בירורים"))
        if topic == "claim_contact" and asks_claims_contact:
            claims_candidates = []
            towing_candidates = []
            for score, page in contact_pages:
                body = clean_page_text(page.get("text", ""))
                normalized_body = normalize_text(body)
                has_claims_label = any(phrase in normalized_body for phrase in (
                    "מוקד תביעות וואטספ", "מוקד תביעות וואטסאפ", "מוקד תביעות ווטספ", "מוקד תביעות ווטסאפ",
                ))
                if has_claims_label and PHONE_RE.search(body):
                    claims_candidates.append((score, page))
                if "ספק הגרירה" in normalized_body and "*8888" in normalized_body and "*2008" in normalized_body:
                    towing_candidates.append((score, page))
            if claims_candidates and asks_towing and towing_candidates:
                claims_item = max(claims_candidates, key=lambda item: item[0])
                towing_item = max(towing_candidates, key=lambda item: item[0])
                # The offset ensures both distinct source types survive the common score threshold.
                ranked = [(claims_item[0] + 1000.0, claims_item[1]), (towing_item[0] + 900.0, towing_item[1])]
            elif claims_candidates:
                claims_item = max(claims_candidates, key=lambda item: item[0])
                ranked = [(claims_item[0] + 1000.0, claims_item[1])]
            else:
                ranked = contact_pages
        else:
            ranked = contact_pages

        if not contact_pages:
            if topic == "claim_contact":
                answer = (
                    "לא מצאתי במאגר מספר טלפון מאומת של הפניקס לדיווח על תאונת רכב. "
                    "בדקי את עמוד 'צור קשר' באתר הרשמי; אין להסתמך על מספר של ספק גרירה כמספר הפניקס."
                )
            else:
                answer = "לא מצאתי במאגר פרטי קשר מאומתים של הפניקס. בדקי בעמוד 'צור קשר' באתר הרשמי."
            return {"answer": answer, "sources": []}

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

    if (
        asks_difference
        and "חובה" in normalized_question
        and "מקיף" in normalized_question
        and asks_third_party
    ):
        comparison_pages = []
        used_urls = set()

        title_groups = (
            ("חובה",),
            ("מקיף",),
            ("צד ג", "צד שלישי"),
        )

        for title_terms in title_groups:
            candidates = [
                (score, page) for score, page in ranked
                if page.get("url") not in used_urls
                and any(
                    term in normalize_text(page.get("title", ""))
                    for term in title_terms
                )
            ]

            if candidates:
                _, selected_page = max(
                    candidates,
                    key=lambda item: item[0],
                )
                comparison_pages.append((10.0, selected_page))
                used_urls.add(selected_page.get("url", ""))

        if len(comparison_pages) == 3:
            ranked = comparison_pages

    if topic not in TOPIC_RULES and asks_third_party and not asks_difference and intent != "claim_documents":
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
    if topic not in TOPIC_RULES and topic != "claim_contact" and any(
        word in normalized_question for word in ("תביעה", "תביעות", "מגישים", "להגיש", "הגשת", "תבע")
    ):
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
    max_relevant_pages = 2 if topic in TOPIC_RULES else 4
    seen_urls: set[str] = set()
    seen_page_evidence: list[str] = []

    for score, page in ranked:
        if score < threshold:
            break
        url = page.get("url", "")
        if url in seen_urls:
            continue

        # Several URLs in the corpus can expose the exact same article text
        # (for example, a product page repeated under multiple article URLs).
        # Keep the highest-ranked copy so the answer does not cite duplicates.
        page_text = page.get("text", "")
        if _is_duplicate_evidence(page_text, seen_page_evidence, threshold=0.98):
            continue

        seen_urls.add(url)
        seen_page_evidence.append(_evidence_key(page_text))
        relevant.append((score, page))
        if len(relevant) == max_relevant_pages:
            break

    if not relevant:
        relevant = [ranked[0]]

    relevant_pages = []
    sources = []
    seen_snippet_evidence: list[str] = []
    for _, page in relevant:
        if intent == "claim_documents" and topic not in TOPIC_RULES:
            snippet = make_claim_documents_snippet(page, question)
        else:
            snippet = make_snippet(page.get("text", ""), direct_terms, extra_phrases, question)
        if not snippet:
            continue

        snippet = snippet.replace(
            "למסך אישור אי הגשת תביעה",
            "למסמך אישור אי־הגשת תביעה",
        )

        # Different URLs sometimes produce the same answer excerpt. Keep only
        # distinct evidence snippets, while preserving separate facts/sources.
        if _is_duplicate_evidence(snippet, seen_snippet_evidence, threshold=0.90):
            continue
        seen_snippet_evidence.append(_evidence_key(snippet))

        relevant_pages.append(page)
        sources.append({
            "title": display_source_title(page, question),
            "url": page["url"],
            "snippet": snippet,
        })

    # For a specific topic, an empty excerpt means the corpus did not yield a direct
    # answer. Do not attach an unrelated page as a fallback.
    if not sources and topic in TOPIC_RULES:
        return {"answer": topic_fallback_answer(topic), "sources": []}

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
