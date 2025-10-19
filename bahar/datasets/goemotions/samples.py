"""Sample texts in Dutch, Persian, and English for emotion classification."""

from __future__ import annotations

from typing import Final

# Sample texts in different languages expressing various emotions
SAMPLE_TEXTS: Final[dict[str, list[dict[str, str]]]] = {
    "english": [
        {
            "text": "I'm so excited about this amazing opportunity! This is going to be great!",
            "expected_emotion": "excitement",
        },
        {
            "text": "I'm really disappointed with the results. This is not what I expected at all.",
            "expected_emotion": "disappointment",
        },
        {
            "text": "Thank you so much for your help! I really appreciate everything you've done for me.",
            "expected_emotion": "gratitude",
        },
        {
            "text": "I'm confused about what happened. Can someone explain this to me?",
            "expected_emotion": "confusion",
        },
        {
            "text": "This is absolutely disgusting. I can't believe this happened.",
            "expected_emotion": "disgust",
        },
    ],
    "dutch": [
        {
            "text": "Ik ben zo blij met dit geweldige nieuws! Dit is fantastisch!",
            "expected_emotion": "joy",
            "translation": "I'm so happy with this great news! This is fantastic!",
        },
        {
            "text": "Ik ben echt teleurgesteld in de resultaten. Dit had ik niet verwacht.",
            "expected_emotion": "disappointment",
            "translation": "I'm really disappointed in the results. I didn't expect this.",
        },
        {
            "text": "Heel erg bedankt voor je hulp! Ik waardeer alles wat je voor me hebt gedaan.",
            "expected_emotion": "gratitude",
            "translation": "Thank you very much for your help! I appreciate everything you've done for me.",
        },
        {
            "text": "Ik ben bang dat dit niet goed gaat aflopen. Wat moeten we nu doen?",
            "expected_emotion": "fear",
            "translation": "I'm afraid this won't end well. What should we do now?",
        },
        {
            "text": "Dit is zo grappig! Ik kan niet stoppen met lachen!",
            "expected_emotion": "amusement",
            "translation": "This is so funny! I can't stop laughing!",
        },
    ],
    "persian": [
        {
            "text": "من از این خبر عالی خیلی خوشحالم! این فوق‌العاده است!",
            "expected_emotion": "joy",
            "translation": "I'm so happy about this great news! This is wonderful!",
        },
        {
            "text": "واقعاً از نتایج ناامید شدم. این چیزی نبود که انتظار داشتم.",
            "expected_emotion": "disappointment",
            "translation": "I'm really disappointed with the results. This wasn't what I expected.",
        },
        {
            "text": "خیلی ممنون از کمکت! واقعاً قدردان همه کارهایی هستم که برای من انجام دادی.",
            "expected_emotion": "gratitude",
            "translation": "Thank you so much for your help! I really appreciate everything you did for me.",
        },
        {
            "text": "من می‌ترسم که این خوب پیش نرود. حالا باید چه کار کنیم؟",
            "expected_emotion": "fear",
            "translation": "I'm afraid this won't go well. What should we do now?",
        },
        {
            "text": "این خیلی خنده‌دار است! نمی‌تونم جلوی خنده‌ام رو بگیرم!",
            "expected_emotion": "amusement",
            "translation": "This is so funny! I can't stop laughing!",
        },
        {
            "text": "متأسفم بابت اتفاقی که افتاد. این تقصیر من بود.",
            "expected_emotion": "remorse",
            "translation": "I'm sorry about what happened. It was my fault.",
        },
    ],
}


def get_all_samples() -> list[dict[str, str]]:
    """Get all sample texts from all languages."""
    all_samples: list[dict[str, str]] = []
    for language, samples in SAMPLE_TEXTS.items():
        for sample in samples:
            sample_with_lang = sample.copy()
            sample_with_lang["language"] = language
            all_samples.append(sample_with_lang)
    return all_samples


def get_samples_by_language(language: str) -> list[dict[str, str]]:
    """Get sample texts for a specific language."""
    return SAMPLE_TEXTS.get(language.lower(), [])

