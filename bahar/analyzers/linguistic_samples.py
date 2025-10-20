"""
Multilingual samples for linguistic dimension testing.

Each sample demonstrates specific linguistic features (formality, tone, intensity, style)
with translations in English, Dutch, and Persian.
"""

from __future__ import annotations

from typing import Final

from bahar.utils.rich_output import console, print_header
from rich.table import Table

# Samples organized by linguistic dimensions
LINGUISTIC_SAMPLES: Final[dict[str, list[dict[str, str]]]] = {
    # FORMALITY: FORMAL
    "formal": [
        {
            "english": "I hereby formally request your assistance with this matter.",
            "dutch": "Hierbij verzoek ik u formeel om uw bijstand in deze aangelegenheid.",
            "persian": "بدین‌وسیله رسماً درخواست کمک شما را در این موضوع دارم.",
            "category": "formal",
        },
        {
            "english": "We are pleased to inform you that your application has been approved.",
            "dutch": "Wij zijn verheugd u te kunnen mededelen dat uw aanvraag is goedgekeurd.",
            "persian": "خوشحالیم که به شما اطلاع دهیم درخواست شما تأیید شده است.",
            "category": "formal",
        },
        {
            "english": "I would like to express my sincere gratitude for your valuable contribution.",
            "dutch": "Ik wil mijn oprechte dankbaarheid uitspreken voor uw waardevolle bijdrage.",
            "persian": "مایلم از مشارکت ارزشمند شما صمیمانه قدردانی کنم.",
            "category": "formal",
        },
    ],
    # FORMALITY: COLLOQUIAL
    "colloquial": [
        {
            "english": "Hey! Thanks so much for helping me out, you're awesome!",
            "dutch": "Hey! Heel erg bedankt voor je hulp, je bent geweldig!",
            "persian": "سلام! خیلی ممنون که کمکم کردی، عالی هستی!",
            "category": "colloquial",
        },
        {
            "english": "I dunno, maybe we could try that if you want?",
            "dutch": "Ik weet het niet, misschien kunnen we dat proberen als je wilt?",
            "persian": "نمی‌دونم، شاید بتونیم اون رو امتحان کنیم اگه می‌خوای؟",
            "category": "colloquial",
        },
        {
            "english": "Wow, that's so cool! I can't believe it!",
            "dutch": "Wow, dat is zo gaaf! Ik kan het niet geloven!",
            "persian": "وای، چقدر باحاله! باورم نمیشه!",
            "category": "colloquial",
        },
    ],
    # TONE: FRIENDLY
    "friendly": [
        {
            "english": "You're so kind! I really appreciate your wonderful support!",
            "dutch": "Je bent zo aardig! Ik waardeer je geweldige steun echt!",
            "persian": "تو خیلی مهربونی! واقعاً از حمایت فوق‌العاده‌ات قدردانی می‌کنم!",
            "category": "friendly",
        },
        {
            "english": "Thanks a lot, friend! You always make my day better!",
            "dutch": "Heel erg bedankt, vriend! Je maakt mijn dag altijd beter!",
            "persian": "خیلی ممنون، دوست! همیشه روزم رو بهتر می‌کنی!",
            "category": "friendly",
        },
        {
            "english": "It's so nice to see you! How have you been?",
            "dutch": "Wat fijn om je te zien! Hoe gaat het met je?",
            "persian": "چقدر خوشحالم که می‌بینمت! چطوری بودی؟",
            "category": "friendly",
        },
    ],
    # TONE: ROUGH
    "rough": [
        {
            "english": "Shut up! I don't want to hear it anymore!",
            "dutch": "Hou je mond! Ik wil het niet meer horen!",
            "persian": "ساکت شو! دیگه نمی‌خوام بشنوم!",
            "category": "rough",
        },
        {
            "english": "This is stupid! What a waste of time!",
            "dutch": "Dit is dom! Wat een tijdverspilling!",
            "persian": "این احمقانه است! چه اتلاف وقتی!",
            "category": "rough",
        },
        {
            "english": "Get lost! I'm done with this nonsense!",
            "dutch": "Ga weg! Ik ben klaar met deze onzin!",
            "persian": "گم شو! دیگه از این مزخرفات خسته شدم!",
            "category": "rough",
        },
    ],
    # TONE: SERIOUS
    "serious": [
        {
            "english": "This is a critical matter that requires immediate attention.",
            "dutch": "Dit is een kritieke kwestie die onmiddellijke aandacht vereist.",
            "persian": "این یک موضوع حیاتی است که نیاز به توجه فوری دارد.",
            "category": "serious",
        },
        {
            "english": "We must address this issue urgently before it escalates.",
            "dutch": "We moeten dit probleem dringend aanpakken voordat het escaleert.",
            "persian": "باید فوراً این مسئله را حل کنیم قبل از اینکه بدتر شود.",
            "category": "serious",
        },
        {
            "english": "The situation is extremely important and cannot be ignored.",
            "dutch": "De situatie is uiterst belangrijk en kan niet worden genegeerd.",
            "persian": "این وضعیت بسیار مهم است و نمی‌توان آن را نادیده گرفت.",
            "category": "serious",
        },
    ],
    # TONE: KIND
    "kind": [
        {
            "english": "I understand how you feel. Let me help you with that.",
            "dutch": "Ik begrijp hoe je je voelt. Laat me je daarmee helpen.",
            "persian": "می‌فهمم چه احساسی داری. بذار کمکت کنم.",
            "category": "kind",
        },
        {
            "english": "Please don't worry. Everything will be alright.",
            "dutch": "Maak je alsjeblieft geen zorgen. Alles komt goed.",
            "persian": "لطفاً نگران نباش. همه چیز درست می‌شه.",
            "category": "kind",
        },
        {
            "english": "Your feelings are valid. I'm here to support you.",
            "dutch": "Je gevoelens zijn geldig. Ik ben er om je te steunen.",
            "persian": "احساسات تو درست هستن. من اینجام که حمایتت کنم.",
            "category": "kind",
        },
    ],
    # INTENSITY: HIGH
    "high_intensity": [
        {
            "english": "I'm EXTREMELY excited about this! This is ABSOLUTELY amazing!!!",
            "dutch": "Ik ben EXTREEM enthousiast hierover! Dit is ABSOLUUT geweldig!!!",
            "persian": "من فوق‌العاده هیجان‌زده‌ام! این واقعاً شگفت‌انگیزه!!!",
            "category": "high_intensity",
        },
        {
            "english": "This is COMPLETELY unacceptable! I'm TOTALLY furious!!",
            "dutch": "Dit is VOLLEDIG onaanvaardbaar! Ik ben TOTAAL woedend!!",
            "persian": "این کاملاً غیرقابل قبوله! من واقعاً خشمگینم!!",
            "category": "high_intensity",
        },
        {
            "english": "I'm INCREDIBLY grateful! You're AMAZINGLY wonderful!!!",
            "dutch": "Ik ben ONGELOOFLIJK dankbaar! Je bent VERBAZINGWEKKEND geweldig!!!",
            "persian": "من فوق‌العاده سپاسگزارم! تو به‌طرز شگفت‌انگیزی عالی هستی!!!",
            "category": "high_intensity",
        },
    ],
    # INTENSITY: MEDIUM
    "medium_intensity": [
        {
            "english": "I'm quite happy about this. It's fairly good news.",
            "dutch": "Ik ben redelijk blij hiermee. Het is vrij goed nieuws.",
            "persian": "من نسبتاً خوشحالم از این. این خبر نسبتاً خوبی است.",
            "category": "medium_intensity",
        },
        {
            "english": "This is rather disappointing. I'm somewhat upset.",
            "dutch": "Dit is nogal teleurstellend. Ik ben enigszins van streek.",
            "persian": "این نسبتاً ناامیدکننده است. من تا حدی ناراحتم.",
            "category": "medium_intensity",
        },
        {
            "english": "I'm pretty grateful for your help. It's moderately helpful.",
            "dutch": "Ik ben behoorlijk dankbaar voor je hulp. Het is redelijk nuttig.",
            "persian": "من نسبتاً از کمکت سپاسگزارم. این تا حدی مفید است.",
            "category": "medium_intensity",
        },
    ],
    # INTENSITY: LOW
    "low_intensity": [
        {
            "english": "I'm slightly happy about this. It's a bit nice.",
            "dutch": "Ik ben een beetje blij hiermee. Het is een beetje leuk.",
            "persian": "من کمی خوشحالم از این. این یه کم خوبه.",
            "category": "low_intensity",
        },
        {
            "english": "This is kind of disappointing. I'm barely upset.",
            "dutch": "Dit is een beetje teleurstellend. Ik ben nauwelijks van streek.",
            "persian": "این یه جورایی ناامیدکننده است. من به سختی ناراحتم.",
            "category": "low_intensity",
        },
        {
            "english": "I'm a little grateful. It's somewhat helpful.",
            "dutch": "Ik ben een beetje dankbaar. Het is enigszins nuttig.",
            "persian": "من کمی سپاسگزارم. این تا حدی مفید است.",
            "category": "low_intensity",
        },
    ],
    # COMMUNICATION STYLE: DIRECT
    "direct": [
        {
            "english": "You must complete this task immediately. Do it now.",
            "dutch": "Je moet deze taak onmiddellijk voltooien. Doe het nu.",
            "persian": "باید این کار را فوراً انجام بدی. الان انجامش بده.",
            "category": "direct",
        },
        {
            "english": "I need your answer right now. Tell me yes or no.",
            "dutch": "Ik heb nu je antwoord nodig. Zeg me ja of nee.",
            "persian": "الان به جوابت نیاز دارم. بگو بله یا نه.",
            "category": "direct",
        },
        {
            "english": "This will not work. We need a different solution.",
            "dutch": "Dit zal niet werken. We hebben een andere oplossing nodig.",
            "persian": "این جواب نمی‌ده. ما به راه‌حل دیگه‌ای نیاز داریم.",
            "category": "direct",
        },
    ],
    # COMMUNICATION STYLE: INDIRECT
    "indirect": [
        {
            "english": "Perhaps we could consider completing this task when possible?",
            "dutch": "Misschien kunnen we overwegen deze taak te voltooien wanneer mogelijk?",
            "persian": "شاید بتونیم در نظر بگیریم این کار رو وقتی ممکنه انجام بدیم؟",
            "category": "indirect",
        },
        {
            "english": "I was wondering if maybe you might have an answer?",
            "dutch": "Ik vroeg me af of je misschien een antwoord zou kunnen hebben?",
            "persian": "داشتم فکر می‌کردم شاید ممکنه جوابی داشته باشی؟",
            "category": "indirect",
        },
        {
            "english": "It might be that this approach could possibly need adjustment.",
            "dutch": "Het zou kunnen dat deze aanpak mogelijk aanpassing nodig heeft.",
            "persian": "ممکنه که این روش احتمالاً نیاز به تنظیم داشته باشه.",
            "category": "indirect",
        },
    ],
    # COMMUNICATION STYLE: ASSERTIVE
    "assertive": [
        {
            "english": "I believe this is the right approach. In my opinion, we should proceed.",
            "dutch": "Ik geloof dat dit de juiste aanpak is. Naar mijn mening moeten we doorgaan.",
            "persian": "من معتقدم این رویکرد درسته. به نظر من، باید ادامه بدیم.",
            "category": "assertive",
        },
        {
            "english": "I think we need to change direction. Clearly, this isn't working.",
            "dutch": "Ik denk dat we van richting moeten veranderen. Duidelijk werkt dit niet.",
            "persian": "فکر می‌کنم باید مسیر رو عوض کنیم. واضحه که این جواب نمی‌ده.",
            "category": "assertive",
        },
        {
            "english": "I'm confident this will succeed. I strongly recommend this solution.",
            "dutch": "Ik ben ervan overtuigd dat dit zal slagen. Ik beveel deze oplossing sterk aan.",
            "persian": "من مطمئنم این موفق می‌شه. قویاً این راه‌حل رو پیشنهاد می‌کنم.",
            "category": "assertive",
        },
    ],
    # COMMUNICATION STYLE: PASSIVE
    "passive": [
        {
            "english": "I'm sorry to bother you, but if possible, could you maybe help?",
            "dutch": "Sorry dat ik je lastig val, maar als het mogelijk is, zou je misschien kunnen helpen?",
            "persian": "متأسفم که مزاحمتون می‌شم، ولی اگه ممکنه، شاید بتونید کمک کنید؟",
            "category": "passive",
        },
        {
            "english": "Excuse me, I don't want to impose, but would you mind assisting?",
            "dutch": "Pardon, ik wil niet opdringen, maar zou je het erg vinden om te helpen?",
            "persian": "ببخشید، نمی‌خوام مزاحم بشم، ولی ممکنه کمک کنید؟",
            "category": "passive",
        },
        {
            "english": "If it's not too much trouble, perhaps you could consider this?",
            "dutch": "Als het niet te veel moeite is, zou je dit misschien kunnen overwegen?",
            "persian": "اگه زیاد زحمت نیست، شاید بتونید این رو در نظر بگیرید؟",
            "category": "passive",
        },
    ],
    # EMOTIONS: SAD
    "sad": [
        {
            "english": "I'm so sad about what happened. It breaks my heart.",
            "dutch": "Ik ben zo verdrietig over wat er is gebeurd. Het breekt mijn hart.",
            "persian": "من خیلی غمگینم از اتفاقی که افتاد. قلبم رو می‌شکنه.",
            "category": "sad",
        },
        {
            "english": "This makes me feel really down. I'm deeply disappointed.",
            "dutch": "Dit maakt me echt somber. Ik ben diep teleurgesteld.",
            "persian": "این باعث می‌شه خیلی افسرده بشم. من عمیقاً ناامیدم.",
            "category": "sad",
        },
        {
            "english": "I feel so lonely and miserable right now.",
            "dutch": "Ik voel me nu zo eenzaam en ellendig.",
            "persian": "الان احساس تنهایی و بدبختی می‌کنم.",
            "category": "sad",
        },
    ],
    # EMOTIONS: SCARED
    "scared": [
        {
            "english": "I'm really scared about what might happen. This terrifies me.",
            "dutch": "Ik ben echt bang voor wat er zou kunnen gebeuren. Dit maakt me doodsbang.",
            "persian": "من واقعاً می‌ترسم از اتفاقی که ممکنه بیفته. این منو وحشت‌زده می‌کنه.",
            "category": "scared",
        },
        {
            "english": "This situation frightens me. I'm very nervous and anxious.",
            "dutch": "Deze situatie maakt me bang. Ik ben erg nerveus en angstig.",
            "persian": "این وضعیت منو می‌ترسونه. من خیلی عصبی و مضطربم.",
            "category": "scared",
        },
        {
            "english": "I feel threatened and unsafe. This is very worrying.",
            "dutch": "Ik voel me bedreigd en onveilig. Dit is zeer zorgwekkend.",
            "persian": "احساس تهدید و ناامنی می‌کنم. این خیلی نگران‌کننده است.",
            "category": "scared",
        },
    ],
    # EMOTIONS: SURPRISED
    "surprised": [
        {
            "english": "Wow! I can't believe this happened! What a surprise!",
            "dutch": "Wow! Ik kan niet geloven dat dit is gebeurd! Wat een verrassing!",
            "persian": "وای! باورم نمیشه این اتفاق افتاد! چه شگفتی!",
            "category": "surprised",
        },
        {
            "english": "This is so unexpected! I'm completely shocked!",
            "dutch": "Dit is zo onverwacht! Ik ben volledig geschokt!",
            "persian": "این خیلی غیرمنتظره بود! من کاملاً شوکه شدم!",
            "category": "surprised",
        },
        {
            "english": "I never saw this coming! This is amazing!",
            "dutch": "Ik had dit nooit zien aankomen! Dit is verbazingwekkend!",
            "persian": "هیچ‌وقت انتظارش رو نداشتم! این شگفت‌انگیزه!",
            "category": "surprised",
        },
    ],
}


def get_samples_by_category(category: str) -> list[dict[str, str]]:
    """Get all samples for a specific linguistic category."""
    return LINGUISTIC_SAMPLES.get(category, [])


def get_all_categories() -> list[str]:
    """Get list of all linguistic categories."""
    return list(LINGUISTIC_SAMPLES.keys())


def get_sample_by_index(category: str, index: int) -> dict[str, str] | None:
    """Get a specific sample by category and index."""
    samples = get_samples_by_category(category)
    if 0 <= index < len(samples):
        return samples[index]
    return None


def get_all_samples_flat() -> list[dict[str, str]]:
    """Get all samples as a flat list."""
    all_samples: list[dict[str, str]] = []
    for samples in LINGUISTIC_SAMPLES.values():
        all_samples.extend(samples)
    return all_samples


def print_category_summary() -> None:
    """Print summary of all categories and sample counts."""
    print_header("Linguistic Sample Categories", f"{len(get_all_categories())} categories")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Category", style="yellow", width=25)
    table.add_column("Samples", style="magenta", justify="right", width=10)

    for category, samples in LINGUISTIC_SAMPLES.items():
        table.add_row(category, str(len(samples)))

    console.print(table)
    console.print(f"\n[bold]Total samples:[/bold] {len(get_all_samples_flat())}")
    console.print(f"[bold]Total categories:[/bold] {len(get_all_categories())}")

