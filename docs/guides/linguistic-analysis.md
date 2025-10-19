# Linguistic Categories and Samples

This document describes all linguistic dimensions analyzed by Bahar, with example sentences in English, Dutch, and Persian.

## Overview

Bahar analyzes **4 main linguistic dimensions** with **15 categories** total:

1. **Formality** (2 categories): formal, colloquial
2. **Tone** (4 categories): friendly, rough, serious, kind
3. **Intensity** (3 categories): high, medium, low
4. **Communication Style** (4 categories): direct, indirect, assertive, passive

Plus **emotion categories** from GoEmotions (28 emotions).

## Sample Count

- **Total samples**: 48 (3 per category × 16 categories)
- **Languages**: 3 (English, Dutch, Persian)
- **Total text instances**: 144 (48 samples × 3 languages)

## Categories with Examples

### 1. FORMALITY

#### Formal
Characteristics: Professional language, complex sentences, no contractions

**English**: "I hereby formally request your assistance with this matter."
**Dutch**: "Hierbij verzoek ik u formeel om uw bijstand in deze aangelegenheid."
**Persian**: "بدین‌وسیله رسماً درخواست کمک شما را در این موضوع دارم."

#### Colloquial
Characteristics: Casual language, contractions, slang

**English**: "Hey! Thanks so much for helping me out, you're awesome!"
**Dutch**: "Hey! Heel erg bedankt voor je hulp, je bent geweldig!"
**Persian**: "سلام! خیلی ممنون که کمکم کردی، عالی هستی!"

---

### 2. TONE

#### Friendly
Characteristics: Warm, appreciative, positive

**English**: "You're so kind! I really appreciate your wonderful support!"
**Dutch**: "Je bent zo aardig! Ik waardeer je geweldige steun echt!"
**Persian**: "تو خیلی مهربونی! واقعاً از حمایت فوق‌العاده‌ات قدردانی می‌کنم!"

#### Rough
Characteristics: Harsh, aggressive, impolite

**English**: "Shut up! I don't want to hear it anymore!"
**Dutch**: "Hou je mond! Ik wil het niet meer horen!"
**Persian**: "ساکت شو! دیگه نمی‌خوام بشنوم!"

#### Serious
Characteristics: Grave, urgent, important

**English**: "This is a critical matter that requires immediate attention."
**Dutch**: "Dit is een kritieke kwestie die onmiddellijke aandacht vereist."
**Persian**: "این یک موضوع حیاتی است که نیاز به توجه فوری دارد."

#### Kind
Characteristics: Compassionate, understanding, supportive

**English**: "I understand how you feel. Let me help you with that."
**Dutch**: "Ik begrijp hoe je je voelt. Laat me je daarmee helpen."
**Persian**: "می‌فهمم چه احساسی داری. بذار کمکت کنم."

---

### 3. INTENSITY

#### High Intensity
Characteristics: Strong emotions, emphatic language, exclamation marks

**English**: "I'm EXTREMELY excited about this! This is ABSOLUTELY amazing!!!"
**Dutch**: "Ik ben EXTREEM enthousiast hierover! Dit is ABSOLUUT geweldig!!!"
**Persian**: "من فوق‌العاده هیجان‌زده‌ام! این واقعاً شگفت‌انگیزه!!!"

#### Medium Intensity
Characteristics: Moderate emotions, qualifiers like "quite", "fairly"

**English**: "I'm quite happy about this. It's fairly good news."
**Dutch**: "Ik ben redelijk blij hiermee. Het is vrij goed nieuws."
**Persian**: "من نسبتاً خوشحالم از این. این خبر نسبتاً خوبی است."

#### Low Intensity
Characteristics: Mild emotions, minimizers like "slightly", "a bit"

**English**: "I'm slightly happy about this. It's a bit nice."
**Dutch**: "Ik ben een beetje blij hiermee. Het is een beetje leuk."
**Persian**: "من کمی خوشحالم از این. این یه کم خوبه."

---

### 4. COMMUNICATION STYLE

#### Direct
Characteristics: Clear commands, imperatives, straightforward

**English**: "You must complete this task immediately. Do it now."
**Dutch**: "Je moet deze taak onmiddellijk voltooien. Doe het nu."
**Persian**: "باید این کار را فوراً انجام بدی. الان انجامش بده."

#### Indirect
Characteristics: Hedging, tentative language, questions

**English**: "Perhaps we could consider completing this task when possible?"
**Dutch**: "Misschien kunnen we overwegen deze taak te voltooien wanneer mogelijk?"
**Persian**: "شاید بتونیم در نظر بگیریم این کار رو وقتی ممکنه انجام بدیم؟"

#### Assertive
Characteristics: Confident statements, opinions expressed clearly

**English**: "I believe this is the right approach. In my opinion, we should proceed."
**Dutch**: "Ik geloof dat dit de juiste aanpak is. Naar mijn mening moeten we doorgaan."
**Persian**: "من معتقدم این رویکرد درسته. به نظر من، باید ادامه بدیم."

#### Passive
Characteristics: Apologetic, tentative, seeking permission

**English**: "I'm sorry to bother you, but if possible, could you maybe help?"
**Dutch**: "Sorry dat ik je lastig val, maar als het mogelijk is, zou je misschien kunnen helpen?"
**Persian**: "متأسفم که مزاحمتون می‌شم، ولی اگه ممکنه، شاید بتونید کمک کنید؟"

---

### 5. EMOTION EXAMPLES

#### Sad
**English**: "I'm so sad about what happened. It breaks my heart."
**Dutch**: "Ik ben zo verdrietig over wat er is gebeurd. Het breekt mijn hart."
**Persian**: "من خیلی غمگینم از اتفاقی که افتاد. قلبم رو می‌شکنه."

#### Scared
**English**: "I'm really scared about what might happen. This terrifies me."
**Dutch**: "Ik ben echt bang voor wat er zou kunnen gebeuren. Dit maakt me doodsbang."
**Persian**: "من واقعاً می‌ترسم از اتفاقی که ممکنه بیفته. این منو وحشت‌زده می‌کنه."

#### Surprised
**English**: "Wow! I can't believe this happened! What a surprise!"
**Dutch**: "Wow! Ik kan niet geloven dat dit is gebeurd! Wat een verrassing!"
**Persian**: "وای! باورم نمیشه این اتفاق افتاد! چه شگفتی!"

---

## Usage

### Test All Categories

```bash
python test_linguistic_categories.py
```

This will:
1. Test all 16 categories in English
2. Show multilingual comparisons for key categories
3. Display comprehensive analysis for each sample

### Access Samples Programmatically

```python
from linguistic_samples import (
    LINGUISTIC_SAMPLES,
    get_samples_by_category,
    get_all_categories,
)

# Get all formal samples
formal_samples = get_samples_by_category("formal")

# Get first formal sample in English
text = formal_samples[0]["english"]

# List all categories
categories = get_all_categories()
```

### Analyze Custom Text

```python
from enhanced_classifier import EnhancedEmotionClassifier

classifier = EnhancedEmotionClassifier()
classifier.load_model()

# Analyze any text
result = classifier.analyze("Your text here", top_k=3)

# Access linguistic features
print(f"Formality: {result.linguistic_features.formality}")
print(f"Tone: {result.linguistic_features.tone}")
print(f"Intensity: {result.linguistic_features.intensity}")
print(f"Style: {result.linguistic_features.communication_style}")
```

## Academic Applications

This comprehensive categorization is suitable for:

- **Discourse Analysis**: Study communication patterns across languages
- **Sentiment Analysis**: Combine emotions with linguistic context
- **Formality Studies**: Analyze register variation in multilingual texts
- **Tone Detection**: Identify emotional coloring in communication
- **Intensity Measurement**: Quantify emotional strength
- **Style Analysis**: Characterize communication approaches
- **Cross-linguistic Research**: Compare expression patterns across languages

## Files

- `linguistic_samples.py` - All sample sentences organized by category
- `test_linguistic_categories.py` - Comprehensive testing script
- `linguistic_analyzer.py` - Core linguistic analysis implementation
- `enhanced_classifier.py` - Combined emotion + linguistic classifier

## References

- GoEmotions for emotion taxonomy
- Linguistic politeness theory for formality/style
- Discourse analysis frameworks for tone/intensity

