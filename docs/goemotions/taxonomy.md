# GoEmotions Taxonomy

Detailed documentation of the 28 emotion categories in the GoEmotions dataset.

## Taxonomy Overview

The GoEmotions taxonomy was designed to:
1. Maximize coverage of emotions expressed in Reddit data
2. Provide coverage of different types of emotional expressions
3. Limit overlap between emotions
4. Balance data sparsity concerns

## Emotion Groups

### Positive Emotions (12)

#### admiration
**Definition:** Respect and warm approval
**Example:** "Your work is absolutely brilliant!"
**Related:** approval, pride

#### amusement
**Definition:** Finding something funny or entertaining
**Example:** "This is hilarious! I can't stop laughing!"
**Related:** joy

#### approval
**Definition:** Expressing agreement or acceptance
**Example:** "Yes, that's exactly right!"
**Related:** admiration, optimism

#### caring
**Definition:** Displaying kindness and concern for others
**Example:** "I hope you feel better soon. Let me know if you need anything."
**Related:** love, gratitude

#### desire
**Definition:** Strong feeling of wanting something
**Example:** "I really want to visit that place!"
**Related:** excitement, optimism

#### excitement
**Definition:** Feeling of great enthusiasm and eagerness
**Example:** "I'm so excited about this opportunity!!!"
**Related:** joy, desire

#### gratitude
**Definition:** Quality of being thankful
**Example:** "Thank you so much for your help!"
**Related:** caring, approval

#### joy
**Definition:** Feeling of great pleasure and happiness
**Example:** "I'm so happy about this news!"
**Related:** excitement, amusement

#### love
**Definition:** Intense feeling of deep affection
**Example:** "I love spending time with you!"
**Related:** caring, admiration

#### optimism
**Definition:** Hopefulness about the future
**Example:** "I'm sure everything will work out fine!"
**Related:** approval, excitement

#### pride
**Definition:** Feeling of satisfaction from achievements
**Example:** "I'm so proud of what we accomplished!"
**Related:** admiration, joy

#### relief
**Definition:** Feeling of reassurance after anxiety
**Example:** "I'm so relieved that it's finally over!"
**Related:** joy, gratitude

---

### Negative Emotions (11)

#### anger
**Definition:** Strong feeling of annoyance or hostility
**Example:** "This is absolutely unacceptable!"
**Related:** annoyance, disapproval

#### annoyance
**Definition:** Feeling of irritation
**Example:** "This is getting really frustrating."
**Related:** anger, disappointment

#### disappointment
**Definition:** Sadness from unmet expectations
**Example:** "I'm really disappointed with the results."
**Related:** sadness, disapproval

#### disapproval
**Definition:** Possession of unfavorable opinion
**Example:** "I don't think that's a good idea."
**Related:** anger, disappointment

#### disgust
**Definition:** Feeling of revulsion or strong disapproval
**Example:** "This is absolutely disgusting!"
**Related:** anger, disapproval

#### embarrassment
**Definition:** Feeling of self-consciousness or shame
**Example:** "I'm so embarrassed about what happened."
**Related:** nervousness, remorse

#### fear
**Definition:** Unpleasant emotion caused by threat
**Example:** "I'm really scared about what might happen."
**Related:** nervousness, sadness

#### grief
**Definition:** Intense sorrow, especially from death
**Example:** "I'm heartbroken about the loss."
**Related:** sadness, remorse

#### nervousness
**Definition:** Feeling of worry or unease
**Example:** "I'm so nervous about the presentation."
**Related:** fear, embarrassment

#### remorse
**Definition:** Deep regret for wrongdoing
**Example:** "I'm so sorry for what I did. It was my fault."
**Related:** sadness, embarrassment

#### sadness
**Definition:** Feeling of sorrow or unhappiness
**Example:** "I'm so sad about what happened."
**Related:** grief, disappointment

---

### Ambiguous Emotions (4)

#### confusion
**Definition:** Lack of understanding or uncertainty
**Example:** "I'm confused about what happened here."
**Related:** curiosity, surprise

#### curiosity
**Definition:** Strong desire to know or learn
**Example:** "I'm really curious about how this works!"
**Related:** confusion, realization

#### realization
**Definition:** Becoming aware of something
**Example:** "Oh, I see what you mean now!"
**Related:** surprise, curiosity

#### surprise
**Definition:** Feeling of mild astonishment
**Example:** "Wow! I didn't expect that!"
**Related:** realization, confusion

---

### Neutral (1)

#### neutral
**Definition:** No particular emotion expressed
**Example:** "The meeting is at 3 PM."
**Related:** None

## Emotion Clustering

Based on rater judgments, emotions cluster by:

### Sentiment
- **Positive emotions** cluster together
- **Negative emotions** cluster together
- **Ambiguous emotions** cluster together (surprisingly close to positive)

### Intensity
- High intensity: excitement ↔ joy, anger ↔ annoyance
- Medium intensity: approval, caring, disappointment
- Low intensity: neutral, curiosity

### Semantic Similarity
- Joy family: joy, excitement, amusement
- Sadness family: sadness, grief, disappointment
- Anger family: anger, annoyance, disgust
- Fear family: fear, nervousness, embarrassment

## Principal Components

Analysis shows all 28 emotions capture unique dimensions:
- Each emotion is statistically significant (p < 1.5e-6)
- Low overlap between categories
- High inter-rater agreement (94%)

## Usage Guidelines

### When to Use Each Emotion

**Use joy vs. excitement:**
- Joy: General happiness
- Excitement: Anticipatory enthusiasm

**Use anger vs. annoyance:**
- Anger: Strong hostility
- Annoyance: Mild irritation

**Use sadness vs. grief:**
- Sadness: General unhappiness
- Grief: Deep sorrow, often from loss

**Use fear vs. nervousness:**
- Fear: Strong threat perception
- Nervousness: Mild anxiety

## Comparison with Basic 6 Emotions

| Basic 6 | GoEmotions Equivalent | Additional Nuance |
|---------|----------------------|-------------------|
| Joy | joy, excitement, amusement, pride, relief | 5 variations |
| Sadness | sadness, grief, disappointment, remorse | 4 variations |
| Anger | anger, annoyance, disgust | 3 variations |
| Fear | fear, nervousness | 2 variations |
| Surprise | surprise, realization | 2 variations |
| Disgust | disgust | (also under anger family) |

**Additional categories not in Basic 6:**
- admiration, approval, caring, desire, gratitude, love, optimism
- disapproval, embarrassment
- confusion, curiosity
- neutral

## Research Applications

### Sentiment Analysis
- Fine-grained positive sentiment (12 categories)
- Nuanced negative sentiment (11 categories)
- Ambiguous cases (4 categories)

### Emotion Recognition
- Conversational AI
- Customer feedback analysis
- Social media monitoring
- Mental health applications

### Linguistic Studies
- Cross-cultural emotion expression
- Emotion intensity measurement
- Context-dependent emotion analysis

## References

- [GoEmotions Research Blog](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/)
- [GoEmotions GitHub](https://github.com/google-research/google-research/tree/master/goemotions)
- [Research Paper](https://arxiv.org/abs/2005.00547)

