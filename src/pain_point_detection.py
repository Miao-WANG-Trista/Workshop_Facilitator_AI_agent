# pain_point_detector.py

import json
from pathlib import Path
from typing import Dict, Optional

# 4) Optional spaCy NER
import spacy

# 2) text2emotion for emotion detection
import text2emotion as te

# LangChain Tool
from langchain.agents import Tool

# 3) Zero-shot intent classification
from open_intent_classifier.model import IntentClassifier

# 1) HuggingFace sentiment
from transformers import pipeline as hf_pipeline

# ─── Configuration ──────────────────────────────────────────────────────────────

# Where to log pain points
PAIN_FILE = Path("./data/logs/pain_points.json")

# HF pipelines (lazy-load)
_sentiment_pipe = None
_intent_model = None
_spacy_nlp    = None

# Candidate complaint intents
INTENT_LABELS = ["Complaint", "Issue", "Problem", "Frustration", "Bug report"]

# ─── Helpers ────────────────────────────────────────────────────────────────────

def get_sentiment() -> hf_pipeline:
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = hf_pipeline("sentiment-analysis")
    return _sentiment_pipe

def get_intent_model() -> IntentClassifier:
    global _intent_model
    if _intent_model is None:
        _intent_model = IntentClassifier()
    return _intent_model

def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp

def save_pain(text: str, metadata: Optional[Dict] = None):
    entry = {"text": text, **(metadata or {})}
    with open(PAIN_FILE, "a") as f:
        json.dump(entry, f)
        f.write("\n")

# ─── Detection Logic ────────────────────────────────────────────────────────────

def detect_pain_point(text: str) -> Optional[str]:
    """
    Returns a summary string if this is flagged as a pain point,
    otherwise returns None.
    """

    # 1) Sentiment: negative if score < 0
    sentiment = get_sentiment()(text)[0]
    if sentiment["label"].startswith("NEG") or sentiment["score"] < 0.4:
        save_pain(text, {"method": "sentiment", "score": sentiment["score"]})
        return f"Negative sentiment detected ({sentiment['score']:.2f})"

    # 2) Emotion: high anger or sadness
    emotions = te.get_emotion(text)
    if emotions.get("Angry", 0) > 0.3 or emotions.get("Sad", 0) > 0.3:
        save_pain(text, {"method": "emotion", **emotions})
        return f"Emotion flagged: {emotions}"

    # 3) Intent classification: top label is in our complaint list
    intent_model = get_intent_model()
    intent = intent_model.predict(text, INTENT_LABELS)
    if intent["label"] in INTENT_LABELS and intent["score"] > 0.5:
        save_pain(text, {"method": "intent", **intent})
        return f"Intent classified as {intent['label']} ({intent['score']:.2f})"

    # 4) Optional NER: detect PRODUCT or ORG followed by complaint keywords
    nlp = get_spacy_nlp()
    doc = nlp(text)
    keywords = {"issue", "problem", "challenge", "bug", "error"}
    for ent in doc.ents:
        if ent.label_ in {"PRODUCT", "ORG"}:
            # look for a complaint word nearby
            window = text.lower()[max(ent.start_char-20,0):ent.end_char+20]
            if any(k in window for k in keywords):
                save_pain(text, {"method": "ner", "entity": ent.text})
                return f"NER flagged entity {ent.text} with complaint context"

    return None  # no pain point detected


