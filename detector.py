"""
detector.py — Core AI analysis engine for Fake Review Detector
Uses HuggingFace Transformers (DistilBERT) + rule-based linguistic analysis.
No API key required. Runs 100% locally.
"""

import re
import math
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple


# ── LINGUISTIC PATTERN ANALYZER ───────────────────────────────────────────────

def analyze_linguistic_patterns(text: str) -> Dict:
    """
    Rule-based linguistic feature extractor.
    Returns normalized scores (0.0 – 1.0) for each feature.
    """
    words       = text.split()
    word_count  = max(len(words), 1)
    sentences   = re.split(r'[.!?]+', text)
    sentences   = [s.strip() for s in sentences if s.strip()]

    # 1. Exclamation density
    exclamation_count    = text.count("!")
    exclamation_density  = min(exclamation_count / word_count, 1.0)

    # 2. ALL CAPS ratio
    caps_words  = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio  = min(caps_words / word_count, 1.0)

    # 3. Average word length (normalized: typical is 4-6 chars)
    avg_word_len = sum(len(w) for w in words) / word_count
    avg_word_length = min(avg_word_len / 10.0, 1.0)

    # 4. Vocabulary diversity (unique/total words ratio)
    unique_words        = set(w.lower().strip(".,!?") for w in words)
    unique_word_ratio   = len(unique_words) / word_count

    # 5. Specificity score — presence of concrete details
    specificity_patterns = [
        r'\b\d+\s*(day|week|month|year|hour|minute)s?\b',   # time references
        r'\b\d+\s*%\b',                                      # percentages
        r'\b(compared to|versus|vs\.?)\b',                   # comparisons
        r'\b(however|although|but|despite|while)\b',         # nuance words
        r'\b(specifically|particularly|especially)\b',        # specificity words
        r'\b(broke|cracked|defective|issue|problem|flaw)\b', # specific problems
        r'\b(model|version|size|color|weight)\b',             # product attributes
    ]
    specificity_hits = sum(
        1 for p in specificity_patterns if re.search(p, text, re.IGNORECASE)
    )
    specificity_score = min(specificity_hits / len(specificity_patterns), 1.0)

    # 6. Sentiment extremity — superlatives and absolute words
    extreme_words = [
        "best", "worst", "ever", "amazing", "terrible", "perfect", "awful",
        "incredible", "horrible", "outstanding", "dreadful", "fantastic",
        "absolutely", "completely", "totally", "definitely", "always", "never",
        "greatest", "worst", "unbelievable", "phenomenal", "disgusting"
    ]
    extreme_count = sum(
        1 for w in words if w.lower().strip(".,!?") in extreme_words
    )
    sentiment_extremity = min(extreme_count / max(word_count * 0.1, 1), 1.0)

    return {
        "exclamation_density":  round(exclamation_density, 3),
        "caps_ratio":           round(caps_ratio, 3),
        "avg_word_length":      round(avg_word_length, 3),
        "unique_word_ratio":    round(unique_word_ratio, 3),
        "specificity_score":    round(specificity_score, 3),
        "sentiment_extremity":  round(sentiment_extremity, 3),
    }


def detect_red_flags(text: str, linguistics: Dict) -> List[str]:
    """Identify specific red flags that suggest a fake review."""
    flags = []

    if linguistics["exclamation_density"] > 0.08:
        flags.append("Excessive exclamation marks")
    if linguistics["caps_ratio"] > 0.12:
        flags.append("Overuse of ALL CAPS")
    if linguistics["sentiment_extremity"] > 0.4:
        flags.append("Extreme/absolute language")
    if linguistics["unique_word_ratio"] < 0.5:
        flags.append("Low vocabulary diversity (repetitive)")
    if linguistics["specificity_score"] < 0.15:
        flags.append("Lacks specific details")

    word_count = len(text.split())
    if word_count < 15:
        flags.append("Too short to be informative")

    # Repetition patterns
    if re.search(r'(\b\w+\b)(?:\s+\1){2,}', text, re.IGNORECASE):
        flags.append("Word repetition detected")

    # Incentivized review hints
    incentive_patterns = [
        r'\b(free|discount|received|given|exchange|complimentary)\b',
        r'\bsent (to|me|this|for)\b',
        r'\bin exchange for\b',
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in incentive_patterns):
        flags.append("Possible incentivized review")

    # Urgency / marketing language
    urgency_patterns = [
        r'\b(buy now|order now|don\'t wait|limited time|act fast)\b',
        r'\b(tell your friends|share with|recommend to everyone)\b',
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in urgency_patterns):
        flags.append("Marketing/urgency language detected")

    return flags


def detect_positive_signals(text: str, linguistics: Dict) -> List[str]:
    """Identify signals that suggest a genuine review."""
    signals = []

    word_count = len(text.split())

    if word_count >= 40:
        signals.append("Adequate review length")
    if linguistics["specificity_score"] >= 0.3:
        signals.append("Contains specific details")
    if linguistics["unique_word_ratio"] >= 0.7:
        signals.append("High vocabulary diversity")
    if linguistics["exclamation_density"] <= 0.03:
        signals.append("Measured punctuation use")
    if linguistics["sentiment_extremity"] <= 0.15:
        signals.append("Balanced tone")

    # Balanced opinion markers
    balance_patterns = [
        r'\b(however|but|although|while|despite|on the other hand)\b',
        r'\b(pros?|cons?|downside|upside|positive|negative)\b',
    ]
    if any(re.search(p, text, re.IGNORECASE) for p in balance_patterns):
        signals.append("Balanced perspective noted")

    # Time-based experience
    if re.search(r'\b\d+\s*(day|week|month|year)s?\b', text, re.IGNORECASE):
        signals.append("Time-based experience mentioned")

    # Comparative language
    if re.search(r'\bcompared? (to|with)\b', text, re.IGNORECASE):
        signals.append("Comparative analysis present")

    return signals


def build_explanation(verdict: str, red_flags: List[str], positive_signals: List[str],
                       fake_prob: float, linguistics: Dict) -> str:
    """Generate a human-readable explanation for the verdict."""
    pct = int(fake_prob * 100)

    if verdict == "FAKE":
        main = f"This review has a {pct}% probability of being fake. "
        if red_flags:
            main += f"Key concerns: {', '.join(red_flags[:3])}. "
        main += "The writing style shows patterns commonly seen in manufactured reviews."
        if linguistics["specificity_score"] < 0.2:
            main += " Notably, it lacks the specific details a real user would mention."

    elif verdict == "GENUINE":
        main = f"This review appears genuine ({100-pct}% confidence). "
        if positive_signals:
            main += f"Positive indicators: {', '.join(positive_signals[:3])}. "
        main += "The writing style, vocabulary and detail level are consistent with authentic user experience."

    else:  # SUSPICIOUS
        main = f"This review shows mixed signals ({pct}% fake probability). "
        if red_flags:
            main += f"Concerns: {', '.join(red_flags[:2])}. "
        if positive_signals:
            main += f"But also has: {', '.join(positive_signals[:2])}. "
        main += "Treat with caution and cross-reference with other reviews."

    return main


# ── MAIN DETECTOR CLASS ───────────────────────────────────────────────────────

class ReviewDetector:
    """
    Main detector class.
    Combines HuggingFace DistilBERT sentiment model with
    rule-based linguistic analysis for fake review detection.
    """

    def __init__(self):
        """Load the transformer model. Downloaded once, cached by HuggingFace."""
        self.device = 0 if torch.cuda.is_available() else -1

        # Use DistilBERT fine-tuned on SST-2 (sentiment analysis)
        # We repurpose sentiment confidence as a deception signal:
        # Extremely positive reviews with no nuance are flagged.
        self.sentiment_pipeline = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
            truncation=True,
            max_length=512,
        )

        # Zero-shot classifier for additional deception detection
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-distilroberta-base",
            device=self.device,
        )

        self.fake_labels = ["fake review", "genuine review"]

    def _get_transformer_score(self, text: str) -> float:
        """
        Use zero-shot classification to score fake probability.
        Falls back to sentiment-based heuristic if model fails.
        """
        try:
            # Primary: zero-shot fake detection
            result = self.zero_shot(
                text[:512],  # truncate to model max
                candidate_labels=self.fake_labels,
                hypothesis_template="This is a {}."
            )
            # Return probability of "fake review" label
            fake_idx = result["labels"].index("fake review")
            return float(result["scores"][fake_idx])

        except Exception:
            # Fallback: use sentiment extremity as proxy
            try:
                sent = self.sentiment_pipeline(text[:512])[0]
                # Very high POSITIVE confidence with no nuance = suspicious
                if sent["label"] == "POSITIVE" and sent["score"] > 0.97:
                    return 0.62  # moderately suspicious
                elif sent["label"] == "NEGATIVE" and sent["score"] > 0.97:
                    return 0.45  # slightly suspicious
                return 0.25  # likely genuine
            except Exception:
                return 0.35  # neutral fallback

    def analyze(self, review: str, sensitivity: int = 6) -> Dict:
        """
        Full analysis pipeline for a single review.

        Args:
            review: The review text to analyze
            sensitivity: 1-10, higher = more aggressive fake detection

        Returns:
            Dict with verdict, fake_probability, red_flags, positive_signals,
            linguistics, and explanation
        """
        # 1. Linguistic feature extraction
        linguistics = analyze_linguistic_patterns(review)

        # 2. Transformer-based probability
        transformer_prob = self._get_transformer_score(review)

        # 3. Rule-based adjustment
        rule_adjustment = 0.0

        # Penalize fake signals
        if linguistics["exclamation_density"] > 0.1:    rule_adjustment += 0.12
        if linguistics["caps_ratio"] > 0.15:            rule_adjustment += 0.10
        if linguistics["sentiment_extremity"] > 0.5:    rule_adjustment += 0.12
        if linguistics["unique_word_ratio"] < 0.5:      rule_adjustment += 0.08
        if linguistics["specificity_score"] < 0.1:      rule_adjustment += 0.10
        if len(review.split()) < 15:                    rule_adjustment += 0.08

        # Reward genuine signals
        if linguistics["specificity_score"] > 0.4:      rule_adjustment -= 0.10
        if linguistics["unique_word_ratio"] > 0.75:     rule_adjustment -= 0.08
        if linguistics["exclamation_density"] < 0.02:   rule_adjustment -= 0.05
        if len(review.split()) > 50:                    rule_adjustment -= 0.07

        # Sensitivity scaling (1-10 maps to 0.7x – 1.3x rule weight)
        sensitivity_factor = 0.7 + (sensitivity - 1) * (0.6 / 9)
        rule_adjustment *= sensitivity_factor

        # 4. Combine scores (60% transformer, 40% rules)
        fake_probability = (transformer_prob * 0.60) + \
                           (min(max(transformer_prob + rule_adjustment, 0.0), 1.0) * 0.40)
        fake_probability = round(min(max(fake_probability, 0.0), 1.0), 3)

        # 5. Determine verdict
        # Thresholds adjusted by sensitivity
        fake_threshold       = max(0.45, 0.70 - (sensitivity - 1) * 0.025)
        suspicious_threshold = max(0.25, 0.40 - (sensitivity - 1) * 0.015)

        if fake_probability >= fake_threshold:
            verdict = "FAKE"
        elif fake_probability >= suspicious_threshold:
            verdict = "SUSPICIOUS"
        else:
            verdict = "GENUINE"

        # 6. Red flags & positive signals
        red_flags        = detect_red_flags(review, linguistics)
        positive_signals = detect_positive_signals(review, linguistics)

        # 7. Explanation
        explanation = build_explanation(verdict, red_flags, positive_signals,
                                        fake_probability, linguistics)

        return {
            "verdict":          verdict,
            "fake_probability": fake_probability,
            "linguistics":      linguistics,
            "red_flags":        red_flags,
            "positive_signals": positive_signals,
            "explanation":      explanation,
        }
