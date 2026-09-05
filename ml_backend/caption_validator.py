"""
caption_validator.py
--------------------
Real-Time Output Validation & Repetition Detector for Vision-Language Captioning Models.

Protects against pathological autoregressive generation failures:
  - Repeated single token/word loops (e.g. 'the ta ta ta ta ta...')
  - Consecutive repeating syllables/subwords (e.g. 'a a a a a...', 'building building...')
  - N-gram loops (e.g. 'in the water in the water in the water')
  - Low lexical diversity / vocabulary collapse (Type-Token Ratio < 0.5)
  - Dominant word monotony (> 40% single word in multi-word caption)
  - Special token leakage / unstripped artifacts ('[PAD]', '[UNK]', '<s>', etc.)
  - Empty, whitespace, or degenerate single-character sequences
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set


@dataclass
class CaptionValidationResult:
    """Detailed audit diagnostics for caption generation quality."""
    is_valid: bool
    status: str  # "valid", "invalid_generation", "empty_output", "special_token_leak"
    reason: Optional[str]
    clean_caption: str
    lexical_diversity: float  # Type-Token Ratio (0.0 to 1.0)
    repetition_score: float  # 0.0 (clean) to 1.0 (severe loop)
    diagnostics: dict


# Common safe English stop words that may naturally appear multiple times in long valid sentences
SAFE_STOP_WORDS: Set[str] = {
    "a", "an", "the", "in", "on", "of", "and", "with", "to", "at", "by", "for", "from", "is", "are"
}


def validate_caption_quality(
    raw_caption: Optional[str],
    token_ids: Optional[List[int]] = None,
) -> CaptionValidationResult:
    """
    Authoritative safety filter executed BEFORE model output reaches synthesis.
    Returns CaptionValidationResult indicating whether output can be trusted.
    """
    if raw_caption is None:
        return CaptionValidationResult(
            is_valid=False,
            status="empty_output",
            reason="Caption is None",
            clean_caption="",
            lexical_diversity=0.0,
            repetition_score=1.0,
            diagnostics={"error": "none_input"},
        )

    text = raw_caption.strip()

    # 1. Empty or whitespace check
    if not text or len(text) < 3:
        return CaptionValidationResult(
            is_valid=False,
            status="empty_output",
            reason="Caption is empty or too short (< 3 characters)",
            clean_caption=text,
            lexical_diversity=0.0,
            repetition_score=1.0,
            diagnostics={"length": len(text)},
        )

    # 2. Special Token Leakage Check
    special_token_patterns = [
        r'\[PAD\]', r'\[CLS\]', r'\[SEP\]', r'\[UNK\]', r'\[MASK\]',
        r'<pad>', r'<s>', r'</s>', r'<unk>', r'<mask\d+>', r'##[a-zA-Z0-9]+'
    ]
    for pat in special_token_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return CaptionValidationResult(
                is_valid=False,
                status="special_token_leak",
                reason=f"Special token remnant or subword artifact detected: '{pat}'",
                clean_caption=text,
                lexical_diversity=0.0,
                repetition_score=1.0,
                diagnostics={"pattern": pat},
            )

    # Tokenize words
    words = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
    if not words:
        return CaptionValidationResult(
            is_valid=False,
            status="empty_output",
            reason="No alphanumeric words found in caption",
            clean_caption=text,
            lexical_diversity=0.0,
            repetition_score=1.0,
            diagnostics={"raw_text": text},
        )

    total_words = len(words)
    unique_words = set(words)
    ttr = len(unique_words) / float(total_words)

    # 3. Direct Consecutive Word / Subword Repetition (e.g. 'ta ta ta', 'a a a', 'the the the')
    # Catches 2+ repetitions of any word 3+ times consecutively
    consecutive_match = re.search(r'\b(\w+)(?:\s+\1){2,}\b', text, re.IGNORECASE)
    if consecutive_match:
        repeated_word = consecutive_match.group(1)
        return CaptionValidationResult(
            is_valid=False,
            status="invalid_generation",
            reason=f"Consecutive word repetition loop detected for '{repeated_word}' (e.g. '{repeated_word} {repeated_word} {repeated_word}')",
            clean_caption=text,
            lexical_diversity=round(ttr, 3),
            repetition_score=1.0,
            diagnostics={"repeated_word": repeated_word, "total_words": total_words},
        )

    # Also catch 2 repetitions if word is a short syllable or non-word (e.g. 'ta ta')
    two_repeat_match = re.search(r'\b([a-zA-Z]{1,3})\s+\1\b', text, re.IGNORECASE)
    if two_repeat_match and total_words <= 6:
        repeated_short = two_repeat_match.group(1).lower()
        if repeated_short not in {"that"}:
            return CaptionValidationResult(
                is_valid=False,
                status="invalid_generation",
                reason=f"Repetitive short token loop detected: '{repeated_short} {repeated_short}'",
                clean_caption=text,
                lexical_diversity=round(ttr, 3),
                repetition_score=0.85,
                diagnostics={"repeated_token": repeated_short},
            )

    # 4. Word Frequency Monotony (Single dominant word exceeding 35% in multi-word sentence)
    word_counts: dict[str, int] = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    max_word, max_count = max(word_counts.items(), key=lambda item: item[1])
    word_fraction = max_count / float(total_words)

    if total_words >= 4:
        # Non-stop word taking > 35% of total words
        if max_word not in SAFE_STOP_WORDS and word_fraction >= 0.35 and max_count >= 3:
            return CaptionValidationResult(
                is_valid=False,
                status="invalid_generation",
                reason=f"Word frequency monotony: '{max_word}' constitutes {word_fraction*100:.1f}% of output ({max_count}/{total_words} words)",
                clean_caption=text,
                lexical_diversity=round(ttr, 3),
                repetition_score=round(word_fraction, 2),
                diagnostics={"dominant_word": max_word, "fraction": word_fraction},
            )
        # Even safe stop words cannot exceed 45% with 3+ occurrences
        if word_fraction >= 0.45 and max_count >= 3:
            return CaptionValidationResult(
                is_valid=False,
                status="invalid_generation",
                reason=f"Excessive stop-word repetition: '{max_word}' constitutes {word_fraction*100:.1f}% of words",
                clean_caption=text,
                lexical_diversity=round(ttr, 3),
                repetition_score=round(word_fraction, 2),
                diagnostics={"dominant_word": max_word, "fraction": word_fraction},
            )

    # 5. N-gram Loop Detector (2-gram, 3-gram, 4-gram repetition)
    for n in [2, 3, 4]:
        if total_words >= n * 2:
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            ngram_counts: dict[tuple, int] = {}
            for ng in ngrams:
                ngram_counts[ng] = ngram_counts.get(ng, 0) + 1
            
            for ng, count in ngram_counts.items():
                if count >= 3 or (count >= 2 and total_words <= 8):
                    ng_str = " ".join(ng)
                    return CaptionValidationResult(
                        is_valid=False,
                        status="invalid_generation",
                        reason=f"Repetitive {n}-gram loop detected: '{ng_str}' occurred {count} times",
                        clean_caption=text,
                        lexical_diversity=round(ttr, 3),
                        repetition_score=0.9,
                        diagnostics={"ngram": ng_str, "occurrences": count},
                    )

    # 6. Lexical Diversity Collapse (Type-Token Ratio < 0.50 for captions with >= 5 words)
    if total_words >= 5 and ttr < 0.50:
        return CaptionValidationResult(
            is_valid=False,
            status="invalid_generation",
            reason=f"Low lexical diversity (TTR: {ttr:.2f} < 0.50 threshold across {total_words} words)",
            clean_caption=text,
            lexical_diversity=round(ttr, 3),
            repetition_score=round(1.0 - ttr, 2),
            diagnostics={"ttr": ttr, "unique_words": len(unique_words), "total_words": total_words},
        )

    # 7. Token ID Repetition Check (if raw token IDs provided from model)
    if token_ids and len(token_ids) >= 6:
        # Check for repeating token IDs in the trailing generation
        token_repeats = 0
        for i in range(1, len(token_ids)):
            if token_ids[i] == token_ids[i-1]:
                token_repeats += 1
        if token_repeats >= 3:
            return CaptionValidationResult(
                is_valid=False,
                status="invalid_generation",
                reason=f"Raw model token ID repetition detected ({token_repeats} consecutive identical tokens)",
                clean_caption=text,
                lexical_diversity=round(ttr, 3),
                repetition_score=1.0,
                diagnostics={"token_repeats": token_repeats},
            )

    # 8. Out-of-Domain Generic Artifact Filter (rejects 'a map of', 'screenshot', 'computer screen')
    lower_text = text.lower()
    generic_artifact_triggers = [
        "a map showing", "a map of", "a close up of a map", "screenshot",
        "computer screen", "diagram showing", "a drawing of", "a white background",
        "stock photo", "clip art"
    ]
    for trig in generic_artifact_triggers:
        if trig in lower_text:
            return CaptionValidationResult(
                is_valid=False,
                status="generic_artifact",
                reason=f"Out-of-domain generic artifact detected: '{trig}'",
                clean_caption=text,
                lexical_diversity=round(ttr, 3),
                repetition_score=0.75,
                diagnostics={"trigger": trig},
            )

    # Clean sentence capitalization
    clean_text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    return CaptionValidationResult(
        is_valid=True,
        status="valid",
        reason=None,
        clean_caption=clean_text,
        lexical_diversity=round(ttr, 3),
        repetition_score=0.0,
        diagnostics={"total_words": total_words, "unique_words": len(unique_words), "ttr": ttr},
    )
