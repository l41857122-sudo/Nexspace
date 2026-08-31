"""
router.py
---------
Central routing/query-structuring logic for the Geospatial Vision-Language
Interface. This module contains NO model inference — it only decides:

    * which downstream tools should be invoked
    * how to restructure a free-text user query into RSVQA-style
      sub-questions when VQA is required
    * whether a low-confidence counting warning must be surfaced

The output of `route()` is the JSON routing decision described in the
system spec, e.g.:

{
  "target_tools": ["VQA", "Optical_Caption"],
  "restructured_vqa_queries": ["Is there a river present?"],
  "requires_count_warning": false,
  "execution_reasoning": "..."
}
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Lightweight intent classification
# ---------------------------------------------------------------------------

COUNT_PATTERN = re.compile(r"\bhow many\b", re.IGNORECASE)

# Words/phrases that typically signal a closed-ended (yes/no, category,
# attribute) question RSVQA-style models are good at.
CLOSED_ENDED_STARTERS = (
    "is there", "are there", "is it", "are they", "does", "do",
    "was there", "were there", "can you see a", "which", "what type of",
    "what color", "what is the color of",
)

# Phrases that signal a genuinely open-ended descriptive request that
# should never be sent to VQA verbatim.
OPEN_ENDED_MARKERS = (
    "what is visible", "what can you see", "describe", "tell me about",
    "what's in this image", "what is in this image", "summarize",
    "give me an overview", "what does this scene show",
)


def _is_counting(query: str) -> bool:
    return bool(COUNT_PATTERN.search(query))


def _is_open_ended(query: str) -> bool:
    q = query.lower().strip()
    if any(marker in q for marker in OPEN_ENDED_MARKERS):
        return True
    # Heuristic: doesn't start with a closed-ended pattern and doesn't
    # contain a counting pattern -> treat as open-ended by default.
    if _is_counting(q):
        return False
    if any(q.startswith(starter) for starter in CLOSED_ENDED_STARTERS):
        return False
    # Fallback: questions starting with "what" that aren't in the
    # closed-ended starter list are usually open-ended ("what is visible?").
    if q.startswith("what") and not any(q.startswith(s) for s in CLOSED_ENDED_STARTERS):
        return True
    return False


def _normalize_closed_ended(query: str) -> str:
    """Ensure a closed-ended query is phrased in clean RSVQA style."""
    q = query.strip()
    if not q.endswith("?"):
        q += "?"
    return q[0].upper() + q[1:]


def _normalize_counting(query: str) -> str:
    """Force counting queries into the required 'how many [object]?' syntax."""
    match = re.search(r"how many ([a-zA-Z\s]+?)(\?|$)", query, re.IGNORECASE)
    obj = match.group(1).strip() if match else "objects"
    return f"how many {obj}?"


# ---------------------------------------------------------------------------
# Open-ended decomposition
# ---------------------------------------------------------------------------

FEATURE_PROBES = {
    "river": "Is there a river present?",
    "water": "Is there a body of water present?",
    "residential": "Are there residential buildings?",
    "building": "Are there buildings present?",
    "road": "Is there a road present?",
    "vegetation": "Is there vegetation present?",
    "forest": "Is there a forest present?",
    "agriculture": "Is there agricultural land present?",
    "industrial": "Are there industrial structures present?",
}


def decompose_open_ended(query: str, probe_features: Optional[List[str]] = None) -> List[str]:
    """
    Build structured RSVQA sub-questions for an open-ended query.
    """
    if probe_features:
        out = []
        for feat in probe_features:
            key = feat.lower().strip()
            if key in FEATURE_PROBES:
                out.append(FEATURE_PROBES[key])
            else:
                out.append(f"Is there {key} present?")
        return out

    # Default general probe set for a bare "what is visible?" style query.
    return [
        FEATURE_PROBES["water"],
        FEATURE_PROBES["building"],
        FEATURE_PROBES["road"],
        FEATURE_PROBES["vegetation"],
    ]


# ---------------------------------------------------------------------------
# Routing decision object
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    target_tools: List[str] = field(default_factory=list)
    restructured_vqa_queries: List[str] = field(default_factory=list)
    requires_count_warning: bool = False
    execution_reasoning: str = ""
    has_sar: bool = False
    has_optical: bool = False
    has_change_pair: bool = False

    def to_json(self) -> dict:
        return {
            "target_tools": self.target_tools,
            "restructured_vqa_queries": self.restructured_vqa_queries,
            "requires_count_warning": self.requires_count_warning,
            "execution_reasoning": self.execution_reasoning,
        }


def route(
    query: str,
    has_optical: bool = True,
    has_sar: bool = False,
    has_change_pair: bool = False,
    probe_features: Optional[List[str]] = None,
) -> RoutingDecision:
    """
    Core routing function. Decide which tools to call and how to
    restructure the query, following the system's routing rules:

      1. Open-ended -> Optical Captioning (+ optional structured VQA probes)
      2. Counting -> VQA with 'how many X?' syntax + count warning flag
      3. Closed-ended -> VQA directly, normalized to RSVQA phrasing
      4. SAR imagery present -> also route to SAR Captioning
      5. Before/after pair present -> trigger Change Analysis
    """
    decision = RoutingDecision(has_sar=has_sar, has_optical=has_optical, has_change_pair=has_change_pair)
    reasoning_parts = []

    # --- Change analysis short-circuits/augments everything else ---
    if has_change_pair:
        decision.target_tools.append("Change_Analysis")
        reasoning_parts.append(
            "Before/after image pair detected -> triggering Change Analysis "
            "pipeline for pixel-level heatmap and spatial diff."
        )

    if query and query.strip():
        if _is_counting(query):
            decision.target_tools.append("VQA")
            decision.restructured_vqa_queries.append(_normalize_counting(query))
            decision.requires_count_warning = True
            reasoning_parts.append(
                "Detected counting query -> routed to VQA with 'how many [object]?' "
                "syntax; flagged requires_count_warning=True due to known low "
                "accuracy (confidence 0.25-0.40) on counting tasks."
            )
        elif _is_open_ended(query):
            if has_optical:
                decision.target_tools.append("Optical_Caption")
            decision.target_tools.append("VQA")
            decision.restructured_vqa_queries.extend(
                decompose_open_ended(query, probe_features)
            )
            reasoning_parts.append(
                "Open-ended query detected -> routed to Optical Captioning for "
                "free-form scene description, and decomposed into structured "
                "RSVQA-style binary sub-questions for VQA (VQA cannot handle "
                "open-ended phrasing directly)."
            )
        else:
            decision.target_tools.append("VQA")
            decision.restructured_vqa_queries.append(_normalize_closed_ended(query))
            reasoning_parts.append(
                "Closed-ended query detected -> routed directly to VQA after "
                "RSVQA-style normalization."
            )
    elif has_optical and not has_change_pair:
        decision.target_tools.append("Optical_Caption")
        reasoning_parts.append("No query text provided -> defaulting to Optical Captioning for a general scene description.")

    if has_sar:
        decision.target_tools.append("SAR_Caption")
        reasoning_parts.append("SAR imagery provided -> also routing to SAR Captioning for radar-domain description.")

    # Dedup while preserving order
    seen = set()
    decision.target_tools = [t for t in decision.target_tools if not (t in seen or seen.add(t))]

    decision.execution_reasoning = " ".join(reasoning_parts) or "No actionable query or imagery provided."
    return decision
