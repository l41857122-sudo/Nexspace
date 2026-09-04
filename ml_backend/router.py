"""
router.py
---------
Structured routing and intent classification logic for the Geospatial
Vision-Language Controller.

Supports single-task and multi-tool composite queries (e.g. captioning + grounding,
change analysis + change VQA, optical + SAR cross-modal).

Task Types:
  1. VQA
  2. CAPTIONING
  3. GROUNDING
  4. CHANGE_ANALYSIS
  5. CHANGE_VQA
  6. OPTICAL_SAR_ANALYSIS
  7. MULTI_TASK
  8. UNSUPPORTED
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class TaskType(str, Enum):
    VQA = "VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"
    CHANGE_ANALYSIS = "CHANGE_ANALYSIS"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_ANALYSIS = "OPTICAL_SAR_ANALYSIS"
    MULTI_TASK = "MULTI_TASK"
    UNSUPPORTED = "UNSUPPORTED"


# ---------------------------------------------------------------------------
# Classification Result Schema
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: float
    confidence_type: str  # "heuristic", "estimated", "model", "calibrated_model"
    confidence_source: str  # e.g. "deterministic_rule_classifier"
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_tools: List[str] = field(default_factory=list)
    restructured_vqa_queries: List[str] = field(default_factory=list)
    requires_count_warning: bool = False
    reasoning: str = ""
    is_supported: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "confidence_source": self.confidence_source,
            "parameters": self.parameters,
            "target_tools": self.target_tools,
            "restructured_vqa_queries": self.restructured_vqa_queries,
            "requires_count_warning": self.requires_count_warning,
            "reasoning": self.reasoning,
            "is_supported": self.is_supported,
        }


# ---------------------------------------------------------------------------
# NLP Patterns & Feature Probes
# ---------------------------------------------------------------------------

COUNT_PATTERN = re.compile(r"\b(how many|count the|number of|total count|quantity of)\b", re.IGNORECASE)

GROUNDING_PATTERNS = [
    r"\bwhere is\b",
    r"\bwhere are\b",
    r"\blocate\b",
    r"\bdetect\b",
    r"\bfind\b",
    r"\bfind the location of\b",
    r"\bbounding box\b",
    r"\bsegment\b",
    r"\bbox the\b",
    r"\bpinpoint\b",
    r"\bcoordinates of\b",
    r"\bshow me\b",
    r"\bwhere (?:is|are)\b",
]

CHANGE_PATTERNS = [
    r"\bwhat changed\b",
    r"\bchange detection\b",
    r"\bdetect changes\b",
    r"\bdifference between\b",
    r"\bcompare the two\b",
    r"\bbefore and after\b",
    r"\bdeforestation\b",
    r"\bconstruction between\b",
    r"\burban expansion\b",
    r"\bwhat was built\b",
    r"\bwhat was destroyed\b",
    r"\bchanged over time\b",
]

OPTICAL_SAR_PATTERNS = [
    r"\boptical and sar\b",
    r"\bsar and optical\b",
    r"\bcross-modal\b",
    r"\bcompare.*radar.*optical\b",
    r"\bcompare.*optical.*radar\b",
    r"\bradar and optical\b",
    r"\bfuse.*sar\b",
    r"\bfuse.*optical\b",
    r"\bboth sensors\b",
    r"\bboth optical and sar\b",
    r"\banalyze.*optical.*sar\b",
    r"\banalyze.*sar.*optical\b",
    r"\boptical.*sar.*together\b",
    r"\bsar.*optical.*together\b",
    r"\buse both optical and sar\b",
]

OPEN_ENDED_MARKERS = (
    "what is visible", "what can you see", "describe", "tell me about",
    "what's in this image", "what is in this image", "summarize",
    "give me an overview", "what does this scene show", "overview of",
    "explain this scene", "general appearance", "land cover of",
    "description", "give me a description", "give a description", "look like",
)

CLOSED_ENDED_STARTERS = (
    "is there", "are there", "is it", "are they", "does", "do",
    "was there", "were there", "can you see a", "which", "what type of",
    "what color", "what is the color of", "is this", "has this",
)

OUT_OF_DOMAIN_PATTERNS = [
    r"\bwrite a (poem|story|code|essay|song)\b",
    r"\bwho is the president\b",
    r"\bwhat is the capital of\b",
    r"\bbook a (flight|ticket|hotel)\b",
    r"\bplay music\b",
    r"\btell me a joke\b",
    r"\bcalculate \d+\s*[\+\-\*\/]\s*\d+\b",
]

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


def _normalize_closed_ended(query: str) -> str:
    """Ensure a closed-ended query is phrased in clean RSVQA style."""
    q = query.strip()
    if not q.endswith("?"):
        q += "?"
    return q[0].upper() + q[1:]


def _normalize_counting(query: str) -> str:
    """Force counting queries into the required 'how many [object]?' syntax."""
    match = re.search(r"(?:how many|count the|number of)\s+([a-zA-Z\s]+?)(\?|$)", query, re.IGNORECASE)
    obj = match.group(1).strip() if match else "objects"
    return f"how many {obj}?"


def _extract_grounding_target(query: str) -> str:
    """Extract candidate object label or composite target phrase from localization query."""
    q_clean = query.strip()
    q_lower = q_clean.lower()

    # Remove leading politeness / question prefixes
    q_lower = re.sub(r"^(?:can you|could you|please)\s+", "", q_lower).strip()

    # Remove trailing image references
    q_lower = re.sub(
        r"\s+(?:in|on|across|within)\s+(?:this|the)\s+(?:image|scene|satellite imagery|picture|photo|aerial photograph|area)[?.!]*$",
        "",
        q_lower,
    ).strip()
    q_lower = re.sub(r"[?.!]+$", "", q_lower).strip()

    for p in GROUNDING_PATTERNS:
        match = re.search(p + r"\s+(?:the\s+|all\s+|any\s+)?(.+)", q_lower)
        if match:
            target = match.group(1).strip()
            target = re.sub(r"\b(this|image|scene|photo|picture)\b", "", target).strip()
            if target:
                return target
    return "objects"


def decompose_open_ended(query: str, probe_features: Optional[List[str]] = None) -> List[str]:
    """Build structured RSVQA sub-questions for an open-ended query."""
    if probe_features:
        out = []
        for feat in probe_features:
            key = feat.lower().strip()
            if key in FEATURE_PROBES:
                out.append(FEATURE_PROBES[key])
            else:
                out.append(f"Is there {key} present?")
        return out

    return [
        FEATURE_PROBES["water"],
        FEATURE_PROBES["building"],
        FEATURE_PROBES["road"],
        FEATURE_PROBES["vegetation"],
    ]


# ---------------------------------------------------------------------------
# Intent Classifier (Deterministic Planning & Task Disambiguation)
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Deterministic rule-based intent classification and multi-tool planner.
    Disambiguates single-task and multi-tool queries and maps them to specialist tools.
    """

    def classify(
        self,
        query: str = "",
        has_optical: bool = False,
        has_sar: bool = False,
        has_change_pair: bool = False,
        probe_features: Optional[List[str]] = None,
    ) -> ClassificationResult:
        q_raw = (query or "").strip()
        q_lower = q_raw.lower()

        # 1. Out-of-Domain / Unsupported Check
        if q_raw:
            for pattern in OUT_OF_DOMAIN_PATTERNS:
                if re.search(pattern, q_lower):
                    return ClassificationResult(
                        task_type=TaskType.UNSUPPORTED,
                        confidence=0.95,
                        confidence_type="heuristic",
                        confidence_source="deterministic_rule_classifier",
                        parameters={"raw_query": q_raw},
                        target_tools=[],
                        reasoning=f"Query '{q_raw}' is out of domain for geospatial remote sensing tasks.",
                        is_supported=False,
                    )

        # 2. Optical + SAR Cross-Modal Intent
        is_optical_sar_query = any(re.search(p, q_lower) for p in OPTICAL_SAR_PATTERNS)
        if is_optical_sar_query or (has_optical and has_sar and not has_change_pair and not q_raw):
            has_grounding = any(re.search(p, q_lower) for p in GROUNDING_PATTERNS)
            has_vqa = (
                any(q_lower.startswith(s) or (f" {s}" in q_lower) for s in CLOSED_ENDED_STARTERS)
                or bool(COUNT_PATTERN.search(q_lower))
                or "tell me if" in q_lower
                or "is there" in q_lower
                or "are there" in q_lower
            )

            tools = ["Optical_SAR_Analysis"]
            if has_grounding:
                tools.append("Grounding")
            if has_vqa:
                tools.append("VQA")
            if "unusual" in q_lower or "anomal" in q_lower or "region" in q_lower:
                tools.append("Anomaly_Extraction")

            target_phrase = _extract_grounding_target(q_raw) if has_grounding else "objects"
            norm_q = _normalize_closed_ended(q_raw) if has_vqa else None
            sub_queries = [norm_q] if norm_q else []

            task = TaskType.MULTI_TASK if len(tools) > 1 else TaskType.OPTICAL_SAR_ANALYSIS
            reasoning = f"Optical + SAR analysis query detected -> selected {tools} for cross-modal feature fusion."
            return ClassificationResult(
                task_type=task,
                confidence=0.90 if is_optical_sar_query else 0.80,
                confidence_type="heuristic",
                confidence_source="deterministic_rule_classifier",
                parameters={"query": q_raw, "target_phrase": target_phrase, "question": norm_q, "cross_modal": True},
                target_tools=tools,
                restructured_vqa_queries=sub_queries,
                reasoning=reasoning,
                is_supported=True,
            )

        # 3. Change Detection / Anomaly Extraction / Change-VQA Intent
        is_change_query = any(re.search(p, q_lower) for p in CHANGE_PATTERNS)
        if has_change_pair or is_change_query:
            has_anomaly_request = (
                any(w in q_lower for w in ["region", "regions", "where", "areas", "find changes", "unusual", "locate", "anomal"])
                or not q_raw
            )
            has_grounding = any(re.search(p, q_lower) for p in GROUNDING_PATTERNS)
            has_vqa = (
                any(q_lower.startswith(s) or (f" {s}" in q_lower) for s in CLOSED_ENDED_STARTERS)
                or bool(COUNT_PATTERN.search(q_lower))
                or "tell me if" in q_lower
                or "is there" in q_lower
                or "are there" in q_lower
            )

            tools = ["Change_Analysis"]
            if has_anomaly_request or has_grounding:
                tools.append("Anomaly_Extraction")
            if has_grounding:
                tools.append("Grounding")
            if has_vqa:
                tools.append("VQA")

            target_phrase = _extract_grounding_target(q_raw) if has_grounding else "objects"
            norm_q = _normalize_closed_ended(q_raw) if has_vqa else None
            sub_queries = [norm_q] if norm_q else []

            task = TaskType.MULTI_TASK if len(tools) > 1 else TaskType.CHANGE_ANALYSIS
            reasoning = f"Bi-temporal change/anomaly analysis requested -> selected {tools}."

            return ClassificationResult(
                task_type=task,
                confidence=0.92 if has_change_pair else 0.85,
                confidence_type="heuristic",
                confidence_source="deterministic_rule_classifier",
                parameters={"query": q_raw, "target_phrase": target_phrase, "question": norm_q, "change_threshold": 0.15},
                target_tools=tools,
                restructured_vqa_queries=sub_queries,
                reasoning=reasoning,
                is_supported=True,
            )

        # 4. Multi-Tool Planning: Composite Queries (Captioning, Grounding, VQA)
        has_caption = any(marker in q_lower for marker in OPEN_ENDED_MARKERS) or "describe" in q_lower or "caption" in q_lower or (
            q_lower.startswith("what") and not any(q_lower.startswith(s) for s in CLOSED_ENDED_STARTERS) and not COUNT_PATTERN.search(q_lower)
        )
        has_grounding = any(re.search(p, q_lower) for p in GROUNDING_PATTERNS)
        has_vqa_markers = (
            any(q_lower.startswith(s) or (f" {s}" in q_lower) for s in CLOSED_ENDED_STARTERS)
            or bool(COUNT_PATTERN.search(q_lower))
            or "tell me if" in q_lower
            or "is there" in q_lower
            or "are there" in q_lower
            or "what type" in q_lower
            or "how many" in q_lower
            or "water in this image" in q_lower
        )

        composite_count = sum([bool(has_caption), bool(has_grounding), bool(has_vqa_markers)])
        if composite_count >= 2:
            tools = []
            if has_caption:
                tools.append("Optical_Caption")
            if has_grounding:
                tools.append("Grounding")
            if has_vqa_markers:
                tools.append("VQA")

            target_phrase = _extract_grounding_target(q_raw) if has_grounding else "objects"
            norm_q = _normalize_counting(q_raw) if COUNT_PATTERN.search(q_lower) else _normalize_closed_ended(q_raw)
            sub_queries = [norm_q] if has_vqa_markers else []

            reasoning = f"Composite multi-tool query detected -> selected {tools}."
            return ClassificationResult(
                task_type=TaskType.MULTI_TASK,
                confidence=0.88,
                confidence_type="heuristic",
                confidence_source="deterministic_rule_classifier",
                parameters={"query": q_raw, "target_phrase": target_phrase, "question": norm_q},
                target_tools=tools,
                restructured_vqa_queries=sub_queries,
                requires_count_warning=bool(COUNT_PATTERN.search(q_lower)),
                reasoning=reasoning,
                is_supported=True,
            )

        # 5. Pure Visual Grounding Intent
        if has_grounding:
            target_phrase = _extract_grounding_target(q_raw)
            return ClassificationResult(
                task_type=TaskType.GROUNDING,
                confidence=0.88,
                confidence_type="heuristic",
                confidence_source="deterministic_rule_classifier",
                parameters={"query": q_raw, "target_phrase": target_phrase},
                target_tools=["Grounding"],
                reasoning=f"Spatial localization intent for '{target_phrase}' -> selected Grounding tool for bounding box detection.",
                is_supported=True,
            )

        # 6. Counting Query Intent
        if q_raw and COUNT_PATTERN.search(q_lower):
            norm_q = _normalize_counting(q_raw)
            return ClassificationResult(
                task_type=TaskType.VQA,
                confidence=0.85,
                confidence_type="heuristic",
                confidence_source="deterministic_rule_classifier",
                parameters={"question": norm_q, "is_counting": True},
                target_tools=["VQA"],
                restructured_vqa_queries=[norm_q],
                requires_count_warning=True,
                reasoning="Counting query detected -> routed to VQA with 'how many [object]?' syntax; flagged requires_count_warning=True.",
                is_supported=True,
            )

        # 7. Pure Open-Ended Descriptive Intent (e.g. "Describe this image")
        if has_caption or (not q_raw and has_optical):
            # Pure image description requests route ONLY to Optical_Caption (no VQA)
            sub_queries = decompose_open_ended(q_raw, probe_features) if probe_features else []
            tools = ["Optical_Caption"]

            reasoning = (
                f"Image description query detected ('{q_raw}') -> routed to Optical_Caption for holistic scene captioning."
                if q_raw else
                "No query text provided -> defaulting to Optical_Caption for general scene overview."
            )

            return ClassificationResult(
                task_type=TaskType.CAPTIONING,
                confidence=0.89 if q_raw else 0.70,
                confidence_type="heuristic",
                confidence_source="deterministic_rule_classifier",
                parameters={"query": q_raw, "probe_features": probe_features or []},
                target_tools=tools,
                restructured_vqa_queries=sub_queries,
                reasoning=reasoning,
                is_supported=True,
            )

        # 8. Closed-Ended VQA Intent
        if q_raw:
            norm_q = _normalize_closed_ended(q_raw)
            return ClassificationResult(
                task_type=TaskType.VQA,
                confidence=0.91,
                confidence_type="heuristic",
                confidence_source="deterministic_rule_classifier",
                parameters={"question": norm_q, "is_counting": False},
                target_tools=["VQA"],
                restructured_vqa_queries=[norm_q],
                requires_count_warning=False,
                reasoning="Closed-ended query detected -> routed directly to VQA with RSVQA-style normalization.",
                is_supported=True,
            )

        # 9. Fallback / Empty
        return ClassificationResult(
            task_type=TaskType.UNSUPPORTED,
            confidence=0.50,
            confidence_type="heuristic",
            confidence_source="deterministic_rule_classifier",
            parameters={"query": ""},
            target_tools=[],
            reasoning="No actionable query or imagery provided.",
            is_supported=False,
        )


# ---------------------------------------------------------------------------
# Backward Compatibility Layer
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


_classifier = IntentClassifier()


def route(
    query: str,
    has_optical: bool = True,
    has_sar: bool = False,
    has_change_pair: bool = False,
    probe_features: Optional[List[str]] = None,
) -> RoutingDecision:
    """
    Backward-compatible entrypoint wrapping the upgraded multi-tool IntentClassifier.
    """
    res = _classifier.classify(
        query=query,
        has_optical=has_optical,
        has_sar=has_sar,
        has_change_pair=has_change_pair,
        probe_features=probe_features,
    )

    tools = list(res.target_tools)
    if has_sar and "SAR_Caption" not in tools and "Optical_SAR_Analysis" not in tools:
        tools.append("SAR_Caption")

    # Preserve dedup order
    seen = set()
    dedup_tools = [t for t in tools if not (t in seen or seen.add(t))]

    return RoutingDecision(
        target_tools=dedup_tools,
        restructured_vqa_queries=res.restructured_vqa_queries,
        requires_count_warning=res.requires_count_warning,
        execution_reasoning=res.reasoning,
        has_sar=has_sar,
        has_optical=has_optical,
        has_change_pair=has_change_pair,
    )
