"""
router.py
---------
General-purpose Intent Classification & Multi-Tool Routing Engine for
the Geospatial Vision-Language Controller.

Supports broad semantic question families without rigid keyword locks:
  1. SCENE_UNDERSTANDING
  2. OBJECT_IDENTIFICATION
  3. COUNTING
  4. SPATIAL_LOCALIZATION
  5. COMPARISON & CHANGE_INTERPRETATION
  6. OPTICAL_ANALYSIS
  7. SAR_ANALYSIS
  8. OPTICAL_SAR_FUSION
  9. GEOSPATIAL_METADATA
 10. MODEL_INTROSPECTION
 11. OPEN_ENDED_LAND_USE
 12. MULTI_TASK (Decomposed Multi-part queries)
 13. UNSUPPORTED (Out-of-domain)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple


class TaskType(str, Enum):
    VQA = "VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"
    CHANGE_ANALYSIS = "CHANGE_ANALYSIS"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_ANALYSIS = "OPTICAL_SAR_ANALYSIS"
    GEOSPATIAL_METADATA = "GEOSPATIAL_METADATA"
    MODEL_INTROSPECTION = "MODEL_INTROSPECTION"
    OPEN_ENDED_LAND_USE = "OPEN_ENDED_LAND_USE"
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
    confidence_source: str  # e.g. "semantic_intent_router"
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_tools: List[str] = field(default_factory=list)
    restructured_vqa_queries: List[str] = field(default_factory=list)
    requires_count_warning: bool = False
    query_family: str = "GENERAL"
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
            "query_family": self.query_family,
            "reasoning": self.reasoning,
            "is_supported": self.is_supported,
        }


# ---------------------------------------------------------------------------
# Semantic Target & Synonym Extraction
# ---------------------------------------------------------------------------

SYNONYM_MAP = {
    "structures": "buildings",
    "structure": "building",
    "complexes": "buildings",
    "complex": "building",
    "houses": "buildings",
    "house": "building",
    "rooftops": "buildings",
    "roofs": "buildings",
    "infrastructure": "buildings",
    "vessels": "ships",
    "vessel": "ship",
    "boats": "ships",
    "boat": "ship",
    "cars": "vehicles",
    "trucks": "vehicles",
    "automobiles": "vehicles",
    "corridor": "road",
    "highway": "road",
    "street": "road",
    "waterway": "water",
    "river": "water",
    "lake": "water",
    "ocean": "water",
    "sea": "water",
    "canal": "water",
    "piers": "pier",
    "docks": "pier",
    "harbor": "port",
    "forest": "vegetation",
    "trees": "vegetation",
}


def normalize_target_phrase(phrase: str) -> str:
    """Normalize domain synonyms and remove noise words."""
    clean = phrase.strip().lower()
    clean = re.sub(r"\b(this|the|a|an|image|scene|photo|picture|area|region|satellite|imagery)\b", "", clean).strip()
    words = clean.split()
    normalized = [SYNONYM_MAP.get(w, w) for w in words]
    res = " ".join(normalized).strip()
    return res if res else "objects"


def extract_grounding_target(query: str) -> str:
    """Robustly extract target entities from localization, counting, or spatial questions."""
    q_lower = query.strip().lower()

    # Strip conversational prefixes
    q_lower = re.sub(r"^(?:can you|could you|please|i want to|tell me|show me)\s+", "", q_lower).strip()

    # Pattern 1: Counting ("how many X", "count the X", "number of X")
    count_m = re.search(r"(?:how many|count the|count|number of|quantity of|total of)\s+([a-zA-Z\s]+?)(?:\s+(?:are there|can you see|exist|in|on|located)|\?|$)", q_lower)
    if count_m:
        return normalize_target_phrase(count_m.group(1))

    # Pattern 2: Localization ("where are the X", "find the X", "locate the X")
    loc_m = re.search(r"(?:where is|where are|locate|find|detect|pinpoint|segment|box|highlight|identify)\s+(?:the\s+|all\s+|any\s+)?([a-zA-Z\s]+?)(?:\s+(?:in|on|across|within|located)|\?|$)", q_lower)
    if loc_m:
        return normalize_target_phrase(loc_m.group(1))

    # Pattern 3: Presence ("are there X", "is there X")
    pres_m = re.search(r"(?:are there|is there|can you see|do you see)\s+(?:any\s+|a\s+|an\s+)?([a-zA-Z\s]+?)(?:\s+(?:in|on|present|visible)|\?|$)", q_lower)
    if pres_m:
        return normalize_target_phrase(pres_m.group(1))

    # Common remote-sensing target fallback search
    for kw in ["building", "buildings", "ship", "ships", "vessel", "vessels", "road", "roads", "water", "vehicle", "vehicles", "pier", "vegetation", "structure", "structures"]:
        if re.search(rf"\b{kw}\b", q_lower):
            return normalize_target_phrase(kw)

    return "objects"


# Alias for backward compatibility
_extract_grounding_target = extract_grounding_target


# ---------------------------------------------------------------------------
# Out-of-Domain Detection
# ---------------------------------------------------------------------------

OUT_OF_DOMAIN_PATTERNS = [
    r"\bwrite a (?:poem|story|code|essay|song|script)\b",
    r"\bwho is (?:the president|prime minister|ceo)\b",
    r"\bwhat is the capital of\b",
    r"\bbook a (?:flight|ticket|hotel|cab)\b",
    r"\bplay (?:music|song|video)\b",
    r"\btell me a (?:joke|riddle)\b",
    r"\bcalculate \d+\s*[\+\-\*\/]\s*\d+\b",
]


# ---------------------------------------------------------------------------
# Semantic Intent Classifier & Multi-Tool Planner
# ---------------------------------------------------------------------------

class IntentClassifier:
    """
    Intelligent query understanding and multi-tool planner for geospatial queries.
    Determines semantic information need and selects specialist neural tools.
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

        # ---------------------------------------------------------
        # 1. Out-of-Domain Check
        # ---------------------------------------------------------
        if q_raw:
            for pattern in OUT_OF_DOMAIN_PATTERNS:
                if re.search(pattern, q_lower):
                    return ClassificationResult(
                        task_type=TaskType.UNSUPPORTED,
                        confidence=0.95,
                        confidence_type="heuristic",
                        confidence_source="semantic_intent_router",
                        parameters={"raw_query": q_raw},
                        target_tools=[],
                        query_family="UNSUPPORTED",
                        reasoning=f"Query '{q_raw}' is unrelated to geospatial imagery or remote-sensing analysis.",
                        is_supported=False,
                    )

        # ---------------------------------------------------------
        # 2. Geospatial Metadata Queries (CRS, Resolution, GSD, Coords)
        # ---------------------------------------------------------
        is_geo_query = bool(re.search(r"\b(coordinates|crs|resolution|ground sampling distance|gsd|projection|lat.*lon|bounding box coordinates)\b", q_lower))
        if is_geo_query and not any(re.search(p, q_lower) for p in [r"\bchange\b", r"\bdifference\b", r"\bdescribe\b"]):
            return ClassificationResult(
                task_type=TaskType.GEOSPATIAL_METADATA,
                confidence=0.92,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw},
                target_tools=[],
                query_family="GEOSPATIAL_METADATA",
                reasoning="Geospatial metadata inquiry detected (CRS/Resolution/Coordinates/GSD).",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 3. Model Introspection / Provenance Questions
        # ---------------------------------------------------------
        is_introspection_query = bool(re.search(r"\b(how confident|confidence level|why did you|what evidence|which model|how reliable|model provenance)\b", q_lower))
        if is_introspection_query:
            return ClassificationResult(
                task_type=TaskType.MODEL_INTROSPECTION,
                confidence=0.90,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw},
                target_tools=["Grounding", "Optical_Caption"] if has_optical else [],
                query_family="MODEL_INTROSPECTION",
                reasoning="Model introspection and confidence justification requested.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 4. Optical + SAR Cross-Modal Intent
        # ---------------------------------------------------------
        is_optical_sar_query = bool(re.search(
            r"\b(optical and sar|sar and optical|cross-modal|compare.*radar.*optical|compare.*optical.*radar|radar and optical|both sensors|both optical and sar|analyze.*optical.*sar|optical.*sar.*together)\b",
            q_lower,
        ))
        if is_optical_sar_query or (has_optical and has_sar and not has_change_pair and not q_raw):
            tools = ["Optical_SAR_Analysis"]
            target_phrase = extract_grounding_target(q_raw)
            if any(re.search(rf"\b{w}\b", q_lower) for w in ["where", "locate", "find", "detect", "ships", "vessels", "buildings"]):
                tools.append("Grounding")
            if any(re.search(rf"\b{w}\b", q_lower) for w in ["unusual", "anomaly"]):
                tools.append("Anomaly_Extraction")
            if any(re.search(rf"\b{w}\b", q_lower) for w in ["tell me if", "is there", "water", "are there"]):
                tools.append("VQA")

            return ClassificationResult(
                task_type=TaskType.OPTICAL_SAR_ANALYSIS if len(tools) == 1 else TaskType.MULTI_TASK,
                confidence=0.90 if is_optical_sar_query else 0.80,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase, "cross_modal": True},
                target_tools=tools,
                query_family="OPTICAL_SAR_FUSION",
                reasoning="Optical + SAR multimodal query -> routed to cross-modal feature fusion baseline.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 5. SAR / Radar Modality Query
        # ---------------------------------------------------------
        is_sar_only = bool(re.search(r"\b(sar image|radar image|radar imagery|sar data|backscatter|radar pass)\b", q_lower))
        if (is_sar_only or has_sar) and not has_optical and not has_change_pair:
            tools = ["SAR_Caption"]
            if any(w in q_lower for w in ["region", "strong", "anomal", "bright", "where"]):
                tools.append("Anomaly_Extraction")
            return ClassificationResult(
                task_type=TaskType.CAPTIONING if len(tools) == 1 else TaskType.MULTI_TASK,
                confidence=0.88,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw},
                target_tools=tools,
                query_family="SAR_ANALYSIS",
                reasoning="SAR radar query -> routed to SAR backscatter captioning.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 6. Bi-Temporal Change & Anomaly Interpretation
        # ---------------------------------------------------------
        is_change_query = bool(re.search(
            r"\b(what changed|change detection|detect changes|difference between|compare the two|before and after|deforestation|what was built|what was destroyed|changed over time|biggest change|is the change|which areas changed|what disappeared|what was added)\b",
            q_lower,
        ))
        if has_change_pair or is_change_query:
            tools = ["Change_Analysis", "Anomaly_Extraction"]
            target_phrase = extract_grounding_target(q_raw)
            if any(w in q_lower for w in ["building", "structure", "vessel", "ship", "road", "locate", "find", "where"]) and (has_optical or has_change_pair):
                tools.append("Grounding")
            if any(w in q_lower for w in ["tell me if", "is there", "water", "what is", "are there", "describe"]):
                tools.append("VQA")

            norm_vqa = q_raw if q_raw.endswith("?") else f"{q_raw}?"
            return ClassificationResult(
                task_type=TaskType.MULTI_TASK if len(tools) > 1 else TaskType.CHANGE_ANALYSIS,
                confidence=0.92 if has_change_pair else 0.85,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase, "change_threshold": 0.15},
                target_tools=tools,
                restructured_vqa_queries=[norm_vqa] if q_raw else [],
                query_family="COMPARISON",
                reasoning="Bi-temporal change analysis & anomaly proposal requested.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 7. Open-Ended Land-Use / Environmental Feasibility
        # ---------------------------------------------------------
        is_land_use_query = bool(re.search(
            r"\b(suitable for|suitability|can we build|construction feasibility|land use|farming potential|environmental impact|safe to build|good for agriculture)\b",
            q_lower,
        ))
        if is_land_use_query:
            tools = ["Optical_Caption", "Grounding"]
            target_phrase = extract_grounding_target(q_raw)
            return ClassificationResult(
                task_type=TaskType.OPEN_ENDED_LAND_USE,
                confidence=0.78,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase, "open_ended": True},
                target_tools=tools,
                query_family="OPEN_ENDED_LAND_USE",
                reasoning="Open-ended land-use / suitability query -> routed to scene captioning & structural grounding with candid limitations.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 8. Counting Queries ("How many X", "Count the X")
        # ---------------------------------------------------------
        is_counting = bool(re.search(r"\b(how many|count the|count|number of|quantity of|total count)\b", q_lower))
        if is_counting:
            target_phrase = extract_grounding_target(q_raw)
            tools = ["Grounding"]
            # Add VQA query for fallback PaliGemma verification
            vqa_q = f"How many {target_phrase} are present?"
            return ClassificationResult(
                task_type=TaskType.MULTI_TASK,
                confidence=0.86,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase, "is_counting": True},
                target_tools=tools,
                restructured_vqa_queries=[vqa_q],
                requires_count_warning=True,
                query_family="COUNTING",
                reasoning=f"Counting query for '{target_phrase}' -> routed to Grounding DINO feature proposals for exact candidate count.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 9. Spatial / Location Queries ("Where are...", "Which side...")
        # ---------------------------------------------------------
        is_spatial = bool(re.search(r"\b(where is|where are|which side|which region|locate the|largest structure|pinpoint|spatial distribution)\b", q_lower))
        if is_spatial:
            target_phrase = extract_grounding_target(q_raw)
            tools = ["Grounding", "Optical_Caption"]
            return ClassificationResult(
                task_type=TaskType.MULTI_TASK,
                confidence=0.89,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase, "spatial_inquiry": True},
                target_tools=tools,
                query_family="SPATIAL_LOCALIZATION",
                reasoning=f"Spatial location inquiry for '{target_phrase}' -> routed to Grounding DINO and scene context.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 10. Object Identification & Detection ("Can you find...", "Are there...")
        # ---------------------------------------------------------
        is_object_query = bool(re.search(
            r"\b(locate|find|detect|identify|structures present|objects visible|see any|are there|is there|show me)\b",
            q_lower,
        ))
        if is_object_query:
            target_phrase = extract_grounding_target(q_raw)
            tools = ["Grounding", "Optical_Caption"]
            if any(re.search(rf"\b{w}\b", q_lower) for w in ["is there", "are there", "does", "can you tell", "tell me"]):
                tools.append("VQA")
            return ClassificationResult(
                task_type=TaskType.GROUNDING if "locate" in q_lower or "detect" in q_lower else TaskType.MULTI_TASK,
                confidence=0.88,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase},
                target_tools=tools,
                restructured_vqa_queries=[q_raw] if "VQA" in tools else [],
                query_family="OBJECT_IDENTIFICATION",
                reasoning=f"Object identification intent for '{target_phrase}' -> routed to Grounding DINO proposal extractor and visual verification.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 11. General Scene Understanding / Captioning
        # ---------------------------------------------------------
        is_scene_query = bool(re.search(
            r"\b(what is in this image|what do you see|describe the scene|describe this image|describe|what is happening|what type of area|tell me about this area|what can you tell me|overview|summarize)\b",
            q_lower,
        )) or not q_raw
        if is_scene_query:
            tools = ["Optical_Caption"]
            # If structures or objects also mentioned, add grounding
            if any(w in q_lower for w in ["building", "buildings", "structures", "road", "ships", "water"]):
                tools.append("Grounding")
            target_phrase = extract_grounding_target(q_raw) if len(tools) > 1 else "objects"

            return ClassificationResult(
                task_type=TaskType.CAPTIONING if len(tools) == 1 else TaskType.MULTI_TASK,
                confidence=0.90 if q_raw else 0.70,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase},
                target_tools=tools,
                query_family="SCENE_UNDERSTANDING",
                reasoning="General scene understanding request -> routed to BLIP optical scene captioner.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 12. Default Open-Ended Fallback (Safe & Cooperative)
        # ---------------------------------------------------------
        target_phrase = extract_grounding_target(q_raw)
        tools = ["Optical_Caption", "Grounding"]
        norm_vqa = q_raw if q_raw.endswith("?") else f"{q_raw}?"

        return ClassificationResult(
            task_type=TaskType.MULTI_TASK,
            confidence=0.82,
            confidence_type="heuristic",
            confidence_source="semantic_intent_router",
            parameters={"query": q_raw, "target_phrase": target_phrase, "question": norm_vqa},
            target_tools=tools,
            restructured_vqa_queries=[norm_vqa],
            query_family="GENERAL_INQUIRY",
            reasoning=f"Open-ended natural language query '{q_raw}' -> engaged Optical Captioning and Object Grounding for evidence synthesis.",
            is_supported=True,
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
