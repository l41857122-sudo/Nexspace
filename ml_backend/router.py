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


from region_segmenter import TargetGeometry, classify_target_geometry
from target_spec import TargetSpec, SemanticQueryParser, GeometryType
from capability_registry import capability_registry, ModelCapability


class TaskType(str, Enum):
    VQA = "VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"
    COUNTING = "COUNTING"
    PRESENCE_CHECK = "PRESENCE_CHECK"
    SPATIAL_REASONING = "SPATIAL_REASONING"
    REGION_ANALYSIS = "REGION_ANALYSIS"
    GEOLOCATION = "GEOLOCATION"
    CHANGE_ANALYSIS = "CHANGE_ANALYSIS"
    CHANGE_VQA = "CHANGE_VQA"
    OPTICAL_SAR_ANALYSIS = "OPTICAL_SAR_ANALYSIS"
    GEOSPATIAL_METADATA = "GEOSPATIAL_METADATA"
    MODEL_INTROSPECTION = "MODEL_INTROSPECTION"
    OPEN_ENDED_LAND_USE = "OPEN_ENDED_LAND_USE"
    OSM_FEATURE_QUERY = "OSM_FEATURE_QUERY"
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
    target_geometry: Optional[str] = None
    localization_required: bool = False
    target_specs: List[TargetSpec] = field(default_factory=list)
    semantic_plan: Optional[Any] = None
    detection_status: str = "model_ready"

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
            "target_geometry": self.target_geometry,
            "localization_required": self.localization_required,
            "target_specs": [ts.to_dict() for ts in self.target_specs],
            "semantic_plan": self.semantic_plan.to_dict() if hasattr(self.semantic_plan, "to_dict") else None,
            "detection_status": self.detection_status,
        }



# ---------------------------------------------------------------------------
# Semantic Target, Synonym & Generalized Aerial Prompt Expansion
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
    "vessels": "vessels",
    "vessel": "vessel",
    "boats": "boats",
    "boat": "boat",
    "ships": "ships",
    "ship": "ship",
    "cars": "cars",
    "car": "car",
    "trucks": "trucks",
    "truck": "truck",
    "automobiles": "cars",
    "automobile": "car",
    "corridor": "road",
    "highway": "road",
    "highways": "roads",
    "street": "road",
    "streets": "roads",
    "freeway": "road",
    "freeways": "roads",
    "motorway": "road",
    "motorways": "roads",
    "interchange": "road",
    "interchanges": "roads",
    "overpass": "road",
    "overpasses": "roads",
    "viaduct": "bridge",
    "waterway": "water",
    "water body": "water body",
    "water bodies": "water bodies",
    "body of water": "water body",
    "bodies of water": "water bodies",
    "river": "river",
    "rivers": "rivers",
    "lake": "lake",
    "lakes": "lakes",
    "ocean": "water",
    "sea": "water",
    "canal": "water",
    "piers": "pier",
    "docks": "pier",
    "harbor": "port",
    "ports": "port",
    "forest": "vegetation",
    "trees": "vegetation",
    "airplanes": "airplanes",
    "airplane": "airplane",
    "aircraft": "airplanes",
    "planes": "airplanes",
    "plane": "airplane",
    "towers": "towers",
    "tower": "tower",
    "containers": "containers",
    "container": "container",
    "construction site": "construction site",
    "construction sites": "construction sites",
    "construction": "construction site",
    "industrial area": "industrial area",
    "industrial areas": "industrial areas",
    "solar panels": "solar panels",
    "solar panel": "solar panel",
    "solar farm": "solar panels",
    "wind turbines": "wind turbines",
    "wind turbine": "wind turbine",
}

# Reusable semantic expansions for remote-sensing / aerial imagery
AERIAL_PROMPT_EXPANSIONS: Dict[str, str] = {
    "building": "building . rooftop . house . warehouse . structure",
    "buildings": "building . rooftop . house . warehouse . structure",
    "house": "house . residential building . rooftop",
    "houses": "house . residential building . rooftop",
    "structure": "building . structure . industrial facility",
    "structures": "building . structure . industrial facility",
    "warehouse": "warehouse . industrial building . storage depot",
    "warehouses": "warehouse . industrial building . storage depot",
    "vehicle": "car . truck . bus . vehicle",
    "vehicles": "car . truck . bus . vehicle",
    "car": "car . automobile . vehicle",
    "cars": "car . automobile . vehicle",
    "truck": "truck . heavy vehicle",
    "trucks": "truck . heavy vehicle",
    "bus": "bus . passenger bus . vehicle",
    "buses": "bus . passenger bus . vehicle",
    "ship": "ship . boat . vessel . cargo ship",
    "ships": "ship . boat . vessel . cargo ship",
    "boat": "boat . small craft . vessel . yacht",
    "boats": "boat . small craft . vessel . yacht",
    "vessel": "vessel . ship . boat . marine craft",
    "vessels": "vessel . ship . boat . marine craft",
    "airplane": "airplane . aircraft . plane",
    "airplanes": "airplane . aircraft . plane",
    "plane": "airplane . aircraft . plane",
    "planes": "airplane . aircraft . plane",
    "aircraft": "airplane . aircraft . plane",
    "bridge": "bridge . viaduct . highway bridge .",
    "bridges": "bridge . viaduct . highway bridge .",
    "road": "road . highway . bridge . street .",
    "roads": "road . highway . bridge . street .",
    "highway": "highway . motorway . freeway . road .",
    "highways": "highway . motorway . freeway . road .",
    "interchange": "highway interchange . road . highway .",
    "interchanges": "highway interchange . road . highway .",
    "overpass": "road . highway . bridge . street .",
    "overpasses": "road . highway . bridge . street .",
    "freeway": "freeway . highway . motorway . road",
    "freeways": "freeway . highway . motorway . road",
    "motorway": "motorway . highway . freeway . road",
    "motorways": "motorway . highway . freeway . road",
    "water": "water body . water . lake . river . ocean . sea . reservoir . canal",
    "water body": "water body . water . lake . river . ocean . sea . reservoir . canal",
    "water bodies": "water body . water . lake . river . ocean . sea . reservoir . canal",
    "lake": "lake . water body . reservoir . pond",
    "lakes": "lake . water body . reservoir . pond",
    "river": "river . waterway . canal . stream",
    "rivers": "river . waterway . canal . stream",
    "sea": "sea . ocean . coastal water . marine water",
    "ocean": "ocean . sea . coastal water . deep water",
    "coastline": "coastline . shoreline . water edge . beach",
    "shoreline": "shoreline . coastline . water boundary",
    "vegetation": "tree . forest . vegetation . canopy . woodland",
    "forest": "forest . trees . dense vegetation . canopy",
    "tree": "tree . cluster of trees . vegetation",
    "trees": "tree . cluster of trees . vegetation",
    "farmland": "agricultural field . crop plot . farmland . cultivated land",
    "agricultural field": "agricultural field . crop plot . farmland . cultivated land",
    "agricultural fields": "agricultural field . crop plot . farmland . cultivated land",
    "construction site": "construction site . earthworks . bare soil excavation . groundwork",
    "construction sites": "construction site . earthworks . bare soil excavation . groundwork",
    "industrial area": "industrial area . warehouse complex . factory zone . logistics depot",
    "industrial areas": "industrial area . warehouse complex . factory zone . logistics depot",
    "port": "port . harbor . shipping terminal . dock . pier",
    "ports": "port . harbor . shipping terminal . dock . pier",
    "farm": "farm building . agricultural structure",
    "agricultural structure": "farm building . agricultural structure . barn . silo",
    "agricultural structures": "farm building . agricultural structure . barn . silo",
    "tower": "tower . antenna . communication tower . mast",
    "towers": "tower . antenna . communication tower . mast",
    "container": "shipping container . cargo container",
    "containers": "shipping container . cargo container",
    "solar panel": "solar panel . solar array . photovoltaic module",
    "solar panels": "solar panel . solar array . photovoltaic module",
    "wind turbine": "wind turbine . wind generator . turbine tower",
    "wind turbines": "wind turbine . wind generator . turbine tower",
    "pier": "pier . dock . wharf . jetty",
    "piers": "pier . dock . wharf . jetty",
    "airport": "airport runway . tarmac . apron . hangar",
    "airports": "airport runway . tarmac . apron . hangar",
}

# Categorization for semantic bounding-box constraints (compact vs macro)
MACRO_EXTENDED_CATEGORIES = {
    "road", "roads", "highway", "highways", "street", "streets", "freeway", "freeways",
    "motorway", "motorways", "interchange", "interchanges", "bridge", "bridges", "viaduct",
    "overpass", "overpasses", "corridor", "lane", "runway", "airport",
    "water", "water body", "water bodies", "river", "rivers", "ocean", "sea", "lake", "lakes", "canal", "waterway",
    "farmland", "forest", "vegetation", "canopy", "agricultural field", "agricultural fields",
    "construction site", "construction sites", "industrial area", "industrial areas",
    "port", "ports", "coastline", "shoreline"
}


# ---------------------------------------------------------------------------
# Target Geometry Abstraction
# ---------------------------------------------------------------------------
def get_target_geometry(target_phrase: str) -> TargetGeometry:
    """Classifies target phrase into DISCRETE_OBJECT, LINEAR_FEATURE, or REGION_FEATURE."""
    return classify_target_geometry(target_phrase)


def build_generalized_aerial_prompt(target_phrase: str) -> str:
    """
    Builds a generalized, high-performance Grounding DINO aerial prompt.
    Expands known classes into multi-token synonyms; for arbitrary/unknown targets,
    cleans and constructs a generalized singular/plural prompt formatted with trailing period.
    """
    if not target_phrase:
        return "objects."

    clean = target_phrase.strip().lower().rstrip(".")
    # Multi-target split
    if " . " in clean or "," in clean or " and " in clean or " & " in clean or "/" in clean:
        parts = re.split(r"[,&/]|\band\b|\s+\.\s+", clean)
        expanded_parts = []
        for p in parts:
            p_clean = re.sub(r"\b(this|the|a|an|image|scene|photo|picture|area|region|satellite|imagery|visible|present|here|there|detected|seen|are|is|all|any)\b", "", p).strip()
            if not p_clean:
                continue
            if p_clean in AERIAL_PROMPT_EXPANSIONS:
                expanded_parts.append(AERIAL_PROMPT_EXPANSIONS[p_clean])
            else:
                # Generalized unknown target
                words = p_clean.split()
                norm_words = [SYNONYM_MAP.get(w, w) for w in words]
                base_target = " ".join(norm_words)
                if base_target in AERIAL_PROMPT_EXPANSIONS:
                    expanded_parts.append(AERIAL_PROMPT_EXPANSIONS[base_target])
                else:
                    expanded_parts.append(base_target)

        if expanded_parts:
            # Join parts uniquely and append period
            dedup_parts = []
            for ep in expanded_parts:
                for sub in ep.split(" . "):
                    sub_strip = sub.strip()
                    if sub_strip and sub_strip not in dedup_parts:
                        dedup_parts.append(sub_strip)
            return " . ".join(dedup_parts) + " ."

    # Single target
    clean_sub = re.sub(r"\b(this|the|a|an|image|scene|photo|picture|area|region|satellite|imagery|visible|present|here|there|detected|seen|are|is|all|any)\b", "", clean).strip()
    if not clean_sub:
        clean_sub = "objects"

    if clean_sub in AERIAL_PROMPT_EXPANSIONS:
        return AERIAL_PROMPT_EXPANSIONS[clean_sub] + " ."

    words = clean_sub.split()
    norm_words = [SYNONYM_MAP.get(w, w) for w in words]
    base_target = " ".join(norm_words)

    if base_target in AERIAL_PROMPT_EXPANSIONS:
        return AERIAL_PROMPT_EXPANSIONS[base_target] + " ."

    # Arbitrary user target
    if base_target.endswith("s") and len(base_target) > 3:
        return f"{base_target} . {base_target[:-1]} ."
    else:
        return f"{base_target} . {base_target}s ."


def get_canonical_target_label(target_phrase: str) -> str:
    """
    Returns a clean, human-readable canonical entity label (e.g. 'Building', 'Water Body', 'Warehouse', 'Ship', 'Vehicle', 'Road')
    rather than internal dot-delimited prompt expansion tokens, repeated words, or raw BERT subwords.
    """
    if not target_phrase:
        return "Object"
    # Strip subword tokens (e.g. "##pass" -> "pass") and punctuation
    clean = re.sub(r"##", "", target_phrase)
    clean = re.sub(r"[?!.,:;\"'#]", "", clean.strip().lower()).strip()

    # Strip dot-delimited tokens if already passed an expansion string
    if " . " in clean or clean.startswith("building") or clean.startswith("warehouse") or clean.startswith("ship") or clean.startswith("water"):
        primary = clean.split(" . ")[0].strip().rstrip(".")
        if primary in SYNONYM_MAP:
            primary = SYNONYM_MAP[primary]
        return primary.title() if primary else "Object"

    # Strip stopwords and query filler
    clean_sub = re.sub(r"\b(this|the|a|an|image|scene|photo|picture|area|region|satellite|imagery|visible|present|here|there|detected|seen|are|is|all|any|find|locate|where|how|many|can|you|show|me|if|yes|then|it|them)\b", "", clean).strip()
    if not clean_sub:
        clean_sub = clean

    words = clean_sub.split()
    norm_words = []
    for w in words:
        nw = SYNONYM_MAP.get(w, w)
        if nw and nw not in norm_words:
            norm_words.append(nw)

    clean_label = " ".join(norm_words).strip()
    if not clean_label:
        clean_label = "Object"

    return clean_label.title()


def normalize_detection_label(raw_label: str, target_phrase: str = "") -> str:
    """
    Normalizes any sub-token (e.g. 'over', 'pass') or synonym label (e.g. 'highway', 'street', 'boat')
    back to the user's requested canonical target name (e.g. 'Road', 'Ship', 'Building').
    """
    lbl = str(raw_label or "").strip().lower()
    lbl = re.sub(r"##", "", lbl)
    lbl = re.sub(r"[?!.,:;\"'#]", "", lbl).strip()

    tp_raw = str(target_phrase or "").strip()
    tp_clean = re.sub(
        r"\b(find|detect|locate|show|count|where|identify|spot|pinpoint|segment|outline|box|highlight)\b",
        "",
        tp_raw,
        flags=re.IGNORECASE
    ).strip().lower()

    canonical_target = get_canonical_target_label(target_phrase) if target_phrase else (get_canonical_target_label(lbl) if lbl else "Object")

    # 1. Road and Infrastructure Synonyms & Sub-tokens ("over", "pass", "highway", "street", "bridge" -> "Road")
    road_subtokens = {"over", "pass", "overpass", "street", "highway", "avenue", "freeway", "motorway", "interchange", "road", "roads"}
    if any(k in tp_clean for k in ("road", "highway", "street", "lane", "path")):
        if lbl in road_subtokens or "pass" in lbl or "over" in lbl or "road" in lbl or "highway" in lbl or "street" in lbl:
            return canonical_target if canonical_target != "Object" else "Road"
        if lbl in ("bridge", "bridges", "viaduct"):
            return "Road"

    # 2. Bridge queries
    if any(k in tp_clean for k in ("bridge", "viaduct")):
        if lbl in ("bridge", "bridges", "viaduct", "over", "pass", "overpass", "highway overpass"):
            return "Bridge"

    # 3. Ship, Vessel, and Marine Craft ("ship", "boat", "vessel" -> "Ship")
    ship_subtokens = {"ship", "ships", "boat", "boats", "vessel", "vessels", "cargo", "craft", "marine craft"}
    if any(k in tp_clean for k in ("ship", "boat", "vessel", "craft")):
        if lbl in ship_subtokens or any(s in lbl for s in ("ship", "boat", "vessel")):
            return canonical_target if canonical_target != "Object" else "Ship"

    # 4. Building & Structure
    building_subtokens = {"building", "buildings", "house", "houses", "warehouse", "warehouses", "structure", "structures", "roof", "rooftop", "rooftops"}
    if any(k in tp_clean for k in ("building", "house", "warehouse", "structure")):
        if lbl in building_subtokens or any(b in lbl for b in ("build", "house", "warehous", "struct")):
            return canonical_target if canonical_target != "Object" else "Building"

    # 5. Generic sub-word split cleanup (e.g. 'over' or 'pass' standing alone)
    if lbl in ("over", "pass"):
        return canonical_target if canonical_target != "Object" else "Road"

    if canonical_target and canonical_target != "Object" and (canonical_target.lower() in lbl or lbl in canonical_target.lower()):
        return canonical_target

    return get_canonical_target_label(lbl) if lbl else canonical_target


def normalize_target_phrase(phrase: str) -> str:
    """Normalize domain synonyms and format compound multi-targets for joint grounding."""
    clean = phrase.strip().lower()

    # Check for compound multi-targets (e.g. "boats, bridges and buildings", "ships & vehicles")
    if "," in clean or " and " in clean or " & " in clean:
        parts = re.split(r"[,&]|\band\b", clean)
        sub_targets = []
        for p in parts:
            sub_clean = re.sub(r"\b(this|the|a|an|image|scene|photo|picture|area|region|satellite|imagery|visible|present|here|there|detected|seen|are|is|all|any|if|yes|then|it|them)\b", "", p).strip()
            if sub_clean:
                words = sub_clean.split()
                norm_words = [SYNONYM_MAP.get(w, w) for w in words]
                item_str = " ".join(norm_words).strip()
                if item_str and item_str not in sub_targets:
                    sub_targets.append(item_str)
        if len(sub_targets) > 1:
            return " and ".join(sub_targets)
        elif len(sub_targets) == 1:
            return sub_targets[0]

    clean = re.sub(r"\b(this|the|a|an|image|scene|photo|picture|area|region|satellite|imagery|visible|present|here|there|detected|seen|are|is|if|yes|then|it|them)\b", "", clean).strip()
    words = clean.split()
    normalized = [SYNONYM_MAP.get(w, w) for w in words]
    res = " ".join(normalized).strip()
    return res if res else "objects"


def extract_grounding_target(query: str) -> str:
    """Robustly extract single or multi-target entities from localization, counting, or spatial questions."""
    q_lower = query.strip().lower()

    # Strip conversational prefixes and trailing action clauses
    q_lower = re.sub(r"^(?:can you|could you|please|i want to|tell me|show me)\s+", "", q_lower).strip()
    q_lower = re.sub(r"\s+and\s+(?:count|report|summarize|list|segment|box|highlight)\s+(?:them|all|it)\b.*$", "", q_lower).strip()

    # Strip conditional trailing clauses e.g. "if yes then locate it", "if so pinpoint them", "if present locate it"
    q_lower_clean = re.sub(r"\bif\s+(?:yes|so|present|visible|any|there\s+is|there\s+are)\b.*$", "", q_lower).strip()
    q_lower_clean = re.sub(r"\bthen\s+(?:locate|find|detect|pinpoint|box|segment|show)\s+(?:them|it|all)\b.*$", "", q_lower_clean).strip()
    q_lower_clean = re.sub(r"\b(?:locate|find|detect|pinpoint|box|highlight|show)\s+(?:it|them|all)\b.*$", "", q_lower_clean).strip()

    # Preposition boundary detection
    prep_split = re.split(r"\b(?:on|in|near|beside|around|across|inside|along|at|adjacent to|close to|by|surrounding|bordering|flanking|over|under)\b", q_lower_clean)
    primary_clause = prep_split[0].strip() if prep_split else q_lower_clean

    # Pattern 1: Presence queries ("is there a water body in the image", "are there any houses")
    pres_m = re.search(r"(?:are there|is there|can you see|do you see|tell me if there is|tell me if there are)\s+(?:any\s+|a\s+|an\s+)?([a-zA-Z\s,\&]+?)(?:\s+(?:in|on|across|within|present|visible|if|then)|\?|$)", q_lower_clean if q_lower_clean else q_lower)
    if pres_m:
        cand = normalize_target_phrase(pres_m.group(1))
        if cand and cand not in {"it", "them", "all", "any", "objects"}:
            return cand

    # Pattern 2: Counting ("how many X", "count the X", "number of X")
    count_m = re.search(r"(?:how many|count the|count|number of|quantity of|total of)\s+([a-zA-Z\s,\&]+?)(?:\s+(?:are there|are visible|are present|can you see|exist|in|on|located|visible|present)|\?|$)", q_lower_clean if q_lower_clean else q_lower)
    if count_m:
        cand = normalize_target_phrase(count_m.group(1))
        if cand and cand not in {"it", "them", "all", "any"}:
            return cand

    # Pattern 3: Localization ("where are the X", "find the X", "locate the X")
    loc_m = re.search(r"(?:where is|where are|locate|find|detect|pinpoint|segment|box|highlight|identify)\s+(?:the\s+|all\s+|any\s+)?([a-zA-Z\s,\&]+?)(?:\s+(?:in|on|across|within|located|visible|present|if|then)|\?|$)", q_lower)
    if loc_m:
        cand = normalize_target_phrase(loc_m.group(1))
        if cand and cand not in {"it", "them", "all", "any", "objects"}:
            return cand

    # Check for multi-target combinations (e.g. "ships and vehicles", "bridges and roads", "towers and buildings")
    multi_m = re.search(r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+(?:and|&|,)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\b", primary_clause if primary_clause else q_lower)
    if multi_m:
        t1 = multi_m.group(1).strip()
        t2 = multi_m.group(2).strip()
        clean_stopwords = {"in", "on", "the", "a", "an", "this", "that", "scene", "image", "visible", "if", "yes", "then", "it", "find", "locate", "detect", "show"}
        if t1 not in clean_stopwords and t2 not in clean_stopwords:
            return normalize_target_phrase(f"{t1} and {t2}")

    # Positional & Head-Noun Precedence Entity Matching
    # In English noun phrases ("ocean freight container", "storage tank", "water body"), head nouns appear rightmost.
    all_entity_keywords = [
        "water body", "water bodies", "body of water", "bodies of water", "water", "river", "rivers", "lake", "lakes",
        "ocean", "sea", "canal", "coastline", "shoreline", "building", "buildings", "ship", "ships", "boat", "boats",
        "vessel", "vessels", "road", "roads", "highway", "highways", "vehicle", "vehicles", "car", "cars", "truck", "trucks",
        "motor vehicle", "motor vehicles", "freight container", "freight containers",
        "airplane", "airplanes", "aircraft", "plane", "planes", "pier", "piers", "vegetation", "structure", "structures",
        "bridge", "bridges", "tower", "towers", "container", "containers", "farmland", "forest", "construction site",
        "construction sites", "industrial area", "industrial areas", "solar panel", "solar panels", "wind turbine", "wind turbines",
        "runway", "tarmac", "taxiway", "harbor", "port", "storage tank", "storage tanks", "crane", "cranes"
    ]

    # Search in primary target clause first, then full query
    search_scope = primary_clause if primary_clause else q_lower_clean
    found_matches = []
    for kw in all_entity_keywords:
        for m in re.finditer(rf"\b{re.escape(kw)}\b", search_scope):
            found_matches.append((m.start(), len(kw), kw))

    if not found_matches and search_scope != q_lower_clean:
        for kw in all_entity_keywords:
            for m in re.finditer(rf"\b{re.escape(kw)}\b", q_lower_clean):
                found_matches.append((m.start(), len(kw), kw))

    if found_matches:
        # Sort by: 1) Position in clause (rightmost head noun first), 2) Longest match length
        found_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best_kw = found_matches[0][2]
        return normalize_target_phrase(best_kw)

    if SemanticQueryParser.is_scene_description(query) or SemanticQueryParser.is_scene_description(primary_clause):
        return None

    # If no predefined keyword found, normalize primary clause
    cleaned_primary = re.sub(r"\b(find|locate|detect|show|pinpoint|where|is|are|the|a|an|any|all|this|that|image|scene|describe|description)\b", "", primary_clause).strip()
    if cleaned_primary and not SemanticQueryParser.is_scene_description(cleaned_primary) and cleaned_primary not in SemanticQueryParser.NON_TARGET_TOKENS:
        return normalize_target_phrase(cleaned_primary)

    return "objects"


def is_macro_target(target_phrase: str) -> bool:
    """Checks if target phrase represents an extended/macro landscape entity."""
    clean = (target_phrase or "").strip().lower()
    return any(m in clean for m in MACRO_EXTENDED_CATEGORIES)


def extract_detection_targets(query: str) -> List[str]:
    """
    Extracts one or multiple distinct detection target entities from a natural-language query.
    e.g. 'Find cars and buildings' -> ['cars', 'buildings']
         'Detect ships, containers and cranes in the port' -> ['ships', 'containers', 'cranes']
    """
    normalized = extract_grounding_target(query)
    if not normalized:
        return []
    if " and " in normalized or " & " in normalized or "," in normalized:
        parts = re.split(r"[,&]|\band\b", normalized)
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned if cleaned else [normalized]
    return [normalized]


# Alias for backward compatibility
_extract_grounding_target = extract_grounding_target


# ---------------------------------------------------------------------------
# Out-of-Domain Detection
# ---------------------------------------------------------------------------

OUT_OF_DOMAIN_PATTERNS = [
    r"\bwrite a (?:poem|story|code|essay|song|script)\b",
    r"\bwho is (?:the\s+)?(?:president|prime minister|ceo|governor|leader)\b",
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
        res = self._classify_core(
            query=query,
            has_optical=has_optical,
            has_sar=has_sar,
            has_change_pair=has_change_pair,
            probe_features=probe_features,
        )
        q_raw = (query or "").strip()
        from target_spec import extract_safety_constraints
        clean_q, constraints = extract_safety_constraints(q_raw)
        q_lower = clean_q.lower() if clean_q else q_raw.lower()
        plan = SemanticQueryParser.parse_semantic_plan(q_raw)
        res.semantic_plan = plan
        res.detection_status = plan.detection_status
        if constraints:
            res.parameters["constraints"] = constraints
            res.parameters["clean_query"] = clean_q

        # Synchronize task_type with plan
        if res.task_type == TaskType.REGION_ANALYSIS or "Multispectral_Analysis" in res.target_tools:
            if res.parameters.get("multispectral") or any(k in q_lower for k in ["ndvi", "ndwi", "ndbi", "spectral index", "spectral indices", "surface reflectance", "nir band"]):
                res.task_type = TaskType.REGION_ANALYSIS
                res.target_tools = ["Multispectral_Analysis"]
            else:
                res.task_type = TaskType.GROUNDING
                if "Grounding" not in res.target_tools:
                    res.target_tools.append("Grounding")
                if "Multispectral_Analysis" in res.target_tools:
                    res.target_tools.remove("Multispectral_Analysis")
        elif plan.intent_task == "UNSUPPORTED":
            res.task_type = TaskType.UNSUPPORTED
            res.is_supported = False
            res.target_tools = []
        elif plan.intent_task == "GEOSPATIAL_METADATA":
            res.task_type = TaskType.GEOSPATIAL_METADATA
            res.target_tools = []
        elif plan.intent_task == "CHANGE_ANALYSIS":
            is_compound = any(t in res.target_tools for t in ["Grounding", "VQA", "Optical_Caption"])
            res.task_type = TaskType.MULTI_TASK if (is_compound or res.task_type == TaskType.MULTI_TASK) else TaskType.CHANGE_ANALYSIS
        elif plan.intent_task == "OPTICAL_SAR_ANALYSIS":
            is_compound = any(t in res.target_tools for t in ["Grounding", "VQA", "Optical_Caption"])
            res.task_type = TaskType.MULTI_TASK if (is_compound or res.task_type == TaskType.MULTI_TASK) else TaskType.OPTICAL_SAR_ANALYSIS
        elif plan.intent_task == "COUNTING":
            res.task_type = TaskType.COUNTING
            res.parameters["is_counting"] = True
            res.parameters["count_requested"] = True
            if "Grounding" not in res.target_tools:
                res.target_tools.append("Grounding")
        elif plan.intent_task == "PRESENCE_CHECK":
            res.task_type = TaskType.PRESENCE_CHECK
            res.parameters["presence_check"] = True
            if "VQA" not in res.target_tools:
                res.target_tools.append("VQA")
            if "Grounding" not in res.target_tools:
                res.target_tools.append("Grounding")
            res.restructured_vqa_queries = [q_raw if q_raw.endswith("?") else f"{q_raw}?"]
        elif plan.intent_task == "SPATIAL_REASONING":
            res.task_type = TaskType.SPATIAL_REASONING
            res.target_tools = ["Grounding"]
        elif plan.intent_task in ("VQA", "SCENE_UNDERSTANDING") or SemanticQueryParser.is_scene_understanding(q_raw):
            res.task_type = TaskType.OPEN_ENDED_LAND_USE if plan.intent_task == "SCENE_UNDERSTANDING" else TaskType.VQA
            if "VQA" not in res.target_tools:
                res.target_tools.append("VQA")
            if "Optical_Caption" not in res.target_tools:
                res.target_tools.append("Optical_Caption")
            norm_vqa = q_raw if q_raw.endswith("?") else f"{q_raw}?"
            res.restructured_vqa_queries = [norm_vqa]
            res.parameters["question"] = norm_vqa
        elif plan.intent_task == "MULTI_TASK":
            res.task_type = TaskType.MULTI_TASK
            if "Optical_Caption" not in res.target_tools and (plan.description_request or has_optical):
                res.target_tools.append("Optical_Caption")
            if "Grounding" not in res.target_tools and (plan.target or plan.targets):
                res.target_tools.append("Grounding")
        elif plan.intent_task == "CAPTIONING":
            res.task_type = TaskType.CAPTIONING
            if "Optical_Caption" not in res.target_tools:
                res.target_tools.append("Optical_Caption")
        elif plan.intent_task == "DETECTION":
            res.task_type = TaskType.GROUNDING
            if "Grounding" not in res.target_tools:
                res.target_tools.append("Grounding")
        elif plan.intent_task == "GEOLOCATION":
            res.task_type = TaskType.GEOLOCATION
            res.target_tools = []


        target_specs = plan.target_specs
        res.target_specs = target_specs

        valid_targets = [
            ts.concept for ts in target_specs
            if ts.concept and ts.concept not in SemanticQueryParser.NON_TARGET_TOKENS and ts.concept != "objects"
        ]
        if valid_targets:
            res.parameters["all_targets"] = valid_targets

        primary_spec = target_specs[0] if target_specs else None
        if primary_spec and primary_spec.concept not in SemanticQueryParser.NON_TARGET_TOKENS and primary_spec.concept != "objects":
            res.target_geometry = primary_spec.geometry_type.value
            target_phrase = primary_spec.concept
            res.parameters["target_phrase"] = target_phrase
            res.parameters["target_geometry"] = primary_spec.geometry_type.value
            res.parameters["target_specs"] = [ts.to_dict() for ts in target_specs]
            res.parameters["spatial_relations"] = [sr.to_dict() for sr in primary_spec.spatial_relations]
        elif plan.target:
            res.target_geometry = plan.geometry
            res.parameters["target_phrase"] = plan.target
            res.parameters["target_geometry"] = plan.geometry
        elif plan.intent_task == "SPATIAL_REASONING" and ("objects" in q_lower or "object" in q_lower):
            res.target_geometry = "DISCRETE_OBJECT"
            res.parameters["target_phrase"] = "objects"
            res.parameters["target_geometry"] = "DISCRETE_OBJECT"
        else:
            res.target_geometry = None
            res.parameters["target_phrase"] = None
            res.parameters["target_geometry"] = None
            res.parameters["target_specs"] = []
            res.parameters["spatial_relations"] = []

        # Enforce action word blacklist on target_phrase (pure verbs and meta tokens, not physical nouns)
        action_forbidden = {
            "describe", "find", "locate", "count", "compare", "analyze", "show",
            "identify", "detect", "pinpoint", "segment", "box", "highlight",
            "outline", "what", "where", "how", "tell", "summarize", "image",
            "scene", "photo", "picture", "satellite", "presence"
        }
        if res.parameters.get("target_phrase") and res.parameters["target_phrase"].strip().lower() in action_forbidden:
            res.parameters["target_phrase"] = None
            res.target_geometry = None

        # For scene-understanding and pure VQA queries, suppress false object targets
        if SemanticQueryParser.is_scene_understanding(q_raw) or plan.intent_task in ("VQA", "SCENE_UNDERSTANDING"):
            res.parameters["target_phrase"] = None
            res.target_geometry = None
            res.localization_required = False

        if res.task_type in (TaskType.CAPTIONING, TaskType.UNSUPPORTED, TaskType.GEOSPATIAL_METADATA, TaskType.CHANGE_ANALYSIS, TaskType.OPTICAL_SAR_ANALYSIS) and not plan.target:
            res.target_geometry = None
            res.parameters["target_phrase"] = None
            res.parameters["target_geometry"] = None
            res.localization_required = False
        elif res.task_type in (TaskType.GROUNDING, TaskType.COUNTING, TaskType.PRESENCE_CHECK, TaskType.SPATIAL_REASONING, TaskType.MULTI_TASK):
            res.localization_required = True
        # Safety Guard: Autonomous Zero-Shot Object Queries MUST ALWAYS route to GROUNDING or MULTI_TASK, NEVER to VQA!
        detection_starters = ("find", "locate", "detect", "identify", "show", "count", "where", "spot", "pinpoint", "segment", "outline", "box", "highlight")
        filler_re = r"\b(find|locate|detect|identify|show|me|where|is|are|the|a|an|all|any|some|in|on|at|across|this|that|these|those|image|scene|photo|picture|satellite|imagery|view|overhead|raster|count|how many|search|for|spot|pinpoint|box|segment|outline|highlight|can you see|do you see|tell me if there is|tell me if there are)\b"
        stripped_c = re.sub(filler_re, " ", q_lower)
        has_extracted_cand = any(len(tok.strip()) >= 2 and tok.strip() not in ("objects", "object", "here", "there", "it", "them") for tok in re.split(r"[,;&]|\b(?:and|or)\b", stripped_c))
        has_detection_intent = any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in detection_starters) or (has_extracted_cand and not SemanticQueryParser.is_scene_description(q_raw))
        if has_detection_intent:
            if res.task_type == TaskType.VQA:
                res.task_type = TaskType.GROUNDING
            if "VQA" in res.target_tools:
                res.target_tools.remove("VQA")
            if "Grounding" not in res.target_tools:
                res.target_tools.append("Grounding")
            res.localization_required = True

        res.parameters["localization_required"] = res.localization_required
        return res


    def _classify_core(
        self,
        query: str = "",
        has_optical: bool = False,
        has_sar: bool = False,
        has_change_pair: bool = False,
        probe_features: Optional[List[str]] = None,
    ) -> ClassificationResult:
        q_raw = (query or "").strip()
        from target_spec import extract_safety_constraints
        clean_q, constraints = extract_safety_constraints(q_raw)
        q_lower = clean_q.lower() if clean_q else q_raw.lower()

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
                        parameters={"raw_query": q_raw, "constraints": constraints},
                        target_tools=[],
                        query_family="UNSUPPORTED",
                        reasoning=f"Query '{q_raw}' is unrelated to geospatial imagery or remote-sensing analysis.",
                        is_supported=False,
                    )

        # ---------------------------------------------------------
        # 2. Geospatial Metadata Queries (CRS, Resolution, GSD, Coords, Geolocation Origin)
        # ---------------------------------------------------------
        is_geo_origin = bool(re.search(r"\b(where is this (?:imagery|image|scene|photo) from|where is this|what location|where was this taken|location of this (?:image|scene|satellite|raster))\b", q_lower))
        if is_geo_origin:
            return ClassificationResult(
                task_type=TaskType.GEOLOCATION,
                confidence=0.93,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "constraints": constraints},
                target_tools=[],
                query_family="GEOLOCATION",
                reasoning="Geographic location / origin inquiry detected.",
                is_supported=True,
            )

        has_visual_verbs = bool(re.search(
            r"\b(identify|detect|locate|find|segment|outline|count|spot|search for|analyze|extract|map|box|highlight)\b",
            q_lower
        )) or (
            bool(re.search(r"\b(show|show me|pinpoint)\b", q_lower))
            and not bool(re.search(r"\b(?:show|show me)\s+(?:the\s+)?(?:coordinates|crs|resolution|ground sampling distance|gsd|projection)\b", q_lower))
        )
        is_explicit_geo_query = bool(re.search(
            r"\b(what is the (?:crs|projection|resolution|gsd|coordinate system)|show me the (?:crs|coordinates|gsd|resolution)|what are the coordinates|what coordinate reference system|coordinate reference system|ground sampling distance)\b",
            q_lower
        ))
        is_general_geo_kw = bool(re.search(r"\b(coordinates|crs|resolution|ground sampling distance|gsd|projection|lat.*lon|bounding box coordinates)\b", q_lower))

        if (is_explicit_geo_query or is_general_geo_kw) and not has_visual_verbs and not any(re.search(p, q_lower) for p in [r"\bchange\b", r"\bdifference\b", r"\bdescribe\b"]):
            return ClassificationResult(
                task_type=TaskType.GEOSPATIAL_METADATA,
                confidence=0.92,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "constraints": constraints},
                target_tools=[],
                query_family="GEOSPATIAL_METADATA",
                reasoning="Geospatial metadata inquiry detected (CRS/Resolution/Coordinates/GSD).",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 2b. OpenStreetMap / Overpass Feature Queries
        # ---------------------------------------------------------
        is_osm_query = bool(re.search(
            r"\b(openstreetmap|osm|overpass|mapped buildings?|mapped roads?|mapped bridges?|road network|highway network|waterway network|cartographic features?|vector layer)\b",
            q_lower,
        ))
        if is_osm_query:
            tools = ["OpenStreetMap_Feature_Query"]
            if any(w in q_lower for w in ["detect", "locate", "find", "compare"]):
                tools.append("Grounding")
            return ClassificationResult(
                task_type=TaskType.OSM_FEATURE_QUERY if len(tools) == 1 else TaskType.MULTI_TASK,
                confidence=0.94,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "osm_requested": True},
                target_tools=tools,
                query_family="OPENSTREETMAP_QUERY",
                reasoning="OpenStreetMap / Overpass vector cartographic feature query detected.",
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
        # 4b. Multispectral / Spectral Index Intent (NDVI, NDWI, NDBI)
        # ---------------------------------------------------------
        is_spectral_query = bool(re.search(
            r"\b(ndvi|ndwi|ndbi|spectral index|spectral indices|vegetation index|water index|built-up index|multispectral|surface reflectance|nir band|swir band)\b",
            q_lower,
        ))
        if is_spectral_query:
            tools = ["Multispectral_Analysis"]
            indices = []
            if "ndvi" in q_lower or "vegetation" in q_lower:
                indices.append("NDVI")
            if "ndwi" in q_lower or "water" in q_lower:
                indices.append("NDWI")
            if "ndbi" in q_lower or "built-up" in q_lower or "urban" in q_lower:
                indices.append("NDBI")
            if not indices:
                indices = ["NDVI", "NDWI", "NDBI"]

            return ClassificationResult(
                task_type=TaskType.REGION_ANALYSIS,
                confidence=0.92,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "indices": indices, "multispectral": True},
                target_tools=tools,
                query_family="MULTISPECTRAL_ANALYSIS",
                reasoning="Multispectral index calculation requested -> routed to Multispectral_Analysis specialist tool.",
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
            r"\b(what changed|change detection|detect changes|find changes|changes between|changed regions?|scene changed|what areas changed|where has the scene changed|difference between|compare (?:these\s+)?(?:two\s+)?(?:images?|scenes?|photos?|the two)|compare|before and after|deforestation|what was built|what was destroyed|changed over time|biggest change|is the change|which areas changed|what disappeared|what was added)\b",
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
            is_compound = any(t in tools for t in ["Grounding", "VQA", "Optical_Caption"])
            return ClassificationResult(
                task_type=TaskType.MULTI_TASK if is_compound else TaskType.CHANGE_ANALYSIS,
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
        # 8. Multi-Intent Combined Queries (e.g. "Describe image and locate buildings")
        # ---------------------------------------------------------
        has_caption_intent = bool(re.search(
            r"\b(describe|description|overview|summarize|what is in this image|what is in the image|what is in this scene|what do you see|what you see|tell me what you see|what does this image contain|what does the image contain|what is happening)\b",
            q_lower,
        )) or SemanticQueryParser.is_scene_description(q_raw)
        has_grounding_intent = bool(re.search(
            r"\b(locate|find|detect|pinpoint|show me|where is|where are|bounding box|outline|segment)\b",
            q_lower,
        ))
        has_vqa_intent = bool(re.search(
            r"\b(is there\b|are there\b|is it\b|does it\b|tell me if\b|what color\b|what type of area\b|what is the dominant\b|is\s+[a-zA-Z\s]+\s+present\b|are\s+[a-zA-Z\s]+\s+present\b|is\s+[a-zA-Z\s]+\s+visible\b|are\s+[a-zA-Z\s]+\s+visible\b)",
            q_lower,
        ))

        if sum([has_caption_intent, has_grounding_intent, has_vqa_intent]) >= 2:
            tools = []
            if has_caption_intent:
                tools.append("Optical_Caption")
            if has_grounding_intent:
                tools.append("Grounding")
            if has_vqa_intent:
                tools.append("VQA")
            target_phrase = extract_grounding_target(q_raw)
            return ClassificationResult(
                task_type=TaskType.MULTI_TASK,
                confidence=0.90,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase},
                target_tools=tools,
                restructured_vqa_queries=[q_raw] if "VQA" in tools else [],
                query_family="MULTI_TASK",
                reasoning=f"Multi-intent query combining {tools} -> scheduled multi-specialist execution.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 9. Counting Queries ("How many X", "Count the X")
        # ---------------------------------------------------------
        is_counting = bool(re.search(r"\b(how many|count the|count|number of|quantity of|total count)\b", q_lower))
        if is_counting:
            target_phrase = extract_grounding_target(q_raw)
            tools = ["Grounding"]
            vqa_q = f"How many {target_phrase} are present?"
            return ClassificationResult(
                task_type=TaskType.COUNTING,
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
        # 10. Spatial / Location Queries ("Where are...", "Where is...", "Which side...")
        # ---------------------------------------------------------
        is_spatial = bool(re.search(r"\b(where is|where are|which side|which region|locate the|pinpoint|spatial distribution)\b", q_lower))
        if is_spatial:
            target_phrase = extract_grounding_target(q_raw)
            tools = ["Grounding"]
            return ClassificationResult(
                task_type=TaskType.GROUNDING,
                confidence=0.89,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase, "spatial_inquiry": True},
                target_tools=tools,
                query_family="SPATIAL_LOCALIZATION",
                reasoning=f"Spatial location inquiry for '{target_phrase}' -> routed to Grounding DINO spatial localization.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 11. Explicit Object Localization & Grounding ("Locate...", "Find...", "Detect...", "Show me...")
        # ---------------------------------------------------------
        is_grounding_explicit = bool(re.search(
            r"\b(locate|find|detect|identify|pinpoint|box|show\s+me|show|bounding box|box for|highlight|outline)\b",
            q_lower,
        ))

        if is_grounding_explicit:
            target_phrase = extract_grounding_target(q_raw)
            return ClassificationResult(
                task_type=TaskType.GROUNDING,
                confidence=0.88,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase},
                target_tools=["Grounding"],
                query_family="OBJECT_IDENTIFICATION",
                reasoning=f"Object localization intent for '{target_phrase}' -> routed to Grounding DINO bounding box proposal extractor.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 12. General Scene Understanding / Captioning
        # ---------------------------------------------------------
        is_scene_query = bool(re.search(
            r"\b(describe|description of|overview|summarize|what is in this image|what is in the image|what is in this scene|what is in the scene|what is visible|what do you see|what you see|tell me what you see|what does this image contain|what does the image contain|what does this scene contain|what does this image show|what can you see|what is happening|tell me about this area)\b",
            q_lower,
        )) or SemanticQueryParser.is_scene_description(q_raw) or not q_raw
        if is_scene_query and not has_grounding_intent:
            return ClassificationResult(
                task_type=TaskType.CAPTIONING,
                confidence=0.90 if q_raw else 0.70,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": None},
                target_tools=["Optical_Caption"],
                query_family="SCENE_UNDERSTANDING",
                reasoning="General scene understanding request -> routed to BLIP optical scene captioner.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 13. Direct Yes/No, Presence & Attribute VQA Queries
        # ---------------------------------------------------------
        detection_starters = ("find", "locate", "detect", "identify", "show", "count", "where", "spot", "pinpoint", "segment", "outline", "box", "highlight")
        filler_re = r"\b(find|locate|detect|identify|show|me|where|is|are|the|a|an|all|any|some|in|on|at|across|this|that|these|those|image|scene|photo|picture|satellite|imagery|view|overhead|raster|count|how many|search|for|spot|pinpoint|box|segment|outline|highlight|can you see|do you see|tell me if there is|tell me if there are)\b"
        stripped_c = re.sub(filler_re, " ", q_lower)
        has_extracted_cand = any(len(tok.strip()) >= 2 and tok.strip() not in ("objects", "object", "here", "there", "it", "them") for tok in re.split(r"[,;&]|\b(?:and|or)\b", stripped_c))
        has_detection_intent = any(re.search(rf"\b{re.escape(k)}\b", q_lower) for k in detection_starters) or (has_extracted_cand and not SemanticQueryParser.is_scene_description(q_raw))
        is_vqa_query = (
            bool(re.search(
                r"\b(is there|are there|is .* present|are .* present|is .* visible|are .* visible|what type of|what is the dominant|what kind of|what color|tell me if|any .* present|any .* visible)\b",
                q_lower,
            ))
            or (q_raw.endswith("?") and not has_detection_intent)
        ) and not is_scene_query and not has_detection_intent
        if is_vqa_query:
            target_phrase = extract_grounding_target(q_raw)
            norm_vqa = q_raw if q_raw.endswith("?") else f"{q_raw}?"
            return ClassificationResult(
                task_type=TaskType.VQA,
                confidence=0.91,
                confidence_type="heuristic",
                confidence_source="semantic_intent_router",
                parameters={"query": q_raw, "target_phrase": target_phrase, "question": norm_vqa},
                target_tools=["VQA"],
                restructured_vqa_queries=[norm_vqa],
                query_family="VQA",
                reasoning=f"Direct visual question answering for '{q_raw}' -> routed directly to VQA specialist.",
                is_supported=True,
            )

        # ---------------------------------------------------------
        # 14. Default Open-Ended Fallback (Safe & Cooperative)
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
