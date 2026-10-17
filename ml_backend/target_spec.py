"""
target_spec.py
--------------
Generalized Target Representation & Semantic Analysis Layer for Open-Vocabulary
Remote Sensing Intelligence.

Implements:
- TargetSpec: Rich semantic specification of arbitrary remote-sensing concepts.
- GeometryType: DISCRETE_OBJECT, LINEAR_FEATURE, REGION_FEATURE, COMPLEX_CONCEPT.
- ScaleCategory: TINY, SMALL, MEDIUM, LARGE, MACRO.
- SpatialRelation: Geometric spatial relation predicates and reference targets.
- SemanticQueryParser: Open-vocabulary natural language parser that decomposes
  arbitrary queries into structured TargetSpec entities with explicit head-noun resolution.
"""

from __future__ import annotations
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


class GeometryType(str, Enum):
    DISCRETE_OBJECT = "DISCRETE_OBJECT"
    LINEAR_FEATURE = "LINEAR_FEATURE"
    REGION_FEATURE = "REGION_FEATURE"
    COMPLEX_CONCEPT = "COMPLEX_CONCEPT"


class ScaleCategory(str, Enum):
    TINY = "tiny"        # < 15 px (e.g. small vehicles, individual solar panels)
    SMALL = "small"      # 15 - 50 px (e.g. cars, small boats, small buildings, silos)
    MEDIUM = "medium"    # 50 - 150 px (e.g. warehouses, cargo ships, aircraft, locomotives)
    LARGE = "large"      # 150 - 400 px (e.g. industrial complexes, ports, runways, quarries)
    MACRO = "macro"      # > 400 px / landscape (e.g. water bodies, forests, agriculture)


@dataclass
class SpatialRelation:
    relation: str  # "near", "inside", "outside", "around", "beside", "along", "crossing", "adjacent", "north_of", "south_of", "east_of", "west_of", "closest_to"
    reference_entity: str
    distance_threshold: Optional[float] = None  # Normalized distance in [0..1000] space
    raw_phrase: str = ""

    @property
    def predicate(self) -> str:
        return self.relation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation": self.relation,
            "predicate": self.relation,
            "reference_entity": self.reference_entity,
            "distance_threshold": self.distance_threshold,
            "raw_phrase": self.raw_phrase,
        }


@dataclass
class TargetSpec:
    concept: str
    aliases: List[str] = field(default_factory=list)
    semantic_description: str = ""
    geometry_type: GeometryType = GeometryType.DISCRETE_OBJECT
    expected_scale: ScaleCategory = ScaleCategory.MEDIUM
    expected_shape: str = "compact_box"  # "compact_box", "elongated_line", "amorphous_polygon", "patch", "composite"
    visual_attributes: Dict[str, Any] = field(default_factory=dict)
    spectral_attributes: Dict[str, Any] = field(default_factory=dict)
    contextual_attributes: List[str] = field(default_factory=list)
    spatial_relations: List[SpatialRelation] = field(default_factory=list)
    temporal_requirements: Optional[str] = None  # e.g., "new", "demolished", "flooded", "vegetation_loss"
    count_required: bool = False
    localization_required: bool = True
    grounding_prompt: str = ""
    confidence_threshold: float = 0.20

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept": self.concept,
            "aliases": self.aliases,
            "semantic_description": self.semantic_description,
            "geometry_type": self.geometry_type.value,
            "expected_scale": self.expected_scale.value,
            "expected_shape": self.expected_shape,
            "visual_attributes": self.visual_attributes,
            "spectral_attributes": self.spectral_attributes,
            "contextual_attributes": self.contextual_attributes,
            "spatial_relations": [sr.to_dict() for sr in self.spatial_relations],
            "temporal_requirements": self.temporal_requirements,
            "count_required": self.count_required,
            "localization_required": self.localization_required,
            "grounding_prompt": self.grounding_prompt or self.concept,
            "confidence_threshold": self.confidence_threshold,
        }


@dataclass
class SemanticPlan:
    query: str
    intent_task: str  # CAPTIONING, DETECTION, COUNTING, REGION_ANALYSIS, LINEAR_FEATURE_ANALYSIS, CHANGE_ANALYSIS, SPATIAL_REASONING, COMPARISON, PRESENCE_CHECK, GEOLOCATION, MULTI_TASK, UNSUPPORTED
    tasks: List[str] = field(default_factory=list)
    target: Optional[str] = None
    targets: List[str] = field(default_factory=list)
    location: Optional[str] = None
    time: Optional[str] = None
    spatial_relation: Optional[SpatialRelation] = None
    spatial_relations: List[SpatialRelation] = field(default_factory=list)
    geometry: Optional[str] = None
    count_request: bool = False
    comparison_request: bool = False
    description_request: bool = False
    change_request: bool = False
    presence_check: bool = False
    confidence_requirement: float = 0.20
    target_specs: List[TargetSpec] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    detection_status: str = "model_ready"  # GENUINE_ZERO, FILTERED_ZERO, LOCALIZATION_UNAVAILABLE, MODEL_UNAVAILABLE, NOT_APPLICABLE, MODEL_READY
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent_task": self.intent_task,
            "tasks": self.tasks,
            "target": self.target,
            "targets": self.targets,
            "location": self.location,
            "time": self.time,
            "spatial_relation": self.spatial_relation.to_dict() if self.spatial_relation else None,
            "spatial_relations": [sr.to_dict() for sr in self.spatial_relations],
            "geometry": self.geometry,
            "count_request": self.count_request,
            "comparison_request": self.comparison_request,
            "description_request": self.description_request,
            "change_request": self.change_request,
            "presence_check": self.presence_check,
            "confidence_requirement": self.confidence_requirement,
            "target_specs": [ts.to_dict() for ts in self.target_specs],
            "limitations": self.limitations,
            "detection_status": self.detection_status,
            "constraints": self.constraints,
        }


# ---------------------------------------------------------------------------
# Semantic Concept Knowledge & Priors
# ---------------------------------------------------------------------------

DISCRETE_NOUNS = {
    "locomotive", "locomotives", "train", "trains", "car", "cars", "truck", "trucks",
    "vehicle", "vehicles", "motor vehicle", "motor vehicles", "bus", "buses", "automobile", "automobiles",
    "ship", "ships", "boat", "boats", "vessel", "vessels", "cargo ship", "cargo ships",
    "airplane", "airplanes", "aircraft", "plane", "planes", "helicopter", "helicopters",
    "building", "buildings", "house", "houses", "structure", "structures", "warehouse", "warehouses", "hangar", "hangars",
    "tank", "tanks", "storage tank", "storage tanks", "fuel tank", "fuel tanks", "oil tank", "oil tanks", "silo", "silos",
    "container", "containers", "freight container", "freight containers", "shipping container", "shipping containers",
    "tower", "towers", "antenna", "antennas", "radio mast", "radio masts", "crane", "cranes", "chimney", "chimneys",
    "cooling tower", "cooling towers", "wind turbine", "wind turbines", "solar panel", "solar panels", "greenhouse", "greenhouses",
    "terminal", "terminals", "cargo terminal", "cargo terminals", "container terminal", "container terminals"
}

LINEAR_NOUNS = {
    "road", "roads", "highway", "highways", "street", "streets", "avenue", "freeway", "lane",
    "railway", "railways", "railroad", "railroads", "train track", "train tracks", "track", "tracks",
    "runway", "runways", "airport runway", "airport runways", "tarmac", "airstrip", "taxiway", "taxiways",
    "pipeline", "pipelines", "oil pipe", "gas line",
    "power line", "power lines", "transmission line", "transmission lines",
    "canal", "canals", "waterway", "waterways", "aqueduct",
    "bridge", "bridges", "viaduct", "overpass", "causeway",
    "coastline", "shoreline", "river bank",
    "railway infrastructure", "railroad infrastructure", "road infrastructure"
}

REGION_NOUNS = {
    "water", "water body", "water bodies", "body of water", "bodies of water", "water region", "water regions", "water area", "water areas",
    "ocean", "sea", "lake", "lakes", "river", "rivers", "stream", "bay", "harbor", "port", "reservoir", "pond",
    "vegetation", "forest", "forests", "woodland", "trees", "greenery", "canopy", "mangrove", "jungle", "patch", "patches", "forest patch", "forest patches", "vegetation patch", "vegetation patches", "vegetation region", "vegetation regions", "region", "regions",
    "grassland", "grasslands", "grassy area", "grassy areas", "pasture", "meadow", "lawn",
    "farmland", "farmlands", "cropland", "agriculture", "agricultural field", "agricultural fields", "cultivated land", "paddy",
    "agricultural area", "agricultural areas", "agricultural land", "agricultural zone", "agricultural facility", "agricultural facilities",
    "wetland", "wetlands", "marsh", "swamp", "bog",
    "bare soil", "bare ground", "sand", "desert", "dunes", "beach",
    "snow", "ice", "glacier",
    "construction area", "construction site", "construction sites", "construction zone", "construction zones", "construction", "construction activity", "excavation", "earthworks",
    "industrial area", "industrial areas", "industrial zone", "industrial zones", "industrial facility", "industrial facilities", "industrial",
    "urban area", "dense settlement", "residential area", "residential neighborhood", "commercial district",
    "mining area", "mining areas", "quarry", "quarries", "open-pit mine", "mining",
    "parking area", "parking areas", "parking lot", "parking lots",
    "shipyard", "shipyards", "drydock", "drydocks", "dockyard", "dockyards",
    "solar farm", "solar farms", "wind farm", "wind farms", "solar array", "photovoltaic farm",
    "railway yard", "railway yards", "train yard", "train yards", "marshaling yard"
}


SCALE_PRIORS: Dict[str, ScaleCategory] = {
    "car": ScaleCategory.TINY,
    "truck": ScaleCategory.TINY,
    "vehicle": ScaleCategory.TINY,
    "bus": ScaleCategory.TINY,
    "boat": ScaleCategory.SMALL,
    "solar panel": ScaleCategory.TINY,
    "antenna": ScaleCategory.TINY,
    "tower": ScaleCategory.SMALL,
    "house": ScaleCategory.SMALL,
    "silo": ScaleCategory.SMALL,
    "storage tank": ScaleCategory.SMALL,
    "tank": ScaleCategory.SMALL,
    "crane": ScaleCategory.SMALL,
    "wind turbine": ScaleCategory.SMALL,
    "greenhouse": ScaleCategory.SMALL,
    "container": ScaleCategory.TINY,
    "building": ScaleCategory.MEDIUM,
    "warehouse": ScaleCategory.MEDIUM,
    "hangar": ScaleCategory.MEDIUM,
    "ship": ScaleCategory.MEDIUM,
    "vessel": ScaleCategory.MEDIUM,
    "aircraft": ScaleCategory.MEDIUM,
    "airplane": ScaleCategory.MEDIUM,
    "helicopter": ScaleCategory.SMALL,
    "train": ScaleCategory.MEDIUM,
    "locomotive": ScaleCategory.MEDIUM,
    "bridge": ScaleCategory.LARGE,
    "runway": ScaleCategory.LARGE,
    "port": ScaleCategory.LARGE,
    "airport": ScaleCategory.LARGE,
    "quarry": ScaleCategory.LARGE,
    "parking area": ScaleCategory.LARGE,
    "parking lot": ScaleCategory.LARGE,
    "water": ScaleCategory.MACRO,
    "forest": ScaleCategory.MACRO,
    "farmland": ScaleCategory.MACRO,
    "grassland": ScaleCategory.MACRO,
}


# ---------------------------------------------------------------------------
# Semantic Query Parser
# ---------------------------------------------------------------------------

class SemanticQueryParser:
    """
    Open-vocabulary parser converting natural language remote sensing questions
    into structured TargetSpec entities with spatial, temporal, and geometric attributes.
    """

    RELATION_REGEX = re.compile(
        r"\b(near|close to|adjacent to|beside|around|inside|outside|along|crossing|across|between|surrounding|bordering|flanking|north of|south of|east of|west of|closest to)\s+([a-zA-Z\s]+?)(?:\s+(?:and|or|in|on|with|\?|$)|$)",
        re.IGNORECASE,
    )

    TEMPORAL_REGEX = re.compile(
        r"\b(new|newly constructed|newly developed|demolished|destroyed|removed|expanded|flooded|vegetation loss|deforestation|change in|changed)\b",
        re.IGNORECASE,
    )

    COUNT_REGEX = re.compile(
        r"\b(how many|count|number of|quantity of|total of)\b",
        re.IGNORECASE,
    )

    LOCALIZATION_REGEX = re.compile(
        r"\b(locate|find|detect|where|pinpoint|segment|box|show|highlight|identify|outline)\b",
        re.IGNORECASE,
    )

    SCENE_DESCRIPTION_PATTERNS = [
        r"^(?:can you\s+|could you\s+|please\s+)?(?:describe|give(?:\s+me)?(?:\s+a)?|provide(?:\s+an?)?)\s+(?:brief\s+|detailed\s+|full\s+)?(?:description|overview|summary)(?:\s+of)?\s*(?:the|this|that|a|an)?\s*(?:satellite\s+)?(?:image|scene|photo|area|surroundings|landscape|raster|view)?\s*$",
        r"^(?:can you\s+|could you\s+|please\s+)?(?:tell\s+me\s+)?what\s+(?:all\s+)?(?:is\s+in|is\s+visible\s+in|does\s+.*?contain|does\s+.*?show|do\s+you\s+see(?:\s+in)?|can\s+you\s+see(?:\s+in)?)\s*(?:the|this|that|a|an)?\s*(?:satellite\s+)?(?:image|scene|photo|area)?\s*$",
        r"^(?:can you\s+|could you\s+|please\s+)?tell\s+me\s+what\s+you\s+see\s*$",
        r"^(?:what\s+you\s+see|what\s+do\s+you\s+see|what\s+can\s+you\s+see)\s*$",
        r"^(?:can you\s+|could you\s+|please\s+)?summarize\s*(?:the|this|that|a|an)?\s*(?:satellite\s+)?(?:image|scene|photo|area|surroundings)?\s*$",
        r"^(?:scene|image|satellite\s+image)\s+description\s*$",
        r"^(?:overview|summary)\s+of\s+(?:the\s+|this\s+)?(?:image|scene|area)\s*$",
        r"^(?:can you\s+|could you\s+|please\s+)?describe\s+.*(?:in|of)?\s*(?:the|this|that)?\s*(?:satellite\s+)?(?:image|scene|photo|area)?\s*$",
    ]

    NON_TARGET_TOKENS = {
        "it", "them", "all", "image", "images", "scene", "scenes", "photo", "photos",
        "satellite", "satellite image", "satellite images", "satellite imagery",
        "describe", "description", "describe image", "describe this image", "describe the image",
        "describe the scene", "describe scene", "summarize", "summary", "overview",
        "objects", "something", "anything", "everything", "what you see", "what is",
        "what is in this image", "what does this image contain", "what do you see",
        "provide", "provide an overview", "overview of the area", "provide an overview of the area",
        "area", "surroundings", "landscape", "view", "presence",
        "what type area shown", "type of area", "type area", "what type", "what kind", "what kind of area",
        "what type of area", "what type of terrain", "what kind of terrain", "land use", "dominant land use"
    }

    SCENE_UNDERSTANDING_PATTERNS = [
        r".*\b(?:what type of area|what kind of area|what type of terrain|what kind of terrain|what land use|dominant land use|what is this area|what type area|type of area is shown|what kind of place|what area is shown|what type of environment)\b.*",
    ]

    SAFETY_CONSTRAINT_PATTERNS = [
        re.compile(r"\b(?:do not|don't|never|avoid|without)\s+(?:invent|fabricate|hallucinate|guess|assume|extrapolate|generate unverified)\s+[^.!?,\n]+", re.IGNORECASE),
        re.compile(r"\b(?:do not|don't|never|avoid)\s+(?:report|output|provide|use)\s+(?:unverified|fabricated|false|fake|invented)\s+[^.!?,\n]+", re.IGNORECASE),
        re.compile(r"\bonly\s+(?:report|use|output|provide)\s+(?:verified|ground\s*truth|actual|genuine)\s+[^.!?,\n]+", re.IGNORECASE),
    ]

    VERB_PREFIX_PATTERN = re.compile(
        r"^(?:(?:please\s+)?(?:can you\s+|could you\s+|i want to\s+|tell me to\s+)?"
        r"(?:identify and analyze|detect and analyze|locate and analyze|find and analyze|search and find|search for|segment and analyze|classify and locate|"
        r"total quantity of|total count of|quantity of|number of|how many|count the|count|"
        r"is there|are there|can you see|do you see|tell me if there is|tell me if there are|"
        r"see|locate|find|detect|show|pinpoint|identify|outline|segment|box|where\s+are|where\s+is|highlight|analyze|extract|map|inspect)\s+"
        r"(?:any\s+|all\s+|the\s+|a\s+|an\s+)?)+",
        re.IGNORECASE,
    )

    @classmethod
    def extract_safety_constraints(cls, query: str) -> Tuple[str, List[str]]:
        """
        Separates user safety/accuracy constraints (e.g. 'Do not invent coordinates or sensor metadata')
        from the requested visual intelligence / analytical task.
        Returns (cleaned_query_without_constraints, list_of_extracted_constraints).
        """
        q = (query or "").strip()
        if not q:
            return "", []
        extracted: List[str] = []
        cleaned = q
        for pat in cls.SAFETY_CONSTRAINT_PATTERNS:
            for m in pat.finditer(q):
                phrase = m.group(0).strip().rstrip(".!?,")
                if phrase:
                    extracted.append(phrase)
                    cleaned = re.sub(re.escape(m.group(0)), "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"^[.,;!?-]+\s*", "", cleaned).strip()
        cleaned = re.sub(r"\s*[.,;!?-]+$", "", cleaned).strip()
        return cleaned, extracted

    @classmethod
    def is_scene_description(cls, phrase: str) -> bool:
        """Determines whether a phrase or clause represents a holistic scene description intent."""
        p = (phrase or "").strip().lower()
        if not p:
            return False
        p = re.sub(r"[?!.,;]+$", "", p).strip()
        for pat in cls.SCENE_DESCRIPTION_PATTERNS:
            if re.match(pat, p, re.IGNORECASE):
                return True
        return False

    @classmethod
    def is_scene_understanding(cls, phrase: str) -> bool:
        """Determines whether a query is asking for holistic scene classification / land use type."""
        p = (phrase or "").strip().lower()
        if not p:
            return False
        for pat in cls.SCENE_UNDERSTANDING_PATTERNS:
            if re.search(pat, p, re.IGNORECASE):
                return True
        return False

    @classmethod
    def parse(cls, query: str) -> List[TargetSpec]:
        """Parses a natural language query into one or more TargetSpec representations."""
        q_raw = (query or "").strip()
        if not q_raw:
            return []

        # 0. Separate safety/negative constraints before target parsing
        clean_query, _ = cls.extract_safety_constraints(q_raw)
        q_lower = clean_query.lower() if clean_query else q_raw.lower()

        # Clean conversational prefixes
        clean_q = re.sub(r"^(?:can you|could you|please|i want to|tell me|show me|provide|please provide)\s+", "", q_lower).strip()

        # 0b. Holistic scene understanding / classification query check (e.g. 'what type of area is shown?')
        if cls.is_scene_understanding(q_lower):
            return []

        # 1. Pure scene description check: If entire query is scene description without spatial request, return empty targets
        has_spatial_loc = bool(cls.LOCALIZATION_REGEX.search(q_lower))
        has_counting = bool(cls.COUNT_REGEX.search(q_lower))
        if (
            (cls.is_scene_description(clean_q) or cls.is_scene_description(q_lower) or q_lower.startswith(("describe", "summarize", "overview", "provide an overview", "give me a description", "give a description")))
            and not has_spatial_loc
            and not has_counting
        ):
            return []

        # 2. Detect Count & Localization Intents
        count_required = has_counting
        localization_required = has_spatial_loc or not count_required

        # 3. Detect Temporal/Change Requirements
        temporal_match = cls.TEMPORAL_REGEX.search(q_lower)
        temporal_req = temporal_match.group(1).lower() if temporal_match else None

        # Clean trailing conditional clauses
        clean_q = re.sub(r"\bif\s+(?:yes|so|present|visible|any|there\s+is|there\s+are)\b.*$", "", clean_q).strip()
        clean_q = re.sub(r"\bthen\s+(?:locate|find|detect|pinpoint|box|segment|show)\s+(?:them|it|all)\b.*$", "", clean_q).strip()

        # 4. Extract Spatial Relations
        spatial_relations = []
        for rel_m in cls.RELATION_REGEX.finditer(clean_q):
            rel_type = rel_m.group(1).strip().lower().replace(" ", "_")
            ref_entity = rel_m.group(2).strip().lower()
            # Clean stopwords
            ref_entity = re.sub(r"\b(the|a|an|this|that|image|scene|photo|area|region)\b", "", ref_entity).strip()
            if ref_entity and ref_entity not in {"it", "them", "all"}:
                spatial_relations.append(
                    SpatialRelation(
                        relation=rel_type,
                        reference_entity=ref_entity,
                        raw_phrase=rel_m.group(0),
                    )
                )

        # 5. Split Multi-Target Expressions
        prep_split = re.split(
            r"\b(?:on|in|near|beside|around|across|inside|along|at|adjacent to|close to|by|surrounding|bordering|flanking|over|under|crossing|for|with|north of|south of|east of|west of|closest to)\b",
            clean_q,
        )

        primary_clause = prep_split[0].strip() if prep_split else clean_q

        # Strip lead-in verbs and quantity expressions using generalized verb prefix pattern
        primary_clause = cls.VERB_PREFIX_PATTERN.sub("", primary_clause).strip()
        primary_clause = re.sub(r"^(?:and|or)\s+", "", primary_clause).strip()

        # Check for conjunctions (and, commas, then)
        target_phrases = []
        if "," in primary_clause or " and " in primary_clause or " & " in primary_clause or " then " in primary_clause:
            raw_parts = re.split(r"[,&]|\band\b|\bthen\b", primary_clause)
            for part in raw_parts:
                part_clean = part.strip()
                if cls.is_scene_description(part_clean):
                    continue
                p_clean = cls._clean_target_token(part_clean)
                if p_clean and p_clean not in cls.NON_TARGET_TOKENS and not cls.is_scene_description(p_clean):
                    target_phrases.append(p_clean)
        else:
            if not cls.is_scene_description(primary_clause):
                p_clean = cls._clean_target_token(primary_clause)
                if p_clean and p_clean not in cls.NON_TARGET_TOKENS and not cls.is_scene_description(p_clean):
                    target_phrases.append(p_clean)

        if not target_phrases:
            if localization_required and not cls.is_scene_description(clean_q):
                target_phrases = ["objects"]
            else:
                return []

        # 6. Construct TargetSpec for each extracted target
        specs: List[TargetSpec] = []
        for phrase in target_phrases:
            geom = cls._determine_geometry(phrase)
            scale = cls._determine_scale(phrase, geom)
            shape = cls._determine_shape(geom)
            grounding_prompt = cls._build_open_vocab_prompt(phrase)

            spec = TargetSpec(
                concept=phrase,
                aliases=[phrase],
                semantic_description=f"Remote sensing concept: {phrase}",
                geometry_type=geom,
                expected_scale=scale,
                expected_shape=shape,
                spatial_relations=spatial_relations,
                temporal_requirements=temporal_req,
                count_required=count_required,
                localization_required=localization_required,
                grounding_prompt=grounding_prompt,
            )
            specs.append(spec)

        return specs

    @classmethod
    def _clean_target_token(cls, token: str) -> str:
        t = re.sub(r"[?!.,:;\"']", "", token.strip().lower()).strip()
        t = cls.VERB_PREFIX_PATTERN.sub("", t).strip()
        t = re.sub(r"^(?:total quantity of|total count of|quantity of|number of|how many|count the|count|is there|are there|can you see|do you see|tell me if there is|tell me if there are|see|locate|find|detect|show|pinpoint|identify|outline|segment|box|where\s+are|where\s+is|highlight|analyze|search for|spot)\s+(?:any\s+|all\s+|the\s+|a\s+|an\s+)?", "", t).strip()
        t = re.sub(r"^(?:and|or)\s+", "", t).strip()
        t = re.sub(r"^(?:a|an|the|any|all)\s+", "", t).strip()
        t = re.sub(r"\b(this|the|a|an|any|all|present|visible|detected|seen|of|for|in|on|at|are|is|there|here|scene|image|photo)\b", "", t).strip()
        t = re.sub(r"\s+", " ", t).strip()

        # Apply domain synonyms
        synonyms = {
            "houses": "buildings",
            "house": "building",
            "structures": "buildings",
            "structure": "building",
            "automobiles": "cars",
            "automobile": "car",
            "vessels": "ships",
            "vessel": "ship",
            "planes": "aircraft",
            "plane": "aircraft",
        }
        words = t.split()
        norm_words = [synonyms.get(w, w) for w in words]
        res = " ".join(norm_words).strip()
        return res


    @classmethod
    def _determine_geometry(cls, phrase: str) -> GeometryType:
        """
        Determines GeometryType based on semantic head-noun analysis.
        In English noun phrases (e.g. 'railway locomotives', 'mining quarries', 'parking areas'),
        the rightmost noun determines the true geometric category.
        """
        words = phrase.strip().lower().split()
        if not words:
            return GeometryType.DISCRETE_OBJECT

        # Exact multi-word check first (longest match)
        clean = phrase.strip().lower()
        for d in sorted(DISCRETE_NOUNS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(d)}\b", clean):
                # If discrete noun is the head noun (appears at end), return DISCRETE_OBJECT
                if clean.endswith(d) or words[-1] in d.split():
                    return GeometryType.DISCRETE_OBJECT

        for r in sorted(REGION_NOUNS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(r)}\b", clean):
                if clean.endswith(r) or words[-1] in r.split() or "quarr" in clean or "park" in clean or "area" in clean or "patch" in clean or "zone" in clean or "field" in clean or "site" in clean or "region" in clean:
                    return GeometryType.REGION_FEATURE

        for l in sorted(LINEAR_NOUNS, key=len, reverse=True):
            if re.search(rf"\b{re.escape(l)}\b", clean):
                if clean.endswith(l) or words[-1] in l.split():
                    return GeometryType.LINEAR_FEATURE

        # Head noun fallback
        head_noun = words[-1].rstrip("s")
        for d in DISCRETE_NOUNS:
            if head_noun == d.rstrip("s"):
                return GeometryType.DISCRETE_OBJECT
        for r in REGION_NOUNS:
            if head_noun == r.rstrip("s"):
                return GeometryType.REGION_FEATURE
        for l in LINEAR_NOUNS:
            if head_noun == l.rstrip("s"):
                return GeometryType.LINEAR_FEATURE

        return GeometryType.DISCRETE_OBJECT

    @classmethod
    def _determine_scale(cls, phrase: str, geom: GeometryType) -> ScaleCategory:
        if geom == GeometryType.REGION_FEATURE:
            return ScaleCategory.MACRO
        if geom == GeometryType.LINEAR_FEATURE:
            return ScaleCategory.LARGE

        p_lower = phrase.lower()
        for kw, scale in SCALE_PRIORS.items():
            if kw in p_lower:
                return scale

        return ScaleCategory.MEDIUM

    @classmethod
    def _determine_shape(cls, geom: GeometryType) -> str:
        if geom == GeometryType.LINEAR_FEATURE:
            return "elongated_line"
        if geom == GeometryType.REGION_FEATURE:
            return "amorphous_polygon"
        if geom == GeometryType.COMPLEX_CONCEPT:
            return "composite"
        return "compact_box"

    @classmethod
    def _build_open_vocab_prompt(cls, concept: str) -> str:
        """Constructs an open-vocabulary Grounding DINO text prompt."""
        clean = concept.strip().lower()
        if not clean:
            return "objects ."
        if clean.endswith("."):
            return clean
        if clean.endswith("s") and len(clean) > 3:
            return f"{clean} . {clean[:-1]} ."
        else:
            return f"{clean} . {clean}s ."

    @classmethod
    def parse_semantic_plan(cls, query: str) -> SemanticPlan:
        """
        Converts any arbitrary natural-language query into a structured SemanticPlan.
        Enforces strict TASK != TARGET separation: action verbs are NEVER targets.
        """
        q_raw = (query or "").strip()
        clean_q, constraints = cls.extract_safety_constraints(q_raw)
        q_lower = clean_q.lower() if clean_q else q_raw.lower()

        # 1. Out-of-domain / Unsupported
        out_of_domain = [
            r"\bwrite a (?:poem|story|code|essay|song|script)\b",
            r"\bwho is (?:the\s+)?(?:president|prime minister|ceo|governor|leader)\b",
            r"\bwhat is the capital of\b",
            r"\bbook a (?:flight|ticket|hotel|cab)\b",
            r"\bplay (?:music|song|video)\b",
            r"\btell me a (?:joke|riddle)\b",
            r"\bcalculate \d+\s*[\+\-\*\/]\s*\d+\b",
        ]

        for pattern in out_of_domain:
            if re.search(pattern, q_lower):
                return SemanticPlan(
                    query=q_raw,
                    intent_task="UNSUPPORTED",
                    tasks=["UNSUPPORTED"],
                    detection_status="not_applicable",
                    limitations=["Query is outside the domain of remote sensing and satellite image analysis."],
                    constraints=constraints,
                )

        # 2. Geospatial metadata & Geolocation inquiries
        is_geolocation = bool(re.search(r"\b(where is this (?:imagery|image|scene|photo) from|where is this|what location|where was this taken|location of this (?:image|scene|satellite|raster))\b", q_lower))
        if is_geolocation:
            return SemanticPlan(
                query=q_raw,
                intent_task="GEOLOCATION",
                tasks=["GEOLOCATION"],
                detection_status="not_applicable",
                limitations=["Geographic provenance depends on raster georeferencing and STAC metadata."],
                constraints=constraints,
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
            return SemanticPlan(
                query=q_raw,
                intent_task="GEOSPATIAL_METADATA",
                tasks=["GEOSPATIAL_METADATA"],
                detection_status="not_applicable",
                constraints=constraints,
            )

        # 3. Change analysis / comparison
        is_change = bool(re.search(
            r"\b(what changed|change detection|detect changes|find changes|changes between|changed regions?|scene changed|what areas changed|where has the scene changed|difference between|compare (?:these\s+)?(?:two\s+)?(?:images?|scenes?|photos?|the two)|compare|before and after|deforestation|what was built|what was destroyed|changed over time|biggest change|is the change|which areas changed|what disappeared|what was added)\b",
            q_lower,
        ))
        is_optical_sar = bool(re.search(
            r"\b(optical and sar|sar and optical|cross-modal|compare.*radar.*optical|compare.*optical.*radar|radar and optical|both sensors|both optical and sar)\b",
            q_lower,
        ))

        # 4. Counting
        is_counting = bool(cls.COUNT_REGEX.search(q_lower))

        # 5. Presence (e.g. "Is there water?", "Are there any houses?", "Can you see vegetation?")
        is_presence = (
            bool(re.search(r"\b(is there|are there|tell me if there is|tell me if there are)\b", q_lower))
            or bool(re.search(r"^(?:can you see|do you see)\s+(?:any\s+|a\s+|an\s+|the\s+)?([a-zA-Z]+)", q_lower))
        ) and not any(q_lower.startswith(p) for p in ["what do you see", "what can you see", "tell me what you see"])

        # 6. Localization
        is_loc = bool(cls.LOCALIZATION_REGEX.search(q_lower))

        # 7. Spatial relation (e.g. "ships near the port", "buildings adjacent to the highway", "north of the main road", "roads crossing the settlement")
        has_spatial_rel = bool(cls.RELATION_REGEX.search(q_lower)) or bool(re.search(r"\b(near|adjacent to|inside|outside|north of|south of|east of|west of|along|closest to|beside|close to|crossing|across|between|surrounding|bordering|flanking)\b", q_lower))

        # 8. Captioning / holistic scene description
        is_caption = (
            cls.is_scene_description(q_raw)
            or q_lower.startswith(("describe", "give a description", "give me a", "summarize", "overview", "what is in", "tell me what you see", "what do you see"))
            or bool(re.search(r"\b(describe|description of|overview|summarize|what is in this image|what is in the image|what is in this scene|what do you see|what you see|tell me what you see|what does this image contain|what does the image contain|what can you see)\b", q_lower))
        )

        # 8b. Holistic scene understanding / land use inquiry (e.g. 'what type of area is shown?')
        if cls.is_scene_understanding(q_lower):
            return SemanticPlan(
                query=q_raw,
                intent_task="VQA",
                tasks=["VQA"],
                target=None,
                targets=[],
                location=None,
                time=None,
                spatial_relation=None,
                spatial_relations=[],
                geometry=None,
                count_request=False,
                comparison_request=False,
                description_request=True,
                change_request=False,
                presence_check=False,
                confidence_requirement=0.20,
                target_specs=[],
                limitations=[],
                detection_status="not_applicable"
            )

        # Parse target specs
        target_specs = cls.parse(q_raw)
        targets = [ts.concept for ts in target_specs if ts.concept and ts.concept not in cls.NON_TARGET_TOKENS and ts.concept != "objects"]
        primary_target = targets[0] if targets else None

        # Determine tasks
        tasks = []
        if is_change:
            tasks.append("CHANGE_ANALYSIS")
        if is_optical_sar:
            tasks.append("OPTICAL_SAR_ANALYSIS")
        if is_caption:
            tasks.append("CAPTIONING")
        if is_counting:
            tasks.append("COUNTING")
        if is_presence and not is_counting:
            tasks.append("PRESENCE_CHECK")
        if is_loc and not is_counting and not is_presence:
            tasks.append("DETECTION")
        if has_spatial_rel and not is_change and (primary_target or "objects" in q_lower or "object" in q_lower):
            tasks.append("SPATIAL_REASONING")
            if not primary_target and ("objects" in q_lower or "object" in q_lower):
                primary_target = "objects"

        # Determine compound / primary task
        if is_optical_sar:
            intent_task = "OPTICAL_SAR_ANALYSIS"
        elif is_change:
            intent_task = "CHANGE_ANALYSIS"
        elif len(tasks) > 1 and "CAPTIONING" in tasks and ("DETECTION" in tasks or "COUNTING" in tasks or primary_target):
            intent_task = "MULTI_TASK"
        elif len(targets) > 1 and ("DETECTION" in tasks or is_loc):
            intent_task = "MULTI_TASK"
        elif is_counting:
            intent_task = "COUNTING"
        elif has_spatial_rel and primary_target:
            intent_task = "SPATIAL_REASONING"
        elif is_presence:
            intent_task = "PRESENCE_CHECK"
        elif tasks:
            intent_task = tasks[0]
        elif primary_target:
            intent_task = "DETECTION"
        else:
            intent_task = "CAPTIONING"

        geometry = None
        if primary_target:
            geom_type = cls._determine_geometry(primary_target)
            geometry = geom_type.value
        elif target_specs:
            geometry = target_specs[0].geometry_type.value

        spatial_relations = []
        for ts in target_specs:
            spatial_relations.extend(ts.spatial_relations)

        status = "model_ready"
        if intent_task in ("CAPTIONING", "UNSUPPORTED", "GEOSPATIAL_METADATA", "CHANGE_ANALYSIS", "OPTICAL_SAR_ANALYSIS") and not any(noun in q_lower for noun in ["building", "structure", "forest", "road", "water", "vegetation", "ship", "vehicle", "aircraft", "car", "tank"]):
            status = "not_applicable"
            primary_target = None
            targets = []
            target_specs = []
            geometry = None


        return SemanticPlan(
            query=q_raw,
            intent_task=intent_task,
            tasks=tasks if tasks else [intent_task],
            target=primary_target,
            targets=targets,
            location=None,
            time=None,
            spatial_relation=spatial_relations[0] if spatial_relations else None,
            spatial_relations=spatial_relations,
            geometry=geometry,
            count_request=is_counting,
            comparison_request=is_change or is_optical_sar,
            description_request=is_caption,
            change_request=is_change,
            presence_check=is_presence,
            confidence_requirement=0.20,
            target_specs=target_specs,
            limitations=[],
            detection_status=status,
            constraints=constraints,
        )


parse_semantic_plan = SemanticQueryParser.parse_semantic_plan
extract_safety_constraints = SemanticQueryParser.extract_safety_constraints

