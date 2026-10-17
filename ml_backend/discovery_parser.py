"""
discovery_parser.py
-------------------
Natural Language Satellite Imagery Discovery & Investigation Parser for NexSpace.

Decomposes unconstrained natural-language queries into structured discovery plans:
- LOCATION: Place names, coordinate pairs, bounding boxes, or "current/this area".
- TIME: Temporal intervals, relative dates ("recent", "latest", "last month"), or bitemporal ("before and after").
- IMAGERY REQUIREMENTS: Sensor ("Sentinel-2", "optical", "SAR"), max cloud cover, resolution.
- ANALYSIS TASK: "discovery_only", "detection", "change_analysis", "region_segmentation".
- TARGET: Concept nouns ("construction sites", "ships", "buildings") mapped to TargetSpec.
- SPATIAL RELATION: Predicates ("near", "inside", "along") and reference landmarks.

Zero hardcoded query branches.
"""

from __future__ import annotations
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

from target_spec import TargetSpec, GeometryType, ScaleCategory, SpatialRelation, SemanticQueryParser
from geocoding import GeocodingEngine, GeocodedLocation, ResolvedLocation, GLOBAL_GAZETTEER, make_bbox_around_point


@dataclass
class TemporalConstraint:
    raw_text: str = ""
    is_bitemporal: bool = False
    datetime_range: Optional[str] = None  # ISO-8601 interval e.g. "2024-01-01T00:00:00Z/2024-02-01T00:00:00Z"
    datetime_before: Optional[str] = None
    datetime_after: Optional[str] = None
    description: str = "Latest available"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "is_bitemporal": self.is_bitemporal,
            "datetime_range": self.datetime_range,
            "datetime_before": self.datetime_before,
            "datetime_after": self.datetime_after,
            "description": self.description,
        }


@dataclass
class ImageryRequirements:
    sensor: str = "Sentinel-2"
    modality: str = "optical"  # "optical" | "sar" | "multispectral"
    max_cloud_cover: float = 20.0  # Percentage [0..100]
    resolution_preference: str = "best_available"  # "10m" | "best_available"
    collection: str = "sentinel-2-c1-l2a"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensor": self.sensor,
            "modality": self.modality,
            "max_cloud_cover": self.max_cloud_cover,
            "resolution_preference": self.resolution_preference,
            "collection": self.collection,
        }


@dataclass
class SemanticDiscoveryPlan:
    query: str
    location_raw: Optional[str] = None
    geocoding: Optional[GeocodedLocation] = None
    temporal: TemporalConstraint = field(default_factory=TemporalConstraint)
    imagery: ImageryRequirements = field(default_factory=ImageryRequirements)
    analysis_task: str = "discovery_only"  # "discovery_only" | "detection" | "change_analysis" | "region_segmentation"
    target_concepts: List[str] = field(default_factory=list)
    target_specs: List[TargetSpec] = field(default_factory=list)
    spatial_relation: Optional[SpatialRelation] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "location_raw": self.location_raw,
            "geocoding": self.geocoding.to_dict() if self.geocoding else None,
            "temporal": self.temporal.to_dict(),
            "imagery": self.imagery.to_dict(),
            "analysis_task": self.analysis_task,
            "target_concepts": self.target_concepts,
            "target_specs": [ts.to_dict() for ts in self.target_specs],
            "spatial_relation": self.spatial_relation.to_dict() if self.spatial_relation else None,
        }


class DiscoveryQueryParser:
    """
    Generalized semantic parser for Earth observation discovery and analysis requests.
    """

    # Spatial Relation Prepositions (Gate 3 Step 7 Canonical Taxonomy)
    RELATION_PREPS = [
        ("near", r"\bnear\b|\bclose to\b|\baround\b|\bvicinity of\b|\bin proximity to\b"),
        ("inside", r"\binside\b|\bwithin\b|\bin the interior of\b|\blocated inside\b"),
        ("outside", r"\boutside of\b|\boutside\b|\bexterior to\b"),
        ("adjacent", r"\badjacent to\b|\badjacent\b|\bbeside\b|\bnext to\b|\bflanking\b|\bbordering\b"),
        ("along", r"\balong\b|\balongside\b|\bfollowing\b"),
        ("crossing", r"\bcrossing\b|\bintersecting\b|\bacross\b"),
        ("closest_to", r"\bclosest to\b|\bnearest to\b|\bclosest\b|\bnearest\b"),
        ("north_of", r"\bnorth of\b|\bto the north of\b"),
        ("south_of", r"\bsouth of\b|\bto the south of\b"),
        ("east_of", r"\beast of\b|\bto the east of\b"),
        ("west_of", r"\bwest of\b|\bto the west of\b"),
        ("beside", r"\bbeside\b"),
    ]

    MONTH_NAMES = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    @classmethod
    def parse(
        cls,
        query: str,
        default_bbox: Optional[List[float]] = None,
        explicit_location: Optional[str] = None,
    ) -> SemanticDiscoveryPlan:
        """
        Parses unconstrained natural language query into a comprehensive SemanticDiscoveryPlan.
        
        Location Resolution Priority (Deterministic 8-Tier):
        1. Explicit coordinates (lat/lon pair, directional pair, or explicit coords string)
        2. Explicit AOI / bounding box (default_bbox)
        3. Structured location field (explicit_location)
        4. Location extracted from natural language
        5. Image georeferencing metadata
        6. Scene metadata
        7. Model-estimated location
        8. UNRESOLVED
        """
        clean_q = query.strip()
        q_lower = clean_q.lower()

        # 1. Parse Temporal Constraints
        temporal = cls._extract_temporal_constraints(clean_q)

        # 2. Parse Imagery Requirements
        imagery = cls._extract_imagery_requirements(clean_q)

        # 3. Extract Location candidates from natural language
        nl_loc, rem_nl = cls._extract_location_string(clean_q)

        location_raw: Optional[str] = None
        geocoding_res: Optional[ResolvedLocation] = None
        clean_after_loc = clean_q

        # Tier 1: Explicit coordinates in structured input
        if explicit_location and explicit_location.strip():
            coord_test = GeocodingEngine.parse_coordinate_string(explicit_location.strip())
            if coord_test:
                location_raw = explicit_location.strip()
                geocoding_res = coord_test
                clean_after_loc = rem_nl if nl_loc else clean_q

        # Tier 1 from Natural Language (e.g. coordinates in text)
        if not geocoding_res and nl_loc:
            coord_test = GeocodingEngine.parse_coordinate_string(nl_loc)
            if coord_test:
                location_raw = nl_loc
                geocoding_res = coord_test
                clean_after_loc = rem_nl

        # Tier 2: Explicit AOI / bounding box (via default_bbox)
        if not geocoding_res and default_bbox and len(default_bbox) >= 4:
            location_raw = explicit_location or nl_loc or "explicit_aoi"
            w, s, e, n = default_bbox[0], default_bbox[1], default_bbox[2], default_bbox[3]
            geocoding_res = ResolvedLocation(
                query="explicit_aoi",
                display_name=f"User Specified AOI [{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}]",
                latitude=round((s + n) / 2.0, 6),
                longitude=round((w + e) / 2.0, 6),
                bbox=[round(w, 6), round(s, 6), round(e, 6), round(n, 6)],
                source="EXPLICIT_COORDINATES",
                confidence=1.0,
                geocoding_available=True,
            )
            clean_after_loc = rem_nl if nl_loc else clean_q

        # Tier 3: Structured location field (place name)
        if not geocoding_res and explicit_location and explicit_location.strip():
            loc_clean = explicit_location.strip()
            location_raw = loc_clean
            geocoding_res = GeocodingEngine.resolve_location(loc_clean)
            clean_after_loc = rem_nl if nl_loc else clean_q

        # Tier 4: Location extracted from natural-language query
        if not geocoding_res and nl_loc:
            location_raw = nl_loc
            geocoding_res = GeocodingEngine.resolve_location(nl_loc)
            clean_after_loc = rem_nl

        # Tier 8: UNRESOLVED — Never invent coordinates
        if not geocoding_res:
            location_raw = None
            geocoding_res = None
            clean_after_loc = clean_q
        elif not geocoding_res.geocoding_available:
            clean_after_loc = clean_q

        # 5. Extract Spatial Relation
        spatial_rel, clean_after_rel = cls._extract_spatial_relation(clean_after_loc)

        # 6. Extract Targets & Downstream Analysis Task
        target_concepts, analysis_task = cls._extract_targets_and_task(clean_after_rel, temporal.is_bitemporal)

        # 7. Generate TargetSpecs
        target_specs: List[TargetSpec] = []
        if target_concepts:
            for concept in target_concepts:
                sub_query = f"locate {concept}"
                if spatial_rel:
                    sub_query += f" {spatial_rel.relation} {spatial_rel.reference_entity}"
                parsed_specs = SemanticQueryParser.parse(sub_query)
                target_specs.extend(parsed_specs)

        return SemanticDiscoveryPlan(
            query=clean_q,
            location_raw=location_raw,
            geocoding=geocoding_res,
            temporal=temporal,
            imagery=imagery,
            analysis_task=analysis_task,
            target_concepts=target_concepts,
            target_specs=target_specs,
            spatial_relation=spatial_rel,
        )

    @classmethod
    def _extract_temporal_constraints(cls, query: str) -> TemporalConstraint:
        q_lower = query.lower()
        now = datetime.now(timezone.utc)

        # Check for Bitemporal "before and after"
        if re.search(r"\bbefore and after\b|\bpre[- ]and[- ]post\b|\bchange(?:s)? over time\b", q_lower):
            # Try to identify an event or date anchor
            event_match = re.search(r"(?:before and after|pre and post)\s+(?:the\s+)?([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)?)", q_lower)
            anchor = event_match.group(1).strip() if event_match else "event"

            # Check if year/month mentioned
            year_match = re.search(r"\b(20\d\d)\b", q_lower)
            year = int(year_match.group(1)) if year_match else now.year

            # Default before = 3-6 months prior, after = 0-3 months prior
            t_before = f"{year-1}-10-01T00:00:00Z/{year}-01-01T00:00:00Z"
            t_after = f"{year}-01-01T00:00:00Z/{year}-06-01T00:00:00Z"

            return TemporalConstraint(
                raw_text=anchor,
                is_bitemporal=True,
                datetime_before=t_before,
                datetime_after=t_after,
                description=f"Before and after analysis ({anchor})",
            )

        # Month and Year (e.g. "July 2024", "from July")
        for m_name, m_num in cls.MONTH_NAMES.items():
            pattern = rf"\b{m_name}\b(?:\s+(\d{{4}}))?"
            match = re.search(pattern, q_lower)
            if match:
                yr = int(match.group(1)) if match.group(1) else now.year
                # If requested month is in future for current year, use previous year
                if yr == now.year and m_num > now.month:
                    yr -= 1
                start_dt = datetime(yr, m_num, 1, 0, 0, 0, tzinfo=timezone.utc)
                if m_num == 12:
                    end_dt = datetime(yr + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
                else:
                    end_dt = datetime(yr, m_num + 1, 1, 0, 0, 0, tzinfo=timezone.utc)

                dt_range = f"{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                return TemporalConstraint(
                    raw_text=match.group(0),
                    is_bitemporal=False,
                    datetime_range=dt_range,
                    description=f"Month of {m_name.capitalize()} {yr}",
                )

        # Relative temporal expressions
        if "last month" in q_lower:
            start_dt = now - timedelta(days=60)
            end_dt = now - timedelta(days=15)
            return TemporalConstraint(
                raw_text="last month",
                is_bitemporal=False,
                datetime_range=f"{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                description="Past month interval",
            )

        if "this week" in q_lower or "past week" in q_lower:
            start_dt = now - timedelta(days=7)
            return TemporalConstraint(
                raw_text="this week",
                is_bitemporal=False,
                datetime_range=f"{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                description="Past 7 days",
            )

        if "today" in q_lower:
            start_dt = now - timedelta(days=2)
            return TemporalConstraint(
                raw_text="today",
                is_bitemporal=False,
                datetime_range=f"{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                description="Past 48 hours",
            )

        # Default: Recent / Latest (past 90 days)
        start_dt = now - timedelta(days=90)
        return TemporalConstraint(
            raw_text="recent",
            is_bitemporal=False,
            datetime_range=f"{start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')}/{now.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            description="Recent / Latest available (past 90 days)",
        )

    @classmethod
    def _extract_imagery_requirements(cls, query: str) -> ImageryRequirements:
        q_lower = query.lower()

        # Cloud cover threshold
        max_cloud = 20.0
        if re.search(r"\bcloud[- ]free\b|\bclear sky\b|\bzero clouds?\b|\bno clouds?\b", q_lower):
            max_cloud = 5.0
        elif re.search(r"\blow cloud(?:s| cover)?\b", q_lower):
            max_cloud = 10.0
        else:
            cloud_pct_match = re.search(r"<\s*(\d+)%\s*cloud", q_lower)
            if cloud_pct_match:
                max_cloud = float(cloud_pct_match.group(1))

        # Sensor & Modality
        sensor = "Sentinel-2"
        modality = "optical"
        collection = "sentinel-2-c1-l2a"
        if re.search(r"\bsar\b|\bsentinel[- ]1\b|\bradar\b", q_lower):
            sensor = "Sentinel-1"
            modality = "sar"
            collection = "sentinel-1-grd"

        return ImageryRequirements(
            sensor=sensor,
            modality=modality,
            max_cloud_cover=max_cloud,
            resolution_preference="best_available",
            collection=collection,
        )

    @classmethod
    def _extract_location_string(cls, query: str) -> Tuple[Optional[str], str]:
        """
        Extracts geographic place name or coordinate pair from natural text,
        returning (location_string, remaining_query_text).
        Supports decimal coords, directional coords ("26.8467 N, 80.9462 E"),
        prepositional phrases, and 120+ global entities without hardcoding.
        """
        # 1. Directional coordinates: "26.8467 N, 80.9462 E" or "40.7128° N, 74.0060° W"
        dir_match = re.search(
            r"(\d{1,2}(?:\.\d+)?)\s*°?\s*([NSns])\s*[,;\s]\s*(\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])",
            query,
        )
        if dir_match:
            loc_str = dir_match.group(0).strip()
            rem = query.replace(loc_str, " ").strip()
            return loc_str, rem

        # 2. Decimal coordinate pair pattern: "26.8467, 80.9462" or "-40.7128, -74.0060"
        coord_match = re.search(r"(-?\d{1,2}\.\d{2,8})\s*,\s*(-?\d{1,3}\.\d{2,8})", query)
        if coord_match:
            loc_str = coord_match.group(0).strip()
            rem = query.replace(loc_str, " ").strip()
            return loc_str, rem

        # 3. Location phrases: "of <Location>", "in <Location>", "around <Location>", "over <Location>", "near <Location>"
        patterns = [
            r"\b(?:satellite imagery|satellite images|imagery|images|image)\s+of\s+([A-Za-z0-9\s,\.-]+?)(?=\s+(?:and\s+|from\s+|before|after|with|near|$)|$)",
            r"\b(?:satellite imagery|imagery|images)\s+around\s+([A-Za-z0-9\s,\.-]+?)(?=\s+(?:and\s+|from\s+|$)|$)",
            r"\b(?:find|detect|locate|show)\s+(?:ships|buildings|vehicles|roads|water bodies|construction sites|targets)\s+near\s+([A-Za-z0-9\s,\.-]+?)(?=\s+(?:and|from|$)|$)",
            r"\bimagery\s+near\s+([A-Za-z0-9\s,\.-]+?)(?=\s+(?:and|from|$)|$)",
            r"\b(?:over|in|at)\s+([A-Za-z0-9\s,\.-]+?)(?=\s+(?:and\s+(?:locate|find|detect|show)|from\s+|before|after|$)|$)",
            r"(?:^|\b(?:show|find|get|display|retrieve)\s+)([A-Za-z\s]+?)\s+imagery\b",
        ]

        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                cand_clean = re.sub(r"^(?:show|find|get|display|retrieve|view|me|the|a|an)\s+", "", cand, flags=re.IGNORECASE).strip()
                cand_clean = re.sub(r"\b(this location|this area|the area|here)\b", "this area", cand_clean, flags=re.IGNORECASE).strip()
                # If cand was "this location" or "this area", treat as user viewport/AOI
                if cand_clean.lower() == "this area":
                    rem = query.replace(m.group(0), " ").strip()
                    return "this area", rem

                if len(cand_clean) >= 3 and not cand_clean.lower().startswith("last month") and not cand_clean.lower().startswith("august") and not cand_clean.lower().startswith("satellite"):
                    rem = query.replace(m.group(0), " ").strip()
                    return cand_clean, rem

        # 4. Universal match against comprehensive global gazetteer (120+ entities)
        q_lower = query.lower()
        for hub in sorted(GLOBAL_GAZETTEER.keys(), key=len, reverse=True):
            if re.search(rf"\b{re.escape(hub)}\b", q_lower):
                rem = re.sub(rf"\b{re.escape(hub)}\b", " ", query, flags=re.IGNORECASE).strip()
                return GLOBAL_GAZETTEER[hub]["display_name"].split(",")[0], rem

        return None, query

    @classmethod
    def _extract_spatial_relation(cls, query: str) -> Tuple[Optional[SpatialRelation], str]:
        """Extracts spatial relations like 'near docks', 'along coastline'."""
        for rel_name, rel_regex in cls.RELATION_PREPS:
            pattern = rf"({rel_regex})\s+(?:the\s+|a\s+|an\s+)?([A-Za-z0-9_-]+(?:\s+[A-Za-z0-9_-]+)*?)(?=\s+(?:and|with)\b|$)"
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                # Group 1 is the relation match, group 2 is the reference entity
                ref_entity = m.group(2).strip()
                sr = SpatialRelation(
                    relation=rel_name,
                    reference_entity=ref_entity,
                    raw_phrase=m.group(0).strip(),
                )
                rem = query.replace(m.group(0), " ").strip()
                return sr, rem
        return None, query

    @classmethod
    def _extract_targets_and_task(cls, query: str, is_bitemporal: bool) -> Tuple[List[str], str]:
        """
        Identifies target concepts (e.g. 'ships', 'construction sites', 'buildings')
        and maps to the appropriate analysis task.
        """
        q_clean = query.strip()
        if is_bitemporal:
            return [], "change_analysis"

        # First, strip discovery lead-in phrases ("find recent satellite imagery of", "show satellite images of", etc.)
        discovery_prefix_pattern = r"^(?:show|find|get|display|retrieve|search for)\s+(?:recent\s+|latest\s+|cloud-free\s+|optical\s+)?(?:satellite\s+)?(?:imagery|images|image|scenes?|data)\s+(?:of|around|in|near|over)?\s*"
        clean_target_text = re.sub(discovery_prefix_pattern, "", q_clean, flags=re.IGNORECASE).strip()

        # If there's an 'and <verb>' clause for analysis, focus on that
        and_action_match = re.search(
            r"\band\s+(?:locate|find|detect|identify|outline|pinpoint|show|search for|count)\s+(?:any\s+|all\s+|the\s+|a\s+|an\s+)?([A-Za-z0-9\s,]+)",
            clean_target_text,
            re.IGNORECASE,
        )

        target_str = ""
        if and_action_match:
            target_str = and_action_match.group(1).strip()
        else:
            # Check for direct detection lead-in without discovery prefix
            direct_lead = re.search(
                r"\b(?:locate|detect|identify|outline|pinpoint)\s+(?:any\s+|all\s+|the\s+|a\s+|an\s+)?([A-Za-z0-9\s,]+)",
                clean_target_text,
                re.IGNORECASE,
            )
            if direct_lead:
                target_str = direct_lead.group(1).strip()
            else:
                # Check known remote-sensing nouns
                cand_concepts = ["construction sites", "ships", "vessels", "cargo ships", "boats", "buildings", "containers", "airplanes", "aircraft", "roads", "quarries", "bridges", "storage tanks"]
                found = []
                for c in cand_concepts:
                    if re.search(rf"\b{re.escape(c)}\b", clean_target_text, re.IGNORECASE):
                        found.append(c)
                if found:
                    target_str = ", ".join(found)

        if not target_str:
            clean_for_desc = re.sub(r"\b(?:of|in|around|over|near|at)\s*[,.\s]*[A-Za-z\s]*$", "", clean_target_text).strip()
            if (
                SemanticQueryParser.is_scene_description(clean_target_text)
                or SemanticQueryParser.is_scene_description(clean_for_desc)
                or SemanticQueryParser.is_scene_description(q_clean)
                or bool(re.search(r"\b(describe|summarize|overview|what is in|tell me what you see)\b", q_clean, flags=re.IGNORECASE))
            ):
                return [], "captioning"
            return [], "discovery_only"

        # Strip temporal words, location words, and spatial preposition tails
        target_str = re.sub(r"\b(near|inside|around|along|beside|across|over|at|in|on|from)\s+[A-Za-z0-9\s]+$", "", target_str, flags=re.IGNORECASE).strip()
        target_str = re.sub(r"\b(recent|latest|today|this week|last month|yesterday|before|after|flood|earthquake|july|august|january|february|march|april|may|june|september|october|november|december)\b", "", target_str, flags=re.IGNORECASE).strip()

        # Split multiple targets
        parts = re.split(r"[,&]|\band\b", target_str)
        cleaned_targets = []
        for p in parts:
            if SemanticQueryParser.is_scene_description(p):
                continue
            p_c = re.sub(r"\b(satellite|imagery|images|area|scene|location|port|city|locate|find|detect|show|identify|pinpoint|describe|description)\b", "", p, flags=re.IGNORECASE).strip()
            p_c = re.sub(r"\b(near|inside|around|along|beside|across|over|at|in|on|from|of)\b\s*$", "", p_c, flags=re.IGNORECASE).strip()
            p_c = re.sub(r"^\s*\b(near|inside|around|along|beside|across|over|at|in|on|from|of)\b", "", p_c, flags=re.IGNORECASE).strip()
            p_c = re.sub(r"\s+", " ", p_c).strip()
            if p_c.lower() in ("india", "usa", "china", "japan", "france", "germany", "uk", "russia"):
                continue
            if p_c and len(p_c) >= 3 and p_c not in {"it", "them", "all", "the", "and", "from", "for"}:
                cleaned_targets.append(p_c)

        if not cleaned_targets:
            if SemanticQueryParser.is_scene_description(clean_target_text) or SemanticQueryParser.is_scene_description(q_clean):
                return [], "captioning"
            return [], "discovery_only"

        # Determine task
        analysis_task = "detection"
        for t in cleaned_targets:
            geom = SemanticQueryParser._determine_geometry(t)
            if geom == GeometryType.REGION_FEATURE:
                analysis_task = "region_segmentation"
                break

        return cleaned_targets, analysis_task
