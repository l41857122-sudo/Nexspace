"""
evidence_graph.py
-----------------
Traceable Evidence Graph & Provenance Validation Engine.

Structures all specialist model and algorithmic outputs into linked, validated,
and auditable evidence nodes connected to the user query and processing pipeline.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvidenceNode:
    """Individual evidence unit produced by a specialist tool or model."""
    evidence_id: str
    type: str  # "object_detection" | "change_region" | "cross_modal_indicator" | "vqa_answer" | "caption" | "modality_statistics"
    source_tool: str
    source_model: str
    derived_from: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    confidence_type: str = "unavailable"  # "model" | "heuristic" | "unavailable"
    confidence_source: str = "adapter"
    validation_status: str = "valid"  # "valid" | "rejected"
    validation_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "type": self.type,
            "source_tool": self.source_tool,
            "source_model": self.source_model,
            "derived_from": self.derived_from,
            "payload": self.payload,
            "confidence": self.confidence,
            "confidence_type": self.confidence_type,
            "confidence_source": self.confidence_source,
            "validation_status": self.validation_status,
            "validation_error": self.validation_error,
        }


class EvidenceGraph:
    """Directed graph capturing evidence nodes, their provenance, and lineage."""

    def __init__(self, query_id: str = "query_001"):
        self.query_id = query_id
        self.nodes: Dict[str, EvidenceNode] = {}
        self._counter = 0

    def next_id(self, prefix: str = "ev") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter:03d}"

    def add_node(self, node: EvidenceNode) -> bool:
        """Validates and adds an evidence node to the graph with capacity protection."""
        if len(self.nodes) >= 1000:
            node.validation_status = "rejected"
            node.validation_error = "Maximum evidence node capacity reached (1000 nodes)"
            return False

        is_valid, err = self.validate_node(node)
        if not is_valid:
            node.validation_status = "rejected"
            node.validation_error = err
            return False

        node.validation_status = "valid"
        self.nodes[node.evidence_id] = node
        return True

    def create_and_add(
        self,
        type: str,
        source_tool: str,
        source_model: str,
        payload: Dict[str, Any],
        derived_from: Optional[List[str]] = None,
        confidence: Optional[float] = None,
        confidence_type: str = "unavailable",
        confidence_source: str = "adapter",
    ) -> Optional[EvidenceNode]:
        """Convenience factory to create, validate, and add an evidence node."""
        node_id = self.next_id()
        node = EvidenceNode(
            evidence_id=node_id,
            type=type,
            source_tool=source_tool,
            source_model=source_model,
            derived_from=derived_from or [self.query_id],
            payload=payload,
            confidence=confidence,
            confidence_type=confidence_type,
            confidence_source=confidence_source,
        )
        success = self.add_node(node)
        return node if success else None

    def validate_node(self, node: EvidenceNode) -> tuple[bool, Optional[str]]:
        """Validates schema, uniqueness, numeric sanity, and geometry."""
        # 1. Unique ID
        if node.evidence_id in self.nodes:
            return False, f"Duplicate evidence_id: {node.evidence_id}"

        # 2. Source existence
        if not node.source_tool or not node.source_model:
            return False, "Missing source_tool or source_model provenance"

        # 3. Numeric sanity in confidence
        if node.confidence is not None:
            if math.isnan(node.confidence) or math.isinf(node.confidence):
                return False, "Confidence contains NaN or Infinity"
            if node.confidence < 0.0 or node.confidence > 1.0:
                return False, f"Confidence out of range [0, 1]: {node.confidence}"

        # 4. Payload validation
        payload = node.payload
        if not isinstance(payload, dict):
            return False, "Payload must be a dictionary"

        # Check for NaN / Infinity in payload floats
        for k, v in payload.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return False, f"Payload field '{k}' contains NaN or Infinity"

        # Bounding box geometry validation
        if "bbox_pixel" in payload:
            box = payload["bbox_pixel"]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                return False, "bbox_pixel must be a 4-element list"
            x1, y1, x2, y2 = box
            if any(math.isnan(c) or math.isinf(c) for c in (x1, y1, x2, y2)):
                return False, "bbox_pixel coordinates contain NaN or Infinity"
            if x2 <= x1 or y2 <= y1:
                return False, f"Degenerate box dimensions: [{x1}, {y1}, {x2}, {y2}]"

        if "bbox_normalized" in payload:
            norm_box = payload["bbox_normalized"]
            if not isinstance(norm_box, (list, tuple)) or len(norm_box) != 4:
                return False, "bbox_normalized must be a 4-element list"
            for c in norm_box:
                if math.isnan(c) or math.isinf(c) or c < 0.0 or c > 1.0:
                    return False, f"Normalized coordinate out of bounds: {c}"

        if "bbox_world" in payload and payload["bbox_world"] is not None:
            world_box = payload["bbox_world"]
            if isinstance(world_box, dict):
                for k in ("min_x", "min_y", "max_x", "max_y"):
                    val = world_box.get(k)
                    if val is not None and (math.isnan(val) or math.isinf(val)):
                        return False, f"bbox_world field '{k}' contains NaN or Infinity"

        return True, None

    def get_by_type(self, evidence_type: str) -> List[EvidenceNode]:
        return [n for n in self.nodes.values() if n.type == evidence_type]

    def to_list(self) -> List[Dict[str, Any]]:
        return [node.to_dict() for node in self.nodes.values()]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "total_nodes": len(self.nodes),
            "node_types": list({n.type for n in self.nodes.values()}),
            "nodes": self.to_list(),
        }
