"""
capability_registry.py
----------------------
Capability-Based Model Registry and Tool Selector for NexSpace.

Routes by semantic capability, geometry, resolution scale, and sensor input
rather than rigid keyword-to-class if/else tables.
"""

from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set

from target_spec import TargetSpec, GeometryType, ScaleCategory


class ModelCapability(str, Enum):
    OPEN_VOCABULARY_OBJECT_DETECTION = "OPEN_VOCABULARY_OBJECT_DETECTION"
    REMOTE_SENSING_OBJECT_DETECTION = "REMOTE_SENSING_OBJECT_DETECTION"
    SEMANTIC_REGION_SEGMENTATION = "SEMANTIC_REGION_SEGMENTATION"
    LINEAR_FEATURE_EXTRACTION = "LINEAR_FEATURE_EXTRACTION"
    VQA = "VQA"
    IMAGE_CAPTIONING = "IMAGE_CAPTIONING"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    CHANGE_VQA = "CHANGE_VQA"
    SPATIAL_RELATION_REASONING = "SPATIAL_RELATION_REASONING"
    COUNTING = "COUNTING"
    OPTICAL_SAR_ANALYSIS = "OPTICAL_SAR_ANALYSIS"


@dataclass
class CapabilityDescriptor:
    model_id: str
    tool_name: str
    capabilities: Set[ModelCapability]
    supported_geometries: Set[GeometryType]
    supported_scales: Set[ScaleCategory]
    supported_sensors: Set[str]  # "optical_rgb", "sar", "bitemporal_optical", "optical_sar"
    confidence_type: str  # "model", "heuristic", "calibrated"
    latency_profile: str  # "fast", "standard", "heavy"
    is_available: bool = True

    def matches(
        self,
        geometry: GeometryType,
        scale: ScaleCategory,
        sensor: str,
        required_capability: Optional[ModelCapability] = None,
    ) -> bool:
        if geometry not in self.supported_geometries:
            return False
        if sensor not in self.supported_sensors:
            return False
        if required_capability and required_capability not in self.capabilities:
            return False
        return self.is_available


class CapabilityRegistry:
    """
    Central capability registry selecting the most appropriate specialist tool
    and execution plan for any given TargetSpec and sensor context.
    """

    def __init__(self):
        self.descriptors: List[CapabilityDescriptor] = [
            CapabilityDescriptor(
                model_id="Grounding_DINO_Tiny",
                tool_name="Grounding",
                capabilities={
                    ModelCapability.OPEN_VOCABULARY_OBJECT_DETECTION,
                    ModelCapability.REMOTE_SENSING_OBJECT_DETECTION,
                    ModelCapability.COUNTING,
                },
                supported_geometries={GeometryType.DISCRETE_OBJECT, GeometryType.LINEAR_FEATURE},
                supported_scales={ScaleCategory.TINY, ScaleCategory.SMALL, ScaleCategory.MEDIUM, ScaleCategory.LARGE},
                supported_sensors={"optical_rgb"},
                confidence_type="model",
                latency_profile="standard",
            ),
            CapabilityDescriptor(
                model_id="RS_Specialist_Region_Segmenter",
                tool_name="Grounding",
                capabilities={
                    ModelCapability.SEMANTIC_REGION_SEGMENTATION,
                },
                supported_geometries={GeometryType.REGION_FEATURE},
                supported_scales={ScaleCategory.LARGE, ScaleCategory.MACRO},
                supported_sensors={"optical_rgb"},
                confidence_type="heuristic",
                latency_profile="fast",
            ),
            CapabilityDescriptor(
                model_id="Salesforce_BLIP_VQA",
                tool_name="VQA",
                capabilities={
                    ModelCapability.VQA,
                    ModelCapability.IMAGE_CAPTIONING,
                },
                supported_geometries={GeometryType.DISCRETE_OBJECT, GeometryType.REGION_FEATURE, GeometryType.COMPLEX_CONCEPT},
                supported_scales={ScaleCategory.SMALL, ScaleCategory.MEDIUM, ScaleCategory.LARGE, ScaleCategory.MACRO},
                supported_sensors={"optical_rgb"},
                confidence_type="model",
                latency_profile="fast",
            ),
            CapabilityDescriptor(
                model_id="Differential_Change_Analysis_Engine",
                tool_name="Change_Analysis",
                capabilities={
                    ModelCapability.CHANGE_DETECTION,
                },
                supported_geometries={GeometryType.DISCRETE_OBJECT, GeometryType.REGION_FEATURE, GeometryType.LINEAR_FEATURE},
                supported_scales={ScaleCategory.MEDIUM, ScaleCategory.LARGE, ScaleCategory.MACRO},
                supported_sensors={"bitemporal_optical"},
                confidence_type="heuristic",
                latency_profile="fast",
            ),
            CapabilityDescriptor(
                model_id="Optical_SAR_Fusion_Engine",
                tool_name="Optical_SAR_Analysis",
                capabilities={
                    ModelCapability.OPTICAL_SAR_ANALYSIS,
                },
                supported_geometries={GeometryType.DISCRETE_OBJECT, GeometryType.REGION_FEATURE},
                supported_scales={ScaleCategory.MEDIUM, ScaleCategory.LARGE, ScaleCategory.MACRO},
                supported_sensors={"optical_sar"},
                confidence_type="heuristic",
                latency_profile="fast",
            ),
        ]

    def select_tools_for_spec(
        self,
        spec: TargetSpec,
        has_optical: bool = True,
        has_sar: bool = False,
        has_change_pair: bool = False,
    ) -> List[str]:
        """Selects matching tools based on TargetSpec geometry, scale, and sensor availability."""
        selected: List[str] = []

        sensor = "optical_rgb"
        if has_change_pair:
            sensor = "bitemporal_optical"
        elif has_optical and has_sar:
            sensor = "optical_sar"
        elif has_sar:
            sensor = "sar"

        # Check for change requirement
        if spec.temporal_requirements and has_change_pair:
            selected.append("Change_Analysis")

        # Select primary localization / segmentation tool
        for desc in self.descriptors:
            if desc.matches(geometry=spec.geometry_type, scale=spec.expected_scale, sensor=sensor):
                if desc.tool_name not in selected:
                    selected.append(desc.tool_name)

        if not selected:
            # Fallback to Grounding if optical is present
            if has_optical:
                selected.append("Grounding")

        return selected


# Global registry instance
capability_registry = CapabilityRegistry()
