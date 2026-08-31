"""
orchestrator.py
----------------
Top-level controller. Take a raw user request (+ imagery), produce the JSON
routing decision, execute downstream tools, and synthesize a single coherent,
transparent response — including surfacing low-confidence counting warnings
and fusing optical/SAR results.

Usage:
    from orchestrator import GeoVLMController
    controller = GeoVLMController()
    result = controller.handle_request(
        query="What is visible in this area? Is there a river?",
        optical_image=some_pil_image,
    )
    print(result["routing_decision"])   # the JSON routing block
    print(result["response_text"])      # final synthesized answer for the user
"""

from __future__ import annotations
from typing import Optional, List
from PIL import Image

from router import route, RoutingDecision
from tools import vqa_tool, optical_caption_tool, sar_caption_tool, VQAResult, CaptionResult
import change_analysis


COUNT_WARNING_TEXT = (
    "⚠️ Note: Exact numeric counts are derived with low model confidence "
    "(~0.25-0.40). Treat this count as an estimate."
)


class GeoVLMController:
    def __init__(self):
        self.vqa = vqa_tool
        self.optical_caption = optical_caption_tool
        self.sar_caption = sar_caption_tool

    # ------------------------------------------------------------------
    def handle_request(
        self,
        query: str = "",
        optical_image: Optional[Image.Image] = None,
        sar_image: Optional[Image.Image] = None,
        change_image_a: Optional[Image.Image] = None,
        change_image_b: Optional[Image.Image] = None,
        probe_features: Optional[List[str]] = None,
    ) -> dict:
        decision = route(
            query=query,
            has_optical=optical_image is not None,
            has_sar=sar_image is not None,
            has_change_pair=(change_image_a is not None and change_image_b is not None),
            probe_features=probe_features,
        )

        vqa_results: List[VQAResult] = []
        optical_caption_result: Optional[CaptionResult] = None
        sar_caption_result: Optional[CaptionResult] = None
        change_result = None

        # --- Execute each tool the router selected ---
        if "Change_Analysis" in decision.target_tools and change_image_a and change_image_b:
            change_result = change_analysis.analyze(change_image_a, change_image_b)

        if "Optical_Caption" in decision.target_tools and optical_image is not None:
            optical_caption_result = self.optical_caption.caption(optical_image, modality="optical")

        if "VQA" in decision.target_tools:
            target_image = optical_image if optical_image is not None else change_image_b
            if target_image is not None:
                vqa_results = self.vqa.ask_batch(target_image, decision.restructured_vqa_queries)

        if "SAR_Caption" in decision.target_tools and sar_image is not None:
            sar_caption_result = self.sar_caption.caption(sar_image, modality="sar")

        response_text = self._synthesize(
            decision, vqa_results, optical_caption_result, sar_caption_result, change_result
        )

        return {
            "routing_decision": decision.to_json(),
            "vqa_results": [vars(r) for r in vqa_results],
            "optical_caption": optical_caption_result.caption if optical_caption_result else None,
            "sar_caption": sar_caption_result.caption if sar_caption_result else None,
            "change_analysis": {
                "summary": change_result.summary,
                "changed_fraction": change_result.changed_fraction,
                "mean_intensity_delta": change_result.mean_intensity_delta,
            } if change_result else None,
            "response_text": response_text,
        }

    # ------------------------------------------------------------------
    def _synthesize(
        self,
        decision: RoutingDecision,
        vqa_results: List[VQAResult],
        optical_caption_result,
        sar_caption_result,
        change_result,
    ) -> str:
        parts = []

        if optical_caption_result:
            parts.append(f"**Optical scene description:** {optical_caption_result.caption}")

        if sar_caption_result:
            parts.append(f"**SAR scene description:** {sar_caption_result.caption}")

        if vqa_results:
            lines = []
            for r in vqa_results:
                line = f"- {r.question} → {r.answer}"
                if r.low_confidence:
                    line += "  ⚠️ low confidence"
                lines.append(line)
            parts.append("**Structured VQA findings:**\n" + "\n".join(lines))

        if change_result:
            parts.append(f"**Change analysis:** {change_result.summary}")

        if decision.requires_count_warning and any(
            "how many" in r.question.lower() for r in vqa_results
        ):
            parts.append(COUNT_WARNING_TEXT)

        if not parts:
            parts.append("No imagery or actionable query was provided.")

        return "\n\n".join(parts)
