"""
scripts/generate_sample_images_ts.py
------------------------------------
Writes app/utils/sampleImages.ts linking real static files in public/demo/ with base64 data.
"""

import base64
import os

DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "public", "demo")
TARGET_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "utils", "sampleImages.ts")


def get_b64(fname: str) -> str:
    p = os.path.join(DEMO_DIR, fname)
    with open(p, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def main():
    port_b64 = get_b64("port.png")
    urban_b64 = get_b64("urban.png")
    sar_b64 = get_b64("sar.png")
    a_b64 = get_b64("bitemporal_a.png")
    b_b64 = get_b64("bitemporal_b.png")

    content = f'''/**
 * app/utils/sampleImages.ts
 * -------------------------
 * Real 512x512 remote sensing satellite imagery for live browser demo & API analysis.
 */

// Next.js public static asset paths
export const SAMPLE_OPTICAL_PORT_URL = "/demo/port.png";
export const SAMPLE_OPTICAL_URBAN_URL = "/demo/urban.png";
export const SAMPLE_SAR_RADAR_URL = "/demo/sar.png";
export const SAMPLE_CHANGE_A_URL = "/demo/bitemporal_a.png";
export const SAMPLE_CHANGE_B_URL = "/demo/bitemporal_b.png";

// Base64 encoded 512x512 rasters for immediate backend API transmission
export const SAMPLE_OPTICAL_PORT = "{port_b64}";
export const SAMPLE_OPTICAL_URBAN = "{urban_b64}";
export const SAMPLE_SAR_RADAR = "{sar_b64}";
export const SAMPLE_CHANGE_A = "{a_b64}";
export const SAMPLE_CHANGE_B = "{b_b64}";

export interface DemoImageOption {{
  id: string;
  name: string;
  category: "optical" | "sar" | "temporal";
  url: string;
  base64: string;
  dimensions: string;
  description: string;
}}

export const DEMO_IMAGE_CATALOG: DemoImageOption[] = [
  {{
    id: "port",
    name: "Coastal Port and Docks",
    category: "optical",
    url: SAMPLE_OPTICAL_PORT_URL,
    base64: SAMPLE_OPTICAL_PORT,
    dimensions: "512x512 RGB",
    description: "Nadir optical imagery of marine terminal, cargo ships, and coastal water",
  }},
  {{
    id: "urban",
    name: "Urban Grid and Infrastructure",
    category: "optical",
    url: SAMPLE_OPTICAL_URBAN_URL,
    base64: SAMPLE_OPTICAL_URBAN,
    dimensions: "512x512 RGB",
    description: "Optical overhead view of city blocks, road grid, and building rooftops",
  }},
  {{
    id: "sar",
    name: "SAR Radar Backscatter",
    category: "sar",
    url: SAMPLE_SAR_RADAR_URL,
    base64: SAMPLE_SAR_RADAR,
    dimensions: "512x512 Grayscale",
    description: "Synthetic Aperture Radar backscatter showing corner reflection and smooth water return",
  }},
  {{
    id: "bitemporal_a",
    name: "Baseline (Time A)",
    category: "temporal",
    url: SAMPLE_CHANGE_A_URL,
    base64: SAMPLE_CHANGE_A,
    dimensions: "512x512 RGB",
    description: "Pre-construction temporal baseline satellite capture",
  }},
  {{
    id: "bitemporal_b",
    name: "Post-Event (Time B)",
    category: "temporal",
    url: SAMPLE_CHANGE_B_URL,
    base64: SAMPLE_CHANGE_B,
    dimensions: "512x512 RGB",
    description: "Post-construction satellite capture with newly erected industrial facility",
  }},
];
'''
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully generated app/utils/sampleImages.ts with full catalog and base64 data.")


if __name__ == "__main__":
    main()
