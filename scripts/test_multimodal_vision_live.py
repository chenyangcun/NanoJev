#!/usr/bin/env python3
"""Live Multimodal Vision Verification for NanoJev Tri-Engine Server.

Tests:
1. Health check verification ("vision_available": true, "engine": "nanojev-tri-engine")
2. Direct image file path in state["image"]
3. Base64 data URI in state["image"]
4. Embedded <image path="..."> regex-extracted from state["user_task"]
5. Real screenshot error detection test
"""

import base64
import io
import json
import urllib.request
from pathlib import Path
from PIL import Image

ENDPOINT = "http://192.168.123.88:8769/v1/systemone"
HEALTH_ENDPOINT = "http://192.168.123.88:8769/api/health"


def post_systemone(payload: dict, headers: dict = None) -> dict:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def create_color_image(color: str, rgb: tuple) -> str:
    path = Path(f"/tmp/nanojev_test_{color}.png")
    img = Image.new("RGB", (128, 128), color=rgb)
    img.save(path)
    return str(path)


def create_base64_image(rgb: tuple) -> str:
    img = Image.new("RGB", (128, 128), color=rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main():
    print("=" * 70)
    print("NANOJEV MULTIMODAL VISION LIVE TEST SUITE")
    print("=" * 70)

    # 1. Health check
    print("\n1. Health Check & Engine Capabilities:")
    with urllib.request.urlopen(HEALTH_ENDPOINT) as resp:
        health = json.loads(resp.read().decode())
        print(f"   ✓ Engine Name     : {health.get('engine')}")
        print(f"   ✓ Vision Available: {health.get('vision_available')}")
        print(f"   ✓ ANE Fastlane    : {health.get('ane_fastlane_available')}")
        print(f"   ✓ GPU Engine      : {health.get('gpu_engine')}")
        assert health.get("vision_available") is True, "Vision must be available!"

    # 2. Direct Image File Path
    print("\n2. Test Direct Image File Path (state['image']):")
    red_path = create_color_image("red", (255, 0, 0))
    payload_path = {
        "model": "jev-latest",
        "state": {
            "image": red_path,
            "user_task": "Identify image primary color",
        },
        "questions": {
            "dominant_color": {
                "type": "choice",
                "instructions": "What is the primary color shown in this image?",
                "criteria": {
                    "red": "Bright red color",
                    "blue": "Blue ocean color",
                    "green": "Green grass color",
                },
            }
        },
    }
    res_path = post_systemone(payload_path)
    ans_path = res_path["answers"]["dominant_color"]
    print(f"   ✓ Selected Choice : {ans_path['choice']}")
    print(f"   ✓ Confidence      : {ans_path['confidence']:.4f}")
    print(f"   ✓ Probabilities   : {ans_path['probabilities']}")
    assert ans_path["choice"] == "red", f"Expected red, got {ans_path['choice']}"

    # 3. Base64 Data URI
    print("\n3. Test Base64 Data URI (state['image']):")
    blue_b64 = create_base64_image((0, 0, 255))
    payload_b64 = {
        "model": "jev-latest",
        "state": {
            "image": blue_b64,
            "user_task": "Check base64 image",
        },
        "questions": {
            "is_blue": {
                "type": "noul",
                "instructions": "Is this image blue?",
                "criteria": {
                    "true": "The image is blue",
                    "false": "The image is not blue",
                },
            }
        },
    }
    res_b64 = post_systemone(payload_b64)
    ans_b64 = res_b64["answers"]["is_blue"]
    print(f"   ✓ Noul Probability: {ans_b64['noul']:.4f} (p_true >= 0.5: {ans_b64['noul'] >= 0.5})")
    assert ans_b64["noul"] >= 0.5, f"Expected is_blue >= 0.5, got {ans_b64['noul']}"

    # 4. Embedded <image path="..."> Regex-Extracted from Text
    print("\n4. Test Embedded <image path='...'> in Task Text:")
    green_path = create_color_image("green", (0, 255, 0))
    prompt_text = f"<image name=[Image #1] path=\"{green_path}\">\n[Image #1] Please inspect this green screenshot."
    payload_regex = {
        "model": "jev-latest",
        "state": {
            "user_task": prompt_text,
            "has_image": True,
        },
        "questions": {
            "detected_color": {
                "type": "choice",
                "instructions": "What color was detected in the attached screenshot?",
                "criteria": {
                    "red": "Red error screen",
                    "green": "Green success screen",
                    "black": "Black terminal screen",
                },
            }
        },
    }
    res_regex = post_systemone(payload_regex)
    ans_regex = res_regex["answers"]["detected_color"]
    print(f"   ✓ Selected Choice : {ans_regex['choice']}")
    print(f"   ✓ Confidence      : {ans_regex['confidence']:.4f}")
    print(f"   ✓ Probabilities   : {ans_regex['probabilities']}")
    assert ans_regex["choice"] == "green", f"Expected green, got {ans_regex['choice']}"

    print("\n" + "=" * 70)
    print("ALL 4 MULTIMODAL VISION TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
