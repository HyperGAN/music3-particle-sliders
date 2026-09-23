"""Sample cards for the published Hub release and the local energy demo."""

from __future__ import annotations

import json
from pathlib import Path

from music3.defaults import (
    BASE_MODEL_ID,
    ENERGY_DEMO_DURATION_S,
    ENERGY_DEMO_SCALES,
    HUB_SPACE_URL,
    HUB_WEIGHTS_ID,
    HUB_WEIGHTS_URL,
    LISTEN_DURATION_S,
    RELEASE_POLICY,
    RELEASE_SCALES,
    RELEASE_SEED,
    RELEASE_TAG,
    SAMPLE_CARDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def control_paths(control: dict) -> dict:
    ident = control["id"]
    step = int(control["step"])
    return {
        "native": f"weights/{RELEASE_TAG}/{ident}/{ident}_step{step}.safetensors",
        "sidecar": f"weights/{RELEASE_TAG}/{ident}/{ident}_step{step}.json",
        "comfy": f"comfyui/{RELEASE_TAG}/{ident}_step{step}_comfyui.safetensors",
        "samples": f"samples/{RELEASE_TAG}/{ident}/",
    }


def load_sample_cards(path: Path | str | None = None) -> dict:
    file = Path(path) if path else REPO_ROOT / SAMPLE_CARDS
    if not file.is_absolute():
        file = REPO_ROOT / file
    payload = json.loads(file.read_text(encoding="utf-8"))
    controls = payload.get("controls")
    if not isinstance(controls, list) or len(controls) != 16:
        raise ValueError("sample card must list 16 Music 3 controls")
    return payload


def release_card() -> dict:
    catalog = load_sample_cards()
    return {
        "backend": "music3",
        "base_model": BASE_MODEL_ID,
        "hub_weights": HUB_WEIGHTS_ID,
        "hub_url": HUB_WEIGHTS_URL,
        "space": HUB_SPACE_URL,
        "release": RELEASE_TAG,
        "selection": RELEASE_POLICY,
        "seed": RELEASE_SEED,
        "scales": list(RELEASE_SCALES),
        "duration_s": LISTEN_DURATION_S,
        "strength": 1.0,
        "prompt_policy": "neutral caption, fixed lyrics and seed; slider strength only",
        "controls": [{**control, **control_paths(control)} for control in catalog["controls"]],
        "energy_demo": {
            "scales": list(ENERGY_DEMO_SCALES),
            "duration_s": ENERGY_DEMO_DURATION_S,
            "prompts_file": "configs/music3/prompts-music3.yaml",
        },
    }
