"""Music 3 projection hosts. Names only; the routed branch comes from the stamp."""

from __future__ import annotations

from typing import Any

HOSTS: dict[str, dict[str, Any]] = {
    "lm": {
        "kind": "language_model",
        "target_replace": ["Qwen3Attention"],
        "prefix": "lora_te",
        "targets_mode": "attn",
        "description": "AR language model. Voice and arrangement axes (gender, and the LM half of energy/tempo/distortion).",
    },
    "transformer": {
        "kind": "transformer",
        "target_replace": ["MiniMaxMusic3Attention"],
        "target_replace_full": [
            "MiniMaxMusic3Attention",
            "MiniMaxMusic3TransformerBlock",
            "MiniMaxMusic3Transformer1DModel",
        ],
        "prefix": "lora_unet",
        "targets_mode": "full",
        "attn_modules": 144,
        "full_modules": 222,
        "description": "Flow transformer. Production axes (energy, distortion, tempo, space, dust).",
    },
    "encoder": {
        "kind": "condition_encoder",
        "target_replace": ["MiniMaxMusic3ConditionEncoder"],
        "prefix": "lora_unet",
        "targets_mode": "encoder",
        "description": "Condition encoder. Historical notrigger analog; the product game is still the winning stamp.",
    },
}

# The published particle bridge wrapped the LM, not the flow transformer.
PARTICLE_HOST = "lm"


def resolve_host(name: str) -> dict[str, Any]:
    key = str(name).strip().lower()
    if key in ("particle", "music3", "language_model", "text_encoder"):
        key = "lm"
    if key in ("tf", "dit", "flow"):
        key = "transformer"
    if key in ("condition_encoder", "cond"):
        key = "encoder"
    if key not in HOSTS:
        raise ValueError(f"unknown Music 3 host {name!r}; expected lm, transformer, or encoder")
    host = dict(HOSTS[key])
    host["name"] = key
    return host


def target_replace(host: dict[str, Any], targets: str | None = None) -> list[str]:
    mode = (targets or host.get("targets_mode") or "attn").strip().lower()
    if host["name"] == "transformer":
        if mode == "full":
            return list(host["target_replace_full"])
        if mode == "attn":
            return list(host["target_replace"])
        raise ValueError(f"targets mode {targets!r} is not valid for the transformer host")
    if mode != host.get("targets_mode"):
        raise ValueError(f"targets mode {targets!r} is not valid for host {host['name']}")
    return list(host["target_replace"])
