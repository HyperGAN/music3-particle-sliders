"""Music 3 prompt cards. List-form energy YAML and dict-form particle cards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PromptRow:
    neutral: str
    positive: str
    lyrics: str
    negative: str = ""
    target: str = ""
    action: str = "enhance"
    guidance_scale: float = 4.0


@dataclass(frozen=True)
class PromptMeta:
    plus_label: str
    minus_label: str
    recommended_range: tuple[float, float]
    concept: str
    path: str


def _clean(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text or "\x00" in text:
        raise ValueError(f"{field} must be a nonempty string")
    return text


def _as_rows(raw: object) -> tuple[list[dict], dict]:
    if isinstance(raw, list):
        return raw, {}
    if isinstance(raw, dict) and isinstance(raw.get("rows"), list):
        return raw["rows"], raw
    raise ValueError("prompts need a list of rows or a mapping with a rows list")


def load_prompts(path: Path | str) -> tuple[list[PromptRow], PromptMeta]:
    file = Path(path)
    if not file.is_absolute():
        candidate = REPO_ROOT / file
        if candidate.exists():
            file = candidate
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    items, header = _as_rows(raw)
    if not items:
        raise ValueError(f"no prompt rows in {file}")
    recommended = header.get("recommended_range") or [0.0, 1.0]
    if list(recommended) not in ([0.0, 1.0], [0, 1]):
        raise ValueError("winning Music 3 card uses recommended_range [0, 1]")
    rows: list[PromptRow] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each prompt row must be a mapping")
        neutral = _clean(item.get("neutral") or item.get("target"), "neutral")
        positive = _clean(item.get("positive"), "positive")
        lyrics = _clean(item.get("lyrics"), "lyrics")
        if neutral == positive:
            raise ValueError("neutral and positive captions must differ")
        negative = str(item.get("negative") or "").strip()
        rows.append(
            PromptRow(
                neutral=neutral,
                positive=positive,
                lyrics=lyrics,
                negative=negative,
                target=str(item.get("target") or neutral).strip(),
                action=str(item.get("action") or "enhance"),
                guidance_scale=float(item.get("guidance_scale") or item.get("guidance") or 4.0),
            )
        )
    meta = PromptMeta(
        plus_label=str(header.get("plus_label") or "On"),
        minus_label=str(header.get("minus_label") or "Off"),
        recommended_range=(0.0, 1.0),
        concept=str(header.get("concept") or "energy"),
        path=str(file),
    )
    return rows, meta
