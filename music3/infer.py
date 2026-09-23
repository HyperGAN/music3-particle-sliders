"""Sample card for a Music 3 slider.

``--dummy`` writes the listen card and does not render audio or download weights.
A non-dummy run is refused: published wavs live on the Hub, and this skeleton
does not load the Music 3 pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from music3.cards import release_card
from music3.defaults import (
    BASE_MODEL_ID,
    ENERGY_DEMO_DURATION_S,
    ENERGY_DEMO_SCALES,
    ENERGY_PROMPTS,
    HUB_WEIGHTS_ID,
    RELEASE_SCALES,
)
from music3.hosts import resolve_host
from music3.prompts import load_prompts
from music3.surface import load_stamp


def _glue_negative_option(argv: list[str], flag: str) -> list[str]:
    """argparse treats values like -2,-1 as extra options; attach them with '='."""
    glued: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == flag and index + 1 < len(argv) and argv[index + 1].startswith("-"):
            glued.append(f"{flag}={argv[index + 1]}")
            index += 2
            continue
        glued.append(argv[index])
        index += 1
    return glued


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            f"Write a Music 3 listen card for {BASE_MODEL_ID}. "
            "--dummy does not render audio and does not download Hub weights. "
            f"Published clips are on {HUB_WEIGHTS_ID}."
        ),
    )
    parser.add_argument("--name", default="music3-listen")
    parser.add_argument("--host", default="lm", choices=["lm", "transformer", "encoder", "particle"])
    parser.add_argument("--prompts_file", default=ENERGY_PROMPTS)
    parser.add_argument("--weights", default=None, help="adapter path; not read on --dummy")
    parser.add_argument("--scales", default=",".join(str(int(scale) if scale == int(scale) else scale) for scale in RELEASE_SCALES))
    parser.add_argument("--duration", type=float, default=ENERGY_DEMO_DURATION_S)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument("--out_dir", default="samples/music3")
    parser.add_argument("--model_id", default=BASE_MODEL_ID, help="base model id")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--print_card", action="store_true")
    parser.add_argument("--allow_hub", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    return build_parser().parse_args(_glue_negative_option(raw, "--scales"))


def _scales(raw: str) -> list[float]:
    scales = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not scales:
        raise ValueError("no scales given")
    return scales


def infer(args: argparse.Namespace) -> dict | Path:
    if args.model_id != BASE_MODEL_ID:
        raise ValueError(f"Music 3 samples {BASE_MODEL_ID}, got {args.model_id}")
    stamp = load_stamp()
    if args.print_card:
        card = release_card()
        card["formulation_id"] = stamp.formulation_id
        card["architecture_id"] = stamp.architecture_id
        print(json.dumps(card, indent=2))
        return card
    if not args.dummy:
        raise RuntimeError(
            "CPU infer needs --dummy. This skeleton writes a listen card and does not "
            f"load {BASE_MODEL_ID}. Published Off/On pairs are at {HUB_WEIGHTS_ID}."
        )
    rows, meta = load_prompts(args.prompts_file)
    host = resolve_host(args.host)
    scales = _scales(args.scales)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "kind": "music3_listen",
        "dummy": True,
        "allow_hub": bool(args.allow_hub),
        "rendered_audio": False,
        "model_id": BASE_MODEL_ID,
        "hub_weights": HUB_WEIGHTS_ID,
        "host": host["name"],
        "host_kind": host["kind"],
        "name": args.name,
        "weights": args.weights,
        "seed": int(args.seed),
        "duration_s": float(args.duration),
        "scales": scales,
        "neutral": rows[0].neutral,
        "lyrics": rows[0].lyrics,
        "plus_label": meta.plus_label,
        "concept": meta.concept,
        "formulation_id": stamp.formulation_id,
        "architecture_id": stamp.architecture_id,
        "lm_target": stamp.spec["lm_target"],
        "recommended_range": list(stamp.spec["recommended_range"]),
        "energy_demo_scales": list(ENERGY_DEMO_SCALES),
        "note": "neutral caption plus LoRA; the positive caption is the teacher, not the listen prompt",
    }
    path = out_dir / f"{args.name}_card.json"
    path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    infer(parse_args(argv))


if __name__ == "__main__":
    main()
