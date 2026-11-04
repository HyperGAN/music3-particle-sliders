"""Thin shared-export probe CLI. Acoustic Music3 rendering remains unqualified."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from music3.cards import release_card
from music3.defaults import BASE_MODEL_ID
from music3.hosts import resolve_host


def _glue_negative_option(argv, flag):
    glued = []
    index = 0
    while index < len(argv):
        if argv[index] == flag and index + 1 < len(argv) and argv[index + 1].startswith("-"):
            glued.append(f"{flag}={argv[index + 1]}")
            index += 2
        else:
            glued.append(argv[index])
            index += 1
    return glued


def build_parser():
    parser = argparse.ArgumentParser(
        description=f"Packaged PG0.9.1 exports for {BASE_MODEL_ID}; CPU hidden probes, no audio."
    )
    parser.add_argument("--dummy", action="store_true", help="shared genuine random-Qwen3 CPU hidden probes")
    parser.add_argument("--weights", type=Path, help="shared new-lineage adapter export")
    parser.add_argument("--scales", default="0,0.5,1")
    parser.add_argument("--out_dir", default="samples/music3-pg091")
    parser.add_argument("--host", choices=["lm", "particle", "transformer", "encoder"], default="lm")
    parser.add_argument("--model_id", default=BASE_MODEL_ID)
    parser.add_argument("--print_card", action="store_true")
    parser.add_argument("--shared_checkout", help="historical source-checkout option; refused")
    parser.add_argument("--core_checkout", help="historical source-checkout option; refused")
    return parser


def parse_args(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    return build_parser().parse_args(_glue_negative_option(raw, "--scales"))


def _scales(raw):
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values):
        raise ValueError("PG09 unipolar hidden probes require finite scales in [0,1]; negative inference is unqualified")
    return values


def infer(args):
    if args.model_id != BASE_MODEL_ID:
        raise ValueError(f"Music3 requires {BASE_MODEL_ID}")
    if args.print_card:
        card = release_card()
        card.update(catalog_only=True, pg09_checkpoint_compatible=False, native_pg09_audio_qualified=False)
        print(json.dumps(card, indent=2))
        return card
    if not args.dummy:
        raise RuntimeError("--dummy is required for hidden probes; native Music3 acoustic rendering is pending coordinator qualification")
    if args.shared_checkout or args.core_checkout:
        raise ValueError("source-checkout options are refused; install the exact locked wheel")
    if resolve_host(args.host)["name"] != "lm":
        raise ValueError("PG09 currently qualifies only the LM/particle binding")
    values = _scales(args.scales)
    if args.weights is None:
        raise ValueError("--weights shared adapter export is required")
    from music3.shared import bootstrap
    shared = bootstrap()
    result = shared.cli.main([
        "probe", "--family", "music3", "--export", str(args.weights.resolve()),
        "--scales", ",".join(map(str, values)), "--out", str(Path(args.out_dir).resolve()),
    ])
    receipt = dict(lineage="music3-packaged-pg091-v1", source_receipt=shared.receipt,
                   native_audio_qualified=False, old_checkpoint_migration=False)
    (Path(args.out_dir) / "product-integration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


def main(argv=None):
    return infer(parse_args(argv))


if __name__ == "__main__":
    main()
