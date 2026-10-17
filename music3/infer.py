"""Reload shared PG09 adapter exports and execute CPU hidden-state probes.

This is a random native Qwen3 fixture. It does not synthesize acoustic audio.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
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
    parser = argparse.ArgumentParser(description=f"Reload PG09 exports for {BASE_MODEL_ID}; hidden probes only, no audio.")
    parser.add_argument("--dummy", action="store_true", help="actual random Qwen3 CPU response, no pretrained weights")
    parser.add_argument("--weights", type=Path, help="shared adapter-export.pt, read with weights_only=True")
    parser.add_argument("--scales", default="0,0.5,1", help="qualified unipolar probe range [0,1]")
    parser.add_argument("--out_dir", default="samples/music3-pg09")
    parser.add_argument("--host", choices=["lm", "particle", "transformer", "encoder"], default="lm")
    parser.add_argument("--model_id", default=BASE_MODEL_ID)
    parser.add_argument("--shared_checkout")
    parser.add_argument("--core_checkout")
    parser.add_argument("--preset", choices=["bispf", "bispf_optimized"], default="bispf_optimized")
    parser.add_argument("--pg_source_profile", choices=["release", "alz-continuous-v1"], default="release")
    parser.add_argument("--penalty_input_domain", choices=["auto", "continuous"], default="auto")
    parser.add_argument("--print_card", action="store_true", help="historical release catalog only")
    return parser


def parse_args(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    return build_parser().parse_args(_glue_negative_option(raw, "--scales"))


def _scales(raw):
    scales = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not scales or any(not math.isfinite(s) or not 0 <= s <= 1 for s in scales):
        raise ValueError("PG09 unipolar hidden probes require finite scales in [0,1]; negative inference is unqualified")
    return scales


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
    if resolve_host(args.host)["name"] != "lm":
        raise ValueError("PG09 currently qualifies only the LM/particle binding")
    scales = _scales(args.scales)
    if args.weights is None:
        raise ValueError("--weights adapter-export.pt is required; a metadata card is not inference")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(out)
    from music3.shared import bootstrap
    shared = bootstrap(args.shared_checkout, args.core_checkout)
    import torch
    adapter, model, fixed, game = shared.fixtures.make_audio(
        "music3", preset=args.preset, checkpointing=True, critic_class=shared.audio.AudioResponseCritic,
        pg_source_profile=args.pg_source_profile, penalty_input_domain=args.penalty_input_domain)
    payload = torch.load(args.weights, map_location="cpu", weights_only=True)
    if payload.get("schema") != "conditional-pg09-adapter-v1":
        raise ValueError("--weights must be a shared inference export, not a training checkpoint or historical safetensors")
    if type(payload.get("completed_steps")) is not int or not 1 <= payload["completed_steps"] <= 4:
        raise ValueError("CPU fixture requires an export with 1..4 completed steps")
    game.load_export_state_dict(payload)
    heldout = [adapter.prepare(replace(p.row, neutral_ids=(61,) + p.row.neutral_ids[1:])) for p in fixed]
    rng = torch.get_rng_state().clone()
    probes = [dict(scale=s, responses=[adapter.sample(p, scale=s) for p in heldout]) for s in scales]
    zero = [adapter.sample(p, scale=0) for p in heldout]
    if not all(torch.equal(v, p.neutral) for v, p in zip(zero, heldout)):
        raise ValueError("scale0 did not exactly replay the frozen base")
    if not torch.equal(rng, torch.get_rng_state()):
        raise ValueError("hidden probe changed the training RNG")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU hidden probe initialized CUDA")
    out.mkdir(parents=True, exist_ok=False)
    torch.save(dict(probes=probes, neutral=zero), out / "hidden-probes.pt")
    card = dict(kind="music3_pg09_hidden_probe", dummy=True, rendered_audio=False, quality_qualified=False,
                model_id=BASE_MODEL_ID, model_kind="random Qwen3 CPU fixture", host="lm", scales=scales,
                weights=str(args.weights.resolve()), weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                completed_steps=payload["completed_steps"], zero_replay_exact=True, rng_preserved=True,
                adapter_metadata=adapter.metadata(), source_receipt=shared.receipt,
                probe_file="hidden-probes.pt", note="hidden-state responses are not audio or acoustic-quality evidence")
    path = out / "probe-card.json"
    path.write_text(json.dumps(card, indent=2) + "\n")
    print(f"wrote {path}")
    return path


def main(argv=None):
    infer(parse_args(argv))


if __name__ == "__main__":
    main()
