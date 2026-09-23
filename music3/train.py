"""Train a Music 3 slider on the shared winning formulation.

``python scripts/train_music3.py --dummy`` is the CPU smoke path. It does not
download Hub weights. A non-dummy run is refused until a local Music 3
checkout is passed with ``--live``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music3.cards import release_card
from music3.defaults import (
    ADAPTER_ALPHA,
    ADAPTER_RANK,
    BASE_MODEL_ID,
    DEFAULT_PROMPTS,
    DUMMY_HIDDEN,
    DUMMY_MAX_STEPS,
    HUB_WEIGHTS_ID,
)
from music3.hosts import resolve_host, target_replace
from music3.loop import run_dummy
from music3.prompts import load_prompts
from music3.surface import declare, load_stamp

def build_parser(host_name: str = "lm") -> argparse.ArgumentParser:
    host = resolve_host(host_name)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            f"Train a Music 3 {host['kind']} slider with winning_formulation() "
            f"on {BASE_MODEL_ID}. --dummy never downloads Hub weights."
        ),
    )
    parser.add_argument("--name", default=f"music3-{host['name']}")
    parser.add_argument("--host", default=host_name, choices=["lm", "transformer", "encoder", "particle"])
    parser.add_argument("--prompts_file", default=DEFAULT_PROMPTS)
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--save_dir", default="models/music3-slider")
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--g_lr", type=float, default=None, help="model surface; default is the stamp")
    parser.add_argument("--d_lr", type=float, default=None, help="model surface; default is the stamp")
    parser.add_argument("--particle_lr", type=float, default=None, help="model surface; default is the stamp")
    parser.add_argument("--adv_batch", type=int, default=None, help="model surface; Music default is 64")
    parser.add_argument("--targets", default=None, choices=["attn", "full", "encoder"])
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lm_target", default=None, help="locked by the stamp; omit this flag")
    parser.add_argument("--loss", default=None, help="historical mse/nmse is not this game")
    parser.add_argument("--model_id", default=BASE_MODEL_ID)
    parser.add_argument("--model_dir", default=None, help="local MiniMax-Music-3 checkout for --live")
    parser.add_argument("--dummy", action="store_true", help="CPU stamp step, no Hub weights")
    parser.add_argument("--live", action="store_true", help="require a local model checkout")
    parser.add_argument("--allow_hub", action="store_true")
    parser.add_argument("--print_card", action="store_true")
    parser.set_defaults(_entry_host=host_name)
    return parser


def parse_args(argv: list[str] | None = None, *, host: str = "lm") -> argparse.Namespace:
    return build_parser(host).parse_args(argv)


def _reject_forks(parser: argparse.ArgumentParser, args: argparse.Namespace, stamp) -> None:
    if args.rank is not None and int(args.rank) != int(stamp.spec["adapter_rank"]):
        parser.error(
            f"--rank {args.rank} forks adapter_rank {int(stamp.spec['adapter_rank'])}. "
            "Change the stamp in HyperGAN/particle-sliders, not this repo."
        )
    if args.alpha is not None and float(args.alpha) != float(stamp.spec["adapter_alpha"]):
        parser.error(
            f"--alpha {args.alpha} forks adapter_alpha {float(stamp.spec['adapter_alpha'])}."
        )
    if args.lm_target is not None and str(args.lm_target) != str(stamp.spec["lm_target"]):
        parser.error(
            f"--lm_target {args.lm_target} is not the winning teacher {stamp.spec['lm_target']}. "
            "The v9 / faithful_sub_e menu stays in HyperGAN/particle-sliders and conceptmod."
        )
    if args.loss not in (None, "winning"):
        parser.error(
            "--loss mse/nmse/cos is the historical LoRA trainer in HyperGAN/particle-sliders. "
            "This product trains winning_formulation()."
        )
    if args.model_id != BASE_MODEL_ID:
        parser.error(f"Music 3 trains {BASE_MODEL_ID}, got {args.model_id}")
    if args.live and args.dummy:
        parser.error("choose either --live or --dummy")
    if args.adv_batch is not None and int(args.adv_batch) < 1:
        parser.error("--adv_batch must be positive")


def _prompts_from_config(args: argparse.Namespace) -> str:
    if not args.config_file:
        return args.prompts_file
    import yaml

    raw = yaml.safe_load(Path(args.config_file).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config file must be a mapping")
    return str(raw.get("prompts_file") or args.prompts_file)


def train(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> dict | Path:
    parser = parser or build_parser(getattr(args, "_entry_host", args.host))
    stamp = load_stamp()
    _reject_forks(parser, args, stamp)
    if args.print_card:
        card = release_card()
        card["train_host"] = resolve_host(args.host)["name"]
        card["formulation_id"] = stamp.formulation_id
        card["architecture_id"] = stamp.architecture_id
        print(json.dumps(card, indent=2))
        return card

    host = resolve_host(args.host)
    prompts_file = _prompts_from_config(args)
    rows, meta = load_prompts(prompts_file)
    if len(rows) < 2:
        raise ValueError(
            f"{prompts_file} has {len(rows)} row(s); the winning game needs at least two. "
            "Use configs/music3/prompts-particle.yaml."
        )
    declared = declare(
        stamp,
        g_lr=args.g_lr,
        d_lr=args.d_lr,
        particle_lr=args.particle_lr,
        adv_batch=args.adv_batch,
    )
    targets = target_replace(host, args.targets)
    if not args.dummy:
        if not args.live:
            raise RuntimeError(
                "Pass --dummy for the CPU stamp smoke, or --live with --model_dir "
                f"for a local {BASE_MODEL_ID} checkout. This entry does not download Hub weights."
            )
        if not args.model_dir:
            raise RuntimeError(
                "--live requires --model_dir pointing at a local MiniMax-Music-3 checkout. "
                "This command does not download it. Hub adapters stay at "
                f"{HUB_WEIGHTS_ID}."
            )
        model_dir = Path(args.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(model_dir)
        raise RuntimeError(
            f"Found {model_dir}, and the winning formulation stamp is already required. "
            "This skeleton does not load the Music 3 weights into host linears. "
            "Use --dummy for the CPU game step. A follow-up attaches stamp.bridge() "
            f"to {targets} inside that checkout."
        )
    steps = min(int(args.steps), DUMMY_MAX_STEPS)
    logs, built = run_dummy(
        stamp,
        declared,
        rows=len(rows),
        steps=steps,
        seed=int(args.seed),
        hidden=DUMMY_HIDDEN,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.name}_train.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for row in logs:
            handle.write(json.dumps(row) + "\n")
    sidecar = {
        "kind": "music3_particle",
        "entry_host": args.host,
        "host": host["name"],
        "host_kind": host["kind"],
        "target_replace": targets,
        "prefix": host["prefix"],
        "name": args.name,
        "model_id": BASE_MODEL_ID,
        "hub_weights": HUB_WEIGHTS_ID,
        "dummy": True,
        "allow_hub": bool(args.allow_hub),
        "live": False,
        "model_dir": args.model_dir,
        "prompts_file": meta.path,
        "plus_label": meta.plus_label,
        "minus_label": meta.minus_label,
        "concept": meta.concept,
        "recommended_range": list(meta.recommended_range),
        "prompt_rows": len(rows),
        "rank": ADAPTER_RANK,
        "alpha": ADAPTER_ALPHA,
        "architecture_id": stamp.architecture_id,
        "formulation_id": stamp.formulation_id,
        "formulation_provisional": bool(stamp.formulation_provisional),
        "lm_target": stamp.spec["lm_target"],
        "polarity": stamp.spec["polarity"],
        "require": "ok",
        "steps_requested": int(args.steps),
        "steps_ran": len(logs),
        "last": logs[-1] if logs else None,
        "bridge_class": type(built["bridge"]).__name__,
        "bridge_module": type(built["bridge"]).__module__,
        "critic_class": type(built["critic"]).__name__,
        "critic_module": type(built["critic"]).__module__,
        "regularizer_class": type(built["regularizer"]).__name__,
        "regularizer_module": type(built["regularizer"]).__module__,
    }
    sidecar_path = save_dir / f"{args.name}_last.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {sidecar_path}")
    return sidecar_path


def main(argv: list[str] | None = None, *, host: str = "lm") -> None:
    parser = build_parser(host)
    train(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
