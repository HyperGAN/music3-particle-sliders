"""Thin Music3 entry points for the installed shared PG0.9.1 audio CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from music3.cards import release_card
from music3.defaults import BASE_MODEL_ID
from music3.hosts import resolve_host


def build_parser(host_name="lm"):
    parser = argparse.ArgumentParser(description=f"Packaged GAN-only PG0.9.1 Music3 training for {BASE_MODEL_ID}.")
    parser.add_argument("--host", default=host_name, choices=["lm", "particle", "transformer", "encoder"])
    parser.add_argument("--dummy", action="store_true", help="shared genuine random-Qwen3 CPU qualification")
    parser.add_argument("--live", action="store_true", help="shared native plan only; coordinator owns admission")
    parser.add_argument("--print_card", action="store_true", help="historical release catalog only")
    parser.add_argument("--save_dir", default="models/music3-pg091")
    parser.add_argument("--steps", type=int, default=3, help="absolute completed-step target")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--resume", type=Path, help="full new-lineage shared checkpoint; old PG0.9.0 is rejected")
    parser.add_argument("--preset", choices=["bispf", "bispf_optimized"], default="bispf_optimized")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=32, help="CPU fixture response width;4096 requires explicit continuous profile")
    parser.add_argument("--pg_source_profile", choices=["release", "alz-continuous-v1"], default="release")
    parser.add_argument("--penalty_input_domain", choices=["auto", "continuous"], default="auto")
    parser.add_argument("--model_id", default=BASE_MODEL_ID)
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prompts_file")
    parser.add_argument("--heldout_prompts", type=Path)
    for name in ("name", "g_lr", "d_lr", "particle_lr", "adv_batch", "rank", "alpha", "lm_target", "loss", "targets", "config_file", "shared_checkout", "core_checkout"):
        parser.add_argument(f"--{name}", default=None, help="historical/source-checkout option; refused")
    parser.add_argument("--allow_hub", action="store_true", help="unsupported; local model inputs only")
    return parser


def parse_args(argv=None, *, host="lm"):
    return build_parser(host).parse_args(argv)


def train(args, parser=None):
    parser = parser or build_parser(args.host)
    if args.model_id != BASE_MODEL_ID:
        parser.error(f"Music3 requires {BASE_MODEL_ID}")
    if args.print_card:
        card = release_card()
        card.update(catalog_only=True, pg09_checkpoint_compatible=False, native_pg09_audio_qualified=False)
        print(json.dumps(card, indent=2))
        return card
    if resolve_host(args.host)["name"] != "lm":
        parser.error("PG09 currently qualifies only the LM/particle binding; transformer/encoder training is pending")
    old = [name for name in ("name", "g_lr", "d_lr", "particle_lr", "adv_batch", "rank", "alpha", "lm_target", "loss", "targets", "config_file", "shared_checkout", "core_checkout") if getattr(args, name) is not None]
    if old or args.allow_hub:
        parser.error("historical/source-checkout options are refused: " + ", ".join(old or ["allow_hub"]))
    if args.live == args.dummy:
        parser.error("choose --dummy for CPU qualification or --live for a native plan")
    total = args.total_steps if args.total_steps is not None else (4 if args.dummy else 20)
    if not 1 <= args.steps <= total <= 20 or (args.dummy and total != 4):
        parser.error("require1 <=steps <=total_steps <=20; CPU fixture horizon is4")
    if args.dummy and any((args.model_dir, args.manifest, args.prompts_file, args.heldout_prompts)):
        parser.error("--dummy uses the shared fixed fixture rows; model/prompt paths require --live")
    argv = ["--family", "music3", "--out", str(Path(args.save_dir).resolve()), "--until", str(args.steps),
            "--total-steps", str(total), "--preset", args.preset, "--seed", str(args.seed),
            "--pg-source-profile", args.pg_source_profile, "--penalty-input-domain", args.penalty_input_domain]
    if args.resume:
        argv += ["--resume", str(args.resume.resolve())]
    if args.live:
        if args.resume:
            parser.error("native continuation requires its coordinator-owned producer; --resume is CPU-only here")
        argv = ["--family", "music3", "--out", str(Path(args.save_dir).resolve()),
                "--until", str(args.steps), "--total-steps", str(total),
                "--preset", args.preset, "--seed", str(args.seed),
                "--pg-source-profile", args.pg_source_profile, "--penalty-input-domain", args.penalty_input_domain]
        if any(value is None for value in (args.model_dir, args.manifest, args.prompts_file, args.heldout_prompts)):
            parser.error("--live plan requires model_dir, manifest, prompts_file and heldout_prompts")
        argv = ["plan", *argv, "--model-dir", str(args.model_dir.resolve()), "--manifest", str(args.manifest.resolve()),
                "--train-prompts", str(Path(args.prompts_file).resolve()), "--heldout-prompts", str(args.heldout_prompts.resolve())]
    else:
        argv += ["--fixture", "--width", str(args.width)]
    from music3.shared import bootstrap
    shared = bootstrap()
    result = shared.cli.main(argv)
    if args.dummy:
        receipt = dict(lineage="music3-packaged-pg091-v1", product="music3-particle-sliders",
                       mode="shared-random-qwen3-cpu", source_receipt=shared.receipt,
                       native_audio_qualified=False, old_checkpoint_migration=False)
        (Path(args.save_dir) / "product-integration.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return result


def main(argv=None, *, host="lm"):
    parser = build_parser(host)
    return train(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
