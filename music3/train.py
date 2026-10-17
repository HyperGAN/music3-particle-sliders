"""Product CLI for the reviewed shared Music3 PG0.9 worker.

The shared worker owns preparation, every update, checkpoints and qualification.
Native commands are reviewable plans; the coordinator owns GPU execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from music3.cards import release_card
from music3.defaults import BASE_MODEL_ID
from music3.hosts import resolve_host


def build_parser(host_name: str = "lm") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Shared GAN-only PG0.9 Music3 training for {BASE_MODEL_ID}.")
    parser.add_argument("--host", default=host_name, choices=["lm", "particle", "transformer", "encoder"])
    parser.add_argument("--dummy", action="store_true", help="actual random Qwen3 CPU worker; no pretrained weights")
    parser.add_argument("--live", action="store_true", help="write a native command plan for coordinator review; never dispatch")
    parser.add_argument("--print_card", action="store_true", help="historical release catalog only")
    parser.add_argument("--shared_checkout")
    parser.add_argument("--core_checkout")
    parser.add_argument("--save_dir", default="models/music3-pg09")
    parser.add_argument("--steps", type=int, default=3, help="absolute completed step to reach")
    parser.add_argument("--total_steps", type=int, default=None, help="fixed recipe horizon; CPU4, native20")
    parser.add_argument("--resume", type=Path, help="shared full checkpoint.pt; inference exports cannot resume")
    parser.add_argument("--preset", choices=["bispf", "bispf_optimized"], default="bispf_optimized")
    parser.add_argument("--pg_source_profile", choices=["release", "alz-continuous-v1"], default="release")
    parser.add_argument("--penalty_input_domain", choices=["auto", "continuous"], default="auto")
    parser.add_argument("--model_id", default=BASE_MODEL_ID)
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--heldout_prompts", type=Path)
    # Keep old flags visible so an old command fails with a useful explanation.
    for name in ("name", "g_lr", "d_lr", "particle_lr", "adv_batch", "rank", "alpha", "lm_target", "loss", "targets", "config_file", "seed"):
        parser.add_argument(f"--{name}", default=None, help="historical option; unsupported by the reviewed PG09 worker")
    parser.add_argument("--allow_hub", action="store_true", help="unsupported; local inputs only")
    return parser


def parse_args(argv=None, *, host="lm"):
    return build_parser(host).parse_args(argv)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(args, parser=None):
    parser = parser or build_parser(args.host)
    if args.model_id != BASE_MODEL_ID:
        parser.error(f"Music3 requires {BASE_MODEL_ID}")
    if args.print_card:
        card = release_card()
        card.update(catalog_only=True, pg09_checkpoint_compatible=False,
                    native_pg09_audio_qualified=False)
        print(json.dumps(card, indent=2))
        return card
    if resolve_host(args.host)["name"] != "lm":
        parser.error("PG09 currently qualifies only the LM/particle binding; transformer/encoder training is pending")
    historical = [name for name in ("name", "g_lr", "d_lr", "particle_lr", "adv_batch", "rank", "alpha", "lm_target", "loss", "targets", "config_file", "seed") if getattr(args, name) is not None]
    if historical or args.allow_hub:
        parser.error("historical options are not PG09 recipe overrides: " + ", ".join(historical or ["allow_hub"]))
    if args.live == args.dummy:
        parser.error("choose --dummy for CPU execution or --live for a coordinator command plan")
    total = args.total_steps if args.total_steps is not None else (4 if args.dummy else 20)
    if not 1 <= args.steps <= total <= 20 or (args.dummy and total != 4):
        parser.error("require 1 <= steps <= total_steps <=20, and the CPU fixture recipe horizon is4")
    if args.dummy and any((args.model_dir, args.manifest, args.prompts_file, args.heldout_prompts)):
        parser.error("--dummy uses the shared fixed token rows; native prompt/model options require --live")
    from music3.shared import bootstrap
    shared = bootstrap(args.shared_checkout, args.core_checkout)
    worker_args = ["--family", "music3", "--out", str(Path(args.save_dir).resolve()),
                   "--until", str(args.steps), "--total-steps", str(total),
                   "--preset", args.preset, "--pg-source-profile", args.pg_source_profile,
                   "--penalty-input-domain", args.penalty_input_domain]
    if args.resume:
        worker_args += ["--resume", str(args.resume.resolve())]
    if args.live:
        if any(value is None for value in (args.model_dir, args.manifest, args.prompts_file, args.heldout_prompts)):
            parser.error("--live plan requires model_dir, manifest, prompts_file and heldout_prompts")
        for value in (args.manifest, args.prompts_file, args.heldout_prompts):
            if not Path(value).is_file():
                raise FileNotFoundError(value)
        output = Path(args.save_dir).resolve()
        if not output.is_relative_to("/mnt/nvme/sliders") or output.exists():
            parser.error("native planned outputs must be new directories under /mnt/nvme/sliders")
        config = args.model_dir / "language_model/config.json"
        width = json.loads(config.read_text())["hidden_size"]
        shared.audio.require_audio_width(width, pg_source_profile=args.pg_source_profile,
                                         penalty_input_domain=args.penalty_input_domain)
        worker_args += ["--device", "cuda:0", "--model-dir", str(args.model_dir.resolve()),
                        "--manifest", str(args.manifest.resolve()), "--train-prompts", str(Path(args.prompts_file).resolve()),
                        "--heldout-prompts", str(args.heldout_prompts.resolve())]
        roots = [shared.core_checkout / "packages/particle-sliders-core/src",
                 shared.shared_checkout, shared.shared_checkout / "scripts"]
        if args.pg_source_profile == "alz-continuous-v1":
            import particlegan
            overlay = Path(particlegan.__file__).resolve().parent.parent
            if overlay != Path("/tmp/alz-pg09-domain-proposal").resolve():
                raise ValueError("native planning requires the reviewed isolated ALZ overlay; no site-packages injection")
            roots.insert(0, overlay)
        environment = dict(PYTHONPATH=os.pathsep.join(map(str, roots)), PYTHONDONTWRITEBYTECODE="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8",
                           CUDA_VISIBLE_DEVICES="<ROOT_ASSIGNED_SINGLE_PHYSICAL_GPU>")
        plan = dict(status="PROPOSED_NOT_ADMITTED", executed=False, family="music3", model_id=BASE_MODEL_ID,
                    command=[sys.executable, str(shared.worker_path), *worker_args],
                    environment=environment, source_receipt=shared.receipt,
                    native_runtime=shared.worker.runtime_identity(),
                    input_sha256={str(Path(p).resolve()): _sha(p) for p in (args.manifest, args.prompts_file, args.heldout_prompts, config)},
                    coordinator_thread="01a147d7-f540-778e-ad79-4d8eb267403d",
                    remaining=["root-owned phase guardian/card, lease and admission-id/minimum-free-gib",
                               "fresh resource and native runtime qualification", "pretrained Music3 exact-resume smoke",
                               "native acoustic render and listening"])
        print(json.dumps(plan, indent=2))
        return plan
    worker_args += ["--fixture", "--device", "cpu"]
    previous_argv = sys.argv
    try:
        sys.argv = [str(shared.worker_path), *worker_args]
        shared.worker.main()
    finally:
        sys.argv = previous_argv
    out = Path(args.save_dir)
    status = json.loads((out / "progress.json").read_text())
    status.update(product="music3-particle-sliders", mode="random_qwen3_cpu_hidden_response",
                  source_receipt=shared.receipt, acoustic_audio_qualified=False)
    receipt = out / "product-integration.json"
    receipt.write_text(json.dumps(status, indent=2) + "\n")
    return receipt


def main(argv=None, *, host="lm"):
    parser = build_parser(host)
    train(parser.parse_args(argv), parser)


if __name__ == "__main__":
    main()
