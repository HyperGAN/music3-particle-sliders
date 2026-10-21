# Reproduce the packaged Music 3 integration

The locally qualified dependency is `particle-sliders-core` **0.6.0** at
`9be81c1ccfc8ad87eac3efa93fd5a16d0d508bb5`, built on shared CDF
`cdfb80a76b882b4b73861b12f555263bd64ffeb7`, with ParticleGAN **0.9.1**.
This is a new source lineage. Earlier PG 0.9.0, PG42 and intermediate
PG 0.9.1 receipts remain historical.

## Exact local package inputs

Qualification installed these wheels with `--no-index --no-deps` into
`/ml2/codex-storage/future-workspaces/2026-10-17/task-12/venv-pg091`, a
separate environment using the native Music3 runtime's system packages:

| Artifact | Identity |
|---|---|
| Shared wheel | `/ml2/codex-storage/future-workspaces/2026-10-17/task-14/releases/9be81c1ccfc8ad87eac3efa93fd5a16d0d508bb5/dist/particle_sliders_core-0.6.0-py3-none-any.whl` |
| Shared SHA-256 | `60629b05298559340807084d278bb256781bf0421c6d928c23e57199e2b37dc8` |
| Official PG wheel | `/ml2/codex-storage/future-workspaces/2026-10-17/task-14/pg091-release/particlegan-0.9.1-py3-none-any.whl` |
| PG SHA-256 | `b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec` |

Runtime origins are recorded in
[`installed-api-receipt.json`](evidence/pg091/installed-api-receipt.json):
native Torch `2.11.0+cu128`, Transformers `5.15.0`, installed public core
modules and the 49 actual PG release source hashes. No active training
environment or checkout was changed.

`core.lock.json` records the exact source identity, wheel digest and
installed-file contract. The loader verifies distribution ownership,
locked code and metadata, installation provenance and imported module
origins. It rejects editable installs, foreign cached modules and checkout
injection. It never installs or repairs a dependency. Old
`--shared_checkout` and `--core_checkout` options are refused.

Wheel archives retain raw file equality. A verified immutable public Git
installation may vary only the canonical `WHEEL` generator header; the
remaining metadata bytes and all source/dependency metadata hashes stay
locked. This handles the permitted build-backend version without claiming
that an unpublished Git requirement has been fetched or built successfully.

**Public dependency resolution is blocked.** `requirements.txt` contains
the intended immutable source pin, but the package owner's push of this
exact head to `HyperGAN/particle-sliders`,
`refs/heads/audit/shared-audio-cli-20261017`, was rejected by automatic
approval review because authorization named a different embedded branch.
The push did not execute. The parent must obtain exact repository/ref/
payload authorization; no alternate ref, uploaded wheel or source mirror
is a substitute. Public VCS fetch/build acceptance is pending publication.
Local wheel qualification does not establish that acceptance or Ada approval.

## Qualified CPU cohort

The final package passed **80 product tests** and **49 focused shared
regressions**. Product subprocesses use `python -I`, temporary working
directories, installed package origins, and a finder that refuses historical
`conceptmod`, `analysis` and worker imports. Only the product is exposed.
The wide profile separately exposes the exact reviewed PG source overlay;
it never adds a shared implementation checkout.

```bash
PYTHON=/ml2/codex-storage/future-workspaces/2026-10-17/task-12/venv-pg091/bin/python
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$PYTHON" scripts/train_music3.py --dummy --steps 3 \
  --save_dir /tmp/music3-pg091-full3
"$PYTHON" scripts/train_lm_slider_music3.py --dummy --steps 1 \
  --save_dir /tmp/music3-pg091-first1
"$PYTHON" scripts/train_lora_music3_particle.py --dummy --steps 3 \
  --resume /tmp/music3-pg091-first1/checkpoint.pt \
  --save_dir /tmp/music3-pg091-resumed3
"$PYTHON" scripts/infer_music3.py --dummy \
  --weights /tmp/music3-pg091-resumed3/adapter-export.pt \
  --scales 0,0.5,1 --out_dir /tmp/music3-pg091-probes

PARTICLE_SLIDERS_CONTINUOUS_PROFILE=/ml2/codex-storage/future-workspaces/2026-10-17/task-14/pg091-release/continuous \
  "$PYTHON" -m pytest -q -p no:cacheprovider \
  --junitxml=evidence/pg091/product-installed-tests.xml
"$PYTHON" -I evidence/pg091/verify_installed.py
"$PYTHON" -I evidence/pg091/verify_native_plan.py
```

Use new output directories for a new run. `--steps` is the absolute
completed-step target, default 3. The CPU recipe has a fixed horizon of 4,
so the resumed example performs updates 2 and 3. Only a full new-lineage
checkpoint resumes training. Actual old PG 0.9.0 checkpoint/export files
were rejected without receiver-state, RNG or old-file changes.

Both `bispf` and `bispf_optimized` passed exact uninterrupted 3 versus
fresh-process 1+2 continuation across the whole qualification packet:
model/critic, optimizer, controller, prior, RNG, metrics, probes and export.
The width-32 fixture executes genuine random native Qwen3 with vocabulary
151670 and audio-start token 151669. A separate width-4096 random geometry
fixture passed both presets under explicit `alz-continuous-v1`/`continuous`;
its small synthetic vocabulary is not native tokenization evidence.
Neither fixture loads pretrained weights or renders audio.

The suite also verifies bank ownership, response gradients and updates,
frozen base weights, neutral-student/positive-teacher conditions, public
GAN/prior/penalty metric decomposition, exact scale-0 replay, export reload,
probe/factory CPU RNG preservation and no CUDA seed dispatch. Invalid prior
metric history, wrong-family exports, invalid scales, unsupported hosts
and malformed native exports reject before live mutation or output.

All construction, losses, updates, export/probe handling and serialization
run through public `particle_sliders.pg09_audio_cli`. Product entry points
only validate Music3 arguments and invoke its dispatcher.

| Shared artifact | Meaning |
|---|---|
| `checkpoint.pt` | Full new-lineage training checkpoint |
| `adapter-export.pt` | Shared inference export; not a conventional LoRA |
| `qualification.pt` | Complete shared qualification packet |
| `receipt.json`, `progress.json` | Packaged workflow receipts |
| `hidden-probes.pt`, `probe-card.json` | Reloaded neutral-token hidden probes |
| `product-integration.json` | Product's installed-source receipt |

[`final-shared-narrow-tests.xml`](evidence/pg091/final-shared-narrow-tests.xml)
records the 49 focused optimizer-history, CDF-clock and transactional
native-loader cases. Those fixtures were extracted from exact CDF into a
temporary test directory and imported the installed wheel, without a
shared source checkout on the import path. Counts and file digests are
collected in [`acceptance-receipt.json`](evidence/pg091/acceptance-receipt.json).

## Native plan only

The base model ID is `MiniMaxAI/MiniMax-Music3`. The read-only plan receipt
[`native-plan.json`](evidence/pg091/native-plan.json) used actual staged
revision `bd348f9c49ea3c1b39f33ace3436f8fad435f24e`, verified ten LM/tokenizer
files, four training rows and two held-out rows. Guards forbade native
model/fixture allocation; no tensors were deserialized. It records
`PLAN_ONLY_NOT_ADMITTED`, `dispatch_allowed=false` and output absent.

For staged inputs, the product exposes this plan interface:

```bash
"$PYTHON" scripts/train_music3.py --live \
  --model_dir /path/to/staged/MiniMax-Music3 \
  --manifest /path/to/verified-assets.json \
  --prompts_file /path/to/four-train-rows.yaml \
  --heldout_prompts /path/to/two-heldout-rows.yaml \
  --save_dir /mnt/nvme/sliders/qualification/runs/music3-pg091-new-reviewed-run \
  --steps 3 --total_steps 20 \
  --pg_source_profile alz-continuous-v1 \
  --penalty_input_domain continuous
```

The explicit PG profile must already be separately reviewed and staged.
The release/auto wide-response guard stays closed. No CLI flag creates a
GPU lease or admits a process. A supplied native `--resume` is explicitly
refused pending a qualified new-cohort producer; no old checkpoint is rebound.

Remaining gates are exact-head Ada/package publication review, public
installation acceptance, coordinator-owned guardian/card/lease and fresh
resources, native token/history producer qualification, pretrained 3 versus
fresh-process 1+2 and export replay, separately admitted native acoustic
sampling, matched base/off/0.5/1 and positive-caption renders, then listening
review for concept direction, lyrics, voice and artifacts. Hidden changes
or waveform differences alone do not qualify acoustic quality. No GPU job,
resident service or model publication is dispatched here.

## Historical evidence

PG 0.9.0 source-checkout receipts remain under `evidence/derivative-audit/`;
`legacy-shared.lock.json` preserves their identity. Intermediate 4be and670
XML receipts under `evidence/pg091/` are labeled historical in the final
acceptance receipt. They do not qualify this final package or migrate old
checkpoints.

The September 16 `uni16-fresh-selected-v2` conventional LoRAs, ComfyUI
exports and recordings remain at
[ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders).
Their selected steps and paths remain in `configs/music3/sample-cards.json`.
Use [COMFYUI.md](COMFYUI.md) for those historical conventional files only;
`adapter-export.pt` is not accepted by the historical LoRA converter.
