# Music 3 particle sliders

A separate [response-session source proposal](docs/response-session-proposal.md)
registers the owned Music3 factory and data-independent response inference on
reviewed core0.8.2 / PG0.9.1. **22 selected installed CPU checks passed**;
see [publication and qualification status](PUBLICATION-STATUS.md). Native wide
responses and acoustic rendering remain gated.
The historical CLI and its core0.6.0 lock below remain unchanged and require
their original environment; the two environments must remain separate.

Music 3 product integration for `MiniMaxAI/MiniMax-Music3`. This branch
moves training, resume, export and probing to the installed public
`particle-sliders-core` **0.6.0** API with ParticleGAN **0.9.1**. Product
entry points call `particle_sliders.pg09_audio_cli`; the package owns all
GAN losses, updates, native loading and workflow orchestration.

The exact final local wheel passed **80 product CPU tests and 49 focused
shared-state tests**. Native pretrained training, acoustic rendering and
listening review remain pending. The package proposal and this derivative
still require exact-head Ada review and parent integration.

## Installed package contract

`core.lock.json` binds proposed package commit
`9be81c1ccfc8ad87eac3efa93fd5a16d0d508bb5`, wheel SHA-256
`60629b05298559340807084d278bb256781bf0421c6d928c23e57199e2b37dc8`,
and 39 installed-file hashes. Qualification used that exact local wheel.
`requirements.txt` records the intended immutable Git dependency, but
public Git resolution is **blocked**: automatic approval review rejected
the package owner's push to `HyperGAN/particle-sliders`,
`refs/heads/audit/shared-audio-cli-20261017`, because the authorization
named a different embedded branch. That push did not execute. The parent
must resolve the exact publication authorization before this dependency
can be fetched publicly; no alternate publication is implied.

Use a separate reviewed PG 0.9.1 environment with the locked core wheel.
The product loader checks the installed distribution, stable file hashes,
installation provenance and imported module origins. It does not add a
shared checkout to Python's search path or inject source-path environment
variables. Historical checkout arguments and editable/source-tree installs
are refused. Installing a version number alone does not establish the
locked file identity.

Exact wheel installs retain all raw file checks. For an installation with
verified immutable public Git provenance, only the build-generated
`WHEEL` generator line may vary; its other bytes and every source and
dependency-metadata hash remain locked. Public fetch/build acceptance is
still a separate pending gate.

The public workflow API owns `make_audio_game`, `run_training`,
`probe_export` and its train/probe/plan command routing. The product owns
Music 3 metadata, host selection and CLI argument translation. No loss,
optimizer, adapter implementation or training loop is copied here.

## CPU train, resume and probe

The commands below use the qualified new-cohort interface. Use the exact
local wheel in a separate environment and fresh output directories:

```bash
PYTHON=/path/to/reviewed-pg091-environment/bin/python
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/train_music3.py \
  --dummy --steps 1 --save_dir /tmp/music3-pg091-first
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/train_lm_slider_music3.py \
  --dummy --steps 3 --resume /tmp/music3-pg091-first/checkpoint.pt \
  --save_dir /tmp/music3-pg091-resumed
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/infer_music3.py \
  --dummy --weights /tmp/music3-pg091-resumed/adapter-export.pt \
  --scales 0,0.5,1 --out_dir /tmp/music3-pg091-probes
```

`--steps` is the absolute completed-step target, default 3. The CPU recipe
has a fixed total budget of 4 updates. Resume uses the full new-lineage
`checkpoint.pt`; old PG 0.9.0 checkpoints and exports are not migrated.
The shared workflow owns `checkpoint.pt`, `adapter-export.pt`,
`qualification.pt` and `progress.json`, and the probe output contract.
Expected probe outputs are `hidden-probes.pt` and `probe-card.json`.

The CPU fixture executes a genuine random native Qwen3 model with hidden
width 32 and Music 3 audio-start token ID **151669**. Both supported presets
passed response gradients to the actual particle bank, frozen-base checks,
export reload and exact fresh-process 3 versus 1+2 resume. A separate
width-4096 random geometry fixture passed both presets using the explicit
reviewed continuous profile; it uses a small synthetic vocabulary and
does not qualify native tokenization or pretrained behavior.

Probes use neutral tokens at finite scales in [0,1], with the frozen
positive-caption state as teacher. Negative, nonfinite and greater-than-one
scales are refused. Exact scale-0 replay is a required qualification check.
Hidden responses are not tokens, latents or generated audio.

| Entry point | Intended packaged route |
|---|---|
| `train_music3.py`, `train_lm_slider_music3.py`, `train_lora_music3_particle.py` | LM/particle CPU train or native plan |
| `train_lora_music3.py` | Transformer host refused |
| `train_encoder_music3.py` | Encoder host refused |
| `infer_music3.py` | Shared-export CPU probes or historical release card |

`--live` emits a native plan through the public package. It does not launch
GPU work. The coordinator owns the GPU lease, guardian, admission and memory
bounds. [REPRODUCE.md](REPRODUCE.md) lists the final-wheel CPU and native
gates; [FORMULATION.md](FORMULATION.md) describes the response contract.

## Historical published LoRA release

**[Weights](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)** ·
**[Space](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders)** ·
**[YuE2 weights](https://huggingface.co/ntc-ai/yue2-concept-sliders)** (sibling)

The September 16, 2026 release `uni16-fresh-selected-v2`, selected by
`quality-later-v2`, has 16 voice and genre controls. Its conventional native
LoRAs, ComfyUI exports and recordings are historical artifacts. They are
not compatible PG 0.9.1 checkpoints and do not qualify the new cohort.
Published listen pairs keep neutral caption, lyrics and seed 1709 fixed,
comparing strengths 0 and 1. The reward adapter under `reward/` failed
preservation checks and is not the product default.

| Control | Selected step |
|---|---:|
| Female | 1000 |
| Male | 3400 |
| Lo-fi | 3000 |
| Pop | 2000 |
| Hip-Hop | 3000 |
| R&B | 3400 |
| Indie Rock | 1000 |
| Pop Punk | 3400 |
| Metal | 2000 |
| Country | 1000 |
| Acoustic Folk | 2000 |
| House | 1000 |
| Disco Funk | 2000 |
| K-pop | 2000 |
| Reggaeton | 1000 |
| Afrobeats | 1000 |

Exact historical paths remain in `configs/music3/sample-cards.json` and
the release card. [COMFYUI.md](COMFYUI.md) covers those conventional LoRAs.
The transformer/encoder YAML cards and signed energy demo remain archived
recipe information. Earlier PG 0.9.0 qualification evidence is retained as
history and cannot be reused as a PG 0.9.1 test receipt.
