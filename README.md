# Music 3 particle sliders

Music 3 product integration for `MiniMaxAI/MiniMax-Music3`. This repository
owns the CLI, product metadata and prompt cards. Training and checkpoint
handling run through the shared `particle_sliders.ConditionalPG09` game with
ParticleGAN **0.9.0**; this product contains no alternate GAN loop.

The executable route currently qualifies the **LM/particle host on CPU**,
using a random native Qwen3 model with hidden width 32. It trains the actual
hooked routed adapter and particle bank. Transformer and condition-encoder
training are refused until their native bindings are qualified. Native
pretrained training, acoustic rendering and listening review remain pending.

## Local integration dependencies

Use the separately qualified PG 0.9 runtime and these exact source checkouts:

| Source | Required commit | Prepared local checkout |
|---|---|---|
| Shared audio adapter/worker | `9262a930cd351459e93ac33cc542fa379c12fe52` | `/mnt/nvme/sliders/audio-adapters-pg09-20261017` |
| Shared particle-sliders core | `3e94b13eb5d6cd1ceee3f018f019a68603f35696` | `/mnt/nvme/sliders/audio-core-pin-86fc4e0-20261017` |

```bash
export PARTICLE_SLIDERS_SHARED_CHECKOUT=/mnt/nvme/sliders/audio-adapters-pg09-20261017
export PARTICLE_SLIDERS_CORE_CHECKOUT=/mnt/nvme/sliders/audio-core-pin-86fc4e0-20261017
PYTHON=/mnt/nvme/sliders/qualification/envs/pg090/bin/python
```

The CLIs also accept `--shared_checkout` and `--core_checkout`. The source
pins are an integration dependency, recorded in `shared.lock.json` and
`core.lock.json`; they are
not an installable public-core release claim. `requirements.txt` lists
PyYAML, pytest and `particlegan==0.9.0`. Use the prequalified runtime rather
than replacing a native environment or installing an older core git pin.
No command downloads model weights.

## CPU train, resume and probe

Each output directory for training or probes must be new. `--steps` is the absolute
completed-step target, default 3; the CPU recipe has an immutable total
budget of 4 updates. A full shared checkpoint preserves the model, critic,
optimizers, prior, controller and RNG state for continuation.

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/train_music3.py \
  --dummy --steps 1 --save_dir /tmp/music3-first
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/train_lm_slider_music3.py \
  --dummy --steps 3 --resume /tmp/music3-first/checkpoint.pt \
  --save_dir /tmp/music3-resumed
CUDA_VISIBLE_DEVICES='' "$PYTHON" scripts/infer_music3.py \
  --dummy --weights /tmp/music3-resumed/adapter-export.pt \
  --scales 0,0.5,1 --out_dir /tmp/music3-probes
```

Training delegates to the bounded shared CPU worker and writes
`checkpoint.pt`, `adapter-export.pt`, `qualification.pt` and `progress.json`,
with a product source receipt in `product-integration.json`.
Inference reloads the shared adapter export and writes `hidden-probes.pt`
and `probe-card.json`. These are hidden-state responses, with neutral tokens at every
slider scale and the frozen positive-caption state as the teacher. They
are not generated audio. Negative, nonfinite and greater-than-one scales
are refused; scale 0 is the exact frozen neutral response.

| Entry point | Current behavior |
|---|---|
| `train_music3.py`, `train_lm_slider_music3.py`, `train_lora_music3_particle.py` | Shared LM/particle CPU route, or native execution plan |
| `train_lora_music3.py` | Transformer host refused |
| `train_encoder_music3.py` | Encoder host refused |
| `infer_music3.py` | Shared-export CPU hidden probes; historical release card |

`--live` prepares a plan for the root-owned shared native worker. It does
not initialize CUDA or launch that worker. The coordinator supplies the
separate GPU lease, guardian, admission and memory bounds. See
[REPRODUCE.md](REPRODUCE.md) for the plan inputs and remaining gates, and
[FORMULATION.md](FORMULATION.md) for the actual objective.

## Historical published LoRA release

**[Weights](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)** ·
**[Space](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders)** ·
**[YuE2 weights](https://huggingface.co/ntc-ai/yue2-concept-sliders)** (sibling)

The September 16, 2026 release `uni16-fresh-selected-v2`, selected by
`quality-later-v2`, has 16 voice and genre controls. Its native LoRA files,
ComfyUI files and recordings are historical artifacts; the new PG 0.9
worker does not train, load or qualify those files. Published listen pairs
keep neutral caption, lyrics and seed 1709 fixed, comparing strengths 0
and 1. The reward adapter under `reward/` failed preservation checks and
is not the product default.

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

Exact paths remain in `configs/music3/sample-cards.json` and the release
card. [COMFYUI.md](COMFYUI.md) describes their historical loading contract.
The transformer and encoder YAML cards and the signed energy demo are
archived recipe information, not executable PG 0.9 training or scale gates.
