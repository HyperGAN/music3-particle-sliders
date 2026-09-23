# Music 3 particle sliders

Concept sliders for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).
The shared game is [`winning_formulation()`](FORMULATION.md) from
[particle-sliders-core](https://github.com/HyperGAN/particle-sliders/tree/4340e28bed388d50800c469525b460a108091da0/packages/particle-sliders-core).
This repo owns the Hub ids, Comfy notes, host modules, prompt cards, and
train/infer entrypoints.

**[Weights](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)** ·
**[Space](https://huggingface.co/spaces/ntc-ai/minimax-music3-concept-sliders)** ·
**[YuE2 weights](https://huggingface.co/ntc-ai/yue2-concept-sliders)** (sibling, not this product)

The September 16, 2026 release (`uni16-fresh-selected-v2`, policy
`quality-later-v2`) has 16 voice and genre controls. Each listen pair keeps
the neutral caption, lyrics, and seed **1709** fixed and changes strength
from 0 to 1. Native files are under `weights/`, ComfyUI files under
`comfyui/`, and Off/On clips under `samples/`. The reward adapter under
`reward/` failed preservation checks and is not the product default.

| Control | Step | Native | ComfyUI |
|---|---:|---|---|
| Female | 1000 | `weights/uni16-fresh-selected-v2/female/` | `comfyui/uni16-fresh-selected-v2/female_step1000_comfyui.safetensors` |
| Male | 3400 | `weights/uni16-fresh-selected-v2/male/` | `comfyui/uni16-fresh-selected-v2/male_step3400_comfyui.safetensors` |
| Lo-fi | 3000 | `weights/uni16-fresh-selected-v2/lofi/` | `comfyui/uni16-fresh-selected-v2/lofi_step3000_comfyui.safetensors` |
| Pop | 2000 | `weights/uni16-fresh-selected-v2/pop/` | `comfyui/uni16-fresh-selected-v2/pop_step2000_comfyui.safetensors` |
| Hip-Hop | 3000 | `weights/uni16-fresh-selected-v2/hiphop/` | `comfyui/uni16-fresh-selected-v2/hiphop_step3000_comfyui.safetensors` |
| R&B | 3400 | `weights/uni16-fresh-selected-v2/rnb/` | `comfyui/uni16-fresh-selected-v2/rnb_step3400_comfyui.safetensors` |
| Indie Rock | 1000 | `weights/uni16-fresh-selected-v2/indie-rock/` | `comfyui/uni16-fresh-selected-v2/indie-rock_step1000_comfyui.safetensors` |
| Pop Punk | 3400 | `weights/uni16-fresh-selected-v2/pop-punk/` | `comfyui/uni16-fresh-selected-v2/pop-punk_step3400_comfyui.safetensors` |
| Metal | 2000 | `weights/uni16-fresh-selected-v2/metal/` | `comfyui/uni16-fresh-selected-v2/metal_step2000_comfyui.safetensors` |
| Country | 1000 | `weights/uni16-fresh-selected-v2/country/` | `comfyui/uni16-fresh-selected-v2/country_step1000_comfyui.safetensors` |
| Acoustic Folk | 2000 | `weights/uni16-fresh-selected-v2/acoustic-folk/` | `comfyui/uni16-fresh-selected-v2/acoustic-folk_step2000_comfyui.safetensors` |
| House | 1000 | `weights/uni16-fresh-selected-v2/house/` | `comfyui/uni16-fresh-selected-v2/house_step1000_comfyui.safetensors` |
| Disco Funk | 2000 | `weights/uni16-fresh-selected-v2/disco-funk/` | `comfyui/uni16-fresh-selected-v2/disco-funk_step2000_comfyui.safetensors` |
| K-pop | 2000 | `weights/uni16-fresh-selected-v2/kpop/` | `comfyui/uni16-fresh-selected-v2/kpop_step2000_comfyui.safetensors` |
| Reggaeton | 1000 | `weights/uni16-fresh-selected-v2/reggaeton/` | `comfyui/uni16-fresh-selected-v2/reggaeton_step1000_comfyui.safetensors` |
| Afrobeats | 1000 | `weights/uni16-fresh-selected-v2/afrobeats/` | `comfyui/uni16-fresh-selected-v2/afrobeats_step1000_comfyui.safetensors` |

Full paths are in `configs/music3/sample-cards.json`. Loading steps are in
[COMFYUI.md](COMFYUI.md).

## Install

```bash
python -m pip install -r requirements.txt
```

The requirements pin is the only copy of the game:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@4340e28bed388d50800c469525b460a108091da0#subdirectory=packages/particle-sliders-core
```

## Train and infer

```bash
python scripts/train_music3.py --dummy
python scripts/train_lm_slider_music3.py --dummy
python scripts/train_lora_music3.py --dummy --targets full
python scripts/train_lora_music3_particle.py --dummy
python scripts/train_encoder_music3.py --dummy
python scripts/infer_music3.py --dummy
python scripts/train_music3.py --print_card
```

`--dummy` runs the stamp on CPU (four steps, synthetic prompt states, no Hub
download). Four steps is enough for the stamp's lazy gradient cap to fire once. Hosts:

| Script | Host | Modules |
|---|---|---|
| `train_lm_slider_music3.py`, `train_lora_music3_particle.py`, `train_music3.py` | language model | `Qwen3Attention` (`lora_te`) |
| `train_lora_music3.py` | flow transformer | `full` (222) or `--targets attn` (144) |
| `train_encoder_music3.py` | condition encoder | `MiniMaxMusic3ConditionEncoder` |

Prompt cards: `configs/music3/prompts-particle.yaml` (train) and
`configs/music3/prompts-music3.yaml` (historical one-row energy listen).

## What stayed behind

Research trainers in HyperGAN/particle-sliders are untouched:

- `conceptmod/textsliders/train_lora_music3.py`
- `conceptmod/textsliders/train_lora_music3_particle.py`
- `conceptmod/textsliders/train_lm_slider_music3.py`
- `conceptmod/textsliders/train_encoder_music3.py`
- `conceptmod/textsliders/infer_music3.py`
- `conceptmod/textsliders/music3_particle_bridge.py`

Formulation toys stay in HyperGAN/conceptmod. A one-line pointer from the
particle-sliders README to this repo is a follow-up; see [REPRODUCE.md](REPRODUCE.md).
