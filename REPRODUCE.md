# Reproduce the Music 3 shared integration

Use the separately qualified runtime and immutable local sources. Neither
an older public core pin nor a guessed git requirement supplies this
integration.

```bash
export PARTICLE_SLIDERS_SHARED_CHECKOUT=/mnt/nvme/sliders/audio-adapters-pg09-20261017
export PARTICLE_SLIDERS_CORE_CHECKOUT=/mnt/nvme/sliders/audio-core-pin-86fc4e0-20261017
PYTHON=/mnt/nvme/sliders/qualification/envs/pg090/bin/python
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Required source identities:

- Shared audio worker: `9262a930cd351459e93ac33cc542fa379c12fe52`.
- Shared core: `3e94b13eb5d6cd1ceee3f018f019a68603f35696`.
- ParticleGAN distribution: `0.9.0`, on the prequalified source/runtime path.

`--shared_checkout` and `--core_checkout` can supply the same checkout
paths explicitly. The product validates its integration sources. Do not
modify sources pinned by a running training process.

## Bounded CPU route

Output training and probe directories must not already exist. Use new names if these
examples have previously been run.

```bash
"$PYTHON" scripts/train_music3.py --dummy --steps 3 \
  --save_dir /tmp/music3-full3
"$PYTHON" scripts/train_lm_slider_music3.py --dummy --steps 1 \
  --save_dir /tmp/music3-first1
"$PYTHON" scripts/train_lora_music3_particle.py --dummy --steps 3 \
  --resume /tmp/music3-first1/checkpoint.pt \
  --save_dir /tmp/music3-resumed3
"$PYTHON" scripts/infer_music3.py --dummy \
  --weights /tmp/music3-resumed3/adapter-export.pt \
  --scales 0,0.5,1 --out_dir /tmp/music3-probes
"$PYTHON" -m pytest
```

`--steps` is an absolute completed-step target, default 3. Each CPU run uses
the same total recipe horizon of 4 updates, so the resumed example performs
updates 2 and 3. It does not restart or silently extend the recipe. The
shared worker owns all state serialization and restoration.

The CPU model is a random native width-32 Qwen3 model, not a downloaded
pretrained model. Qualification must show the actual bank affects the
response and changes during GAN training, frozen host weights stay fixed,
scale 0 exactly replays the frozen neutral state, and shared export reload
replays heldout responses. Full and resumed execution must compare the
complete model/critic, optimizers, controller, prior, RNG, metrics and
exports. None of this is acoustic or pretrained quality evidence.

Outputs:

| File | Meaning |
|---|---|
| `checkpoint.pt` | Full resumable shared training checkpoint |
| `adapter-export.pt` | Shared inference export, not a conventional LoRA |
| `qualification.pt` | Shared qualification packet and probes |
| `progress.json` | Shared worker progress receipt |
| `product-integration.json` | Product receipt and locked source origins |
| `hidden-probes.pt` and `probe-card.json` | Reloaded neutral-token hidden responses at requested scales |

Negative, nonfinite and greater-than-one probe scales are rejected.
Transformer and encoder train entry points are rejected until a native
shared-game binding is qualified. Historical YAML loss settings do not
select a fallback objective.

## Native plan only

The actual base model ID is `MiniMaxAI/MiniMax-Music3`. A plan for a staged
native model can be prepared without launching CUDA:

```bash
"$PYTHON" scripts/train_music3.py --live \
  --model_dir /path/to/staged/MiniMax-Music3 \
  --manifest /path/to/verified-assets.json \
  --prompts_file /path/to/four-train-rows.yaml \
  --heldout_prompts /path/to/two-heldout-rows.yaml \
  --save_dir /mnt/nvme/sliders/qualification/runs/music3-new-reviewed-run \
  --steps 3 --total_steps 20 \
  --pg_source_profile alz-continuous-v1 \
  --penalty_input_domain continuous
```

This emits a plan for the shared bounded native worker, including the exact
source-only PYTHONPATH and native runtime identity. Planned output must be a
new directory under /mnt/nvme/sliders. The coordinator
adds the physical-GPU lease, guardian, explicit admission and fresh-memory
bounds before any dispatch. An emitted command is not admission. The
width-4096 native response requires the exact reviewed continuous source
profile; released PG 0.9's default width guard remains active.

Pending native gates are the separately admitted pretrained smoke, exact
fresh-process continuation and export replay, continuation to any approved
budget, matched native base/off/0.5/1 and positive-caption acoustic renders,
and listening review for concept direction, lyrics, voice and artifacts.
Hidden-state changes or waveform differences alone do not qualify quality.
The product CLI does not dispatch legacy trainers, change resident services
or publish a model.

## Historical release metadata

The September 16 `uni16-fresh-selected-v2` native LoRAs, ComfyUI exports and
recordings remain at
[ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders).
Their selected steps and paths remain in `configs/music3/sample-cards.json`.
They predate this PG 0.9 integration and cannot substitute for its native
qualification. Use [COMFYUI.md](COMFYUI.md) only for those conventional LoRA
files; do not feed `adapter-export.pt` to the historical converter.
