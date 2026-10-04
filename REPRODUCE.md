# Reproduce

CPU smoke does not download Hub weights and does not need a Music 3 checkout.

```bash
python -m pip install -r requirements.txt
python scripts/train_music3.py --dummy
python scripts/infer_music3.py --dummy
python scripts/train_lora_music3.py --help
python scripts/train_lm_slider_music3.py --help
python scripts/train_encoder_music3.py --help
python scripts/train_lora_music3_particle.py --help
pytest
```

`--dummy` runs four winning-formulation steps on synthetic prompt states and
writes `models/music3-slider/<name>_last.json`. The bridge and critic modules
recorded there must be `particle_sliders.reference`. The regularizer module
must be `particlegan`, not a class defined in this repo.

## Live weights

Base model: [MiniMaxAI/MiniMax-Music-3](https://huggingface.co/MiniMaxAI/MiniMax-Music-3).

Adapter release:
[ntc-ai/minimax-music3-concept-sliders](https://huggingface.co/ntc-ai/minimax-music3-concept-sliders)
(`weights/uni16-fresh-selected-v2/`, `comfyui/uni16-fresh-selected-v2/`,
`samples/uni16-fresh-selected-v2/`).

```bash
python scripts/train_music3.py --live --model_dir /path/to/MiniMax-Music-3
```

`--live` reads that local checkout and binds `stamp.bridge()` to each
`nn.Linear` inside the host class. The language-model entry
(`language_model/` plus `tokenizer/`) trains `Qwen3Attention` linears
(`q_proj`, `k_proj`, `v_proj`, `o_proj`). The feature-space step is
`FormulationGame` from the pinned core, on the last `<|audio_start|>`
token. It does not download the 8B weights. `--dummy` stays the CPU stamp
on synthetic prompt states and does not load those linears or write a
bridge checkpoint. `--dummy` is not the live path.

The transformer host (`transformer/`, `MiniMaxMusic3Attention`, plus full
blocks when `--targets full`) uses the same attach, then calls
`MiniMaxMusic3Transformer1DModel.forward(hidden_states, timestep,
encoder_hidden_states)` on the loaded diffusers module and steps
`FormulationGame` on the pooled velocity. That method is not copied from
`conceptmod/textsliders`. The call needs local tensors next to the checkout:

- `surface/transformer/latents.pt` — Flow-VAE latents `[batch, in_channels, length]`
- `surface/transformer/neutral.pt` and `positive.pt` — frame-aligned
  `encoder_hidden_states` `[batch, length, condition_dim]`

A checkout without those files binds the linears, then raises. The error
names the missing tensors. The historical producer is
`conceptmod/textsliders/train_lora_music3.py` (`_load_ar_pipeline`,
`_encode_condition`, `_cache_or_encode`). It also runs the nmse slider loss
and `LoRANetwork`, so this repo does not copy it and does not invent latents.

The condition encoder (`condition_encoder/`,
`MiniMaxMusic3ConditionEncoder`) has the same kind of forward:
`MiniMaxMusic3ConditionEncoder.forward(hidden_states)` in diffusers, called
from this product when `surface/encoder/neutral.pt` and `positive.pt` hold
AR frame hiddens. On the published class `proj` is `Conv1d`, not
`nn.Linear`, so `stamp.bridge()` finds nothing to bind and `--live` stops
there. Conv1d targeting from `conceptmod/textsliders/train_encoder_music3.py`
(`cache_real_hiddens` and the mse loop) was not copied.

No run in this repo loaded the 8B MiniMax-Music-3 weights. Do not point the
loader at a second copy of the game inside a particle-sliders checkout.

`--lm_target v9`, `--loss nmse`, and encoder rank 64 are rejected. Those
recipes remain in HyperGAN/particle-sliders `conceptmod/textsliders/`.

## Follow-up in particle-sliders

This task does not open a pull request on HyperGAN/particle-sliders and does
not delete the Music 3 trainers there. A one-line README pointer from that
repo to `https://github.com/HyperGAN/music3-particle-sliders` is still worth
adding:

```markdown
Music 3 product: [music3-particle-sliders](https://github.com/HyperGAN/music3-particle-sliders).
```
