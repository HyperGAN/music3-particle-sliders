# Formulation

`music3-particle-sliders` owns the Music 3 integration: the base id
`MiniMaxAI/MiniMax-Music-3`, the adapter release
`ntc-ai/minimax-music3-concept-sliders`, Comfy key notes, which host a
slider attaches to (`Qwen3Attention`, `MiniMaxMusic3Attention`, or
`MiniMaxMusic3ConditionEncoder`), the prompt cards under `configs/music3/`,
and the train/infer entrypoints in `music3/` and `scripts/`.

The game is not implemented in this repo. Train calls:

```python
from particle_sliders import winning_formulation

stamp = winning_formulation()
stamp.require(declared)
```

`declared` starts as `stamp.as_dict()`. Music 3 may override model-surface
keys only (`g_lr`, `d_lr`, `particle_lr`, `adv_batch`, and the other names on
`stamp.model_surface_keys`). The published particle bridge used `adv_batch`
64; that override is the default here. Architecture, caps, particle count,
teacher, and auxiliary weights stay on the stamp.

Today that stamp is gmix architecture plus the provisional overlay
`particle-gmix-1600-v2` (`formulation_provisional` is true until
[ParticleGAN #38](https://github.com/255BITS/ParticleGAN/pull/38) crowns a
full live leaderboard winner: 9 toys and all 29 bounds). This repo picks up
a new overlay by bumping the pin in `requirements.txt` and `core.lock.json`.
It does not fork the knobs.

The pin is:

```text
particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@a119ca1ecd3d5d6c437065839d22739b04f2f4d8#subdirectory=packages/particle-sliders-core
```

`stamp.bridge()`, `stamp.critic()`, `stamp.regularizer()`, `stamp.losses()`,
and `stamp.noise_std_at()` are the routed adapter, global-mix critic,
ParticleGAN gradient cap, paired losses plus particle VIC, and noise curve.
This repo does not copy `RoutedMLP`, the gmix critic, `GradRegularizer`,
`locked_shared`, or ParticleGAN excerpts.

Formulation toys and the `--lm_target` menu (`v9`, `faithful_sub_e`, lyric
holds, and the rest) stay in
[HyperGAN/conceptmod](https://github.com/HyperGAN/conceptmod) and in the
research trainers under
[HyperGAN/particle-sliders](https://github.com/HyperGAN/particle-sliders)
`conceptmod/textsliders/`. Those files are not deleted by this product. The
historical transformer dust recipe (`--loss nmse`, `--targets full`, lr
`2e-3`) and the encoder MSE loop are the same kind of research surface: the
host names moved here, the old losses did not.

`locked_shared` is an endpoint recipe inside the core. It is not the Music 3
product game. Do not call it from this repo.
