# Music3 HyperGAN/HNDL candidate

This separately packaged CPU integration uses core0.12 at
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, official PG0.9.4, actual clean
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7` and HNDL0.10.0.
It binds the existing shared HNDL routed projection to the genuine Qwen3
attention q/k/v/o linears, using one owned128x4 bank, rank8 and alpha8.
The AR readout retains the existing linear/SiLU/linear architecture, executed
as an HNDL graph through HyperGAN's verified builder. The shared
`ConditionalPG09` owns all losses, optimizers, updates, prior and checkpointing.

Install only in a separate Python3.11–3.14 environment. The exact core candidate
is currently local and not published; the VCS dependency records the intended
immutable cohort and cannot yet be assumed publicly fetchable. Qualification
uses the owner's exact installed local wheel and source closure in
`music3_hndl/core.lock.json`. Build this candidate with
`python -m build --no-isolation cohorts/hndl`, then install its wheel in that
already qualified environment. This candidate does not install the historical
Music3 package or change its core0.8.2/PG0.9.1 pin.

```python
from music3_hndl import build_cpu_game

with build_cpu_game() as run:
    run.game.step()  # actual shared ParticleGAN update
    checkpoint = run.game.state_dict()
```

`build_cpu_game` exposes a bounded random CPU width32 Qwen3 fixture, 1..4 steps,
and explicit seed/checkpointing/preset. Native width4096 is rejected by the
shared continuous-domain admission gate before model allocation. No pretrained
loader, model download, AR sampler, decoder or acoustic output is exposed.
The fixture's vocabulary is synthetic; it does not qualify native tokenization.

This is a fresh checkpoint cohort. Keep historical environment, source locks
and checkpoint identities. Old routed exports are not ordinary LoRAs; this
candidate does not claim ordinary-LoRA distillation or ComfyUI qualification.
Matched-weight parity, particle gradients, exact-zero behavior and complete
shared checkpoint restore must pass on installed packages before adoption.
No package/model release, deployment or GPU work is triggered by this source.
