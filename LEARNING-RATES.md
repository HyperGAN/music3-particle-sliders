# Learning rates

## Current finished-audio decision: Crisp Piano RAW32

Use the existing **cell6 G1e-6 RAW32** checkpoint, best tested among eight saved
rates on four matched cases with batch2/horizon128. Actual D is
`0.018000000000000002` and bank `0.029999999999999995`; exact common overrides
are `lr=1e-6`, `d_lr_mult=18000.000000000004`,
`prior_lr_mult=29999.999999999996`, `ema_decay=None`.
The equal-case mean estimated-piano attack-rise ON−OFF gain is
**+0.08936980775216061**. All four gains remain:
`+0.27936507936507937`, `+0.06586634653861534`,
`-0.017876039304610902`, `+0.03012384440955862`.

This is a **frozen estimated-piano attack-rise proxy**, not a human judgment or
broad acoustic-quality metric. The experimental piano separator can leak other
instruments and artifacts; its two reference anchors establish direction only,
not independent piano attribution in mixtures. The result is best tested in
this fixed RAW32/h128 bank with fixed D/bank, not a global/joint optimum or
new training default. Preserve the negative case. The complete
[portable decision metadata](docs/finished-output-lr-decision.json) retains all
eight candidate means, four gains and exact artifact/result hashes without
private filesystem paths or weights.

## Selection policy and actual common options

The sole score is the equal-case mean intended-effect intensity ON minus
matched OFF in finished outputs. Keep negative/zero outcomes and every case's
weight; technical failures remain missing. Teacher/student relative SSE,
preservation drift, generic beauty, confidence and completion are neither
selectors nor tiebreaks. Reuse completed searches, saved RAW checkpoints,
finished audio and first scores. No new search, generation, rerating or extra
confirmation is required for this decision.

The current response API accepts `lr`, `d_lr_mult`, `prior_lr_mult` and
`ema_decay=None` in `SessionConfig.options`. Shared ParticleGAN derives
D=`lr*d_lr_mult` and bank=`lr*prior_lr_mult`. Preserve the exact literal
multipliers in the linked JSON; rounded quotients can change the binary rates.
The summary is documentation metadata, not a full session or artifact loader.
Omitted controls still use inherited optimized defaults G0.012, D multiplier1.5,
bank multiplier2.5; selecting the supported `bispf` preset retains its own
inherited values. No preset, code, lock or default is changed by this guidance.

For configuration inspection with the real registered API:

```python
import json
from pathlib import Path
from particle_sliders.lifecycle import SessionConfig

decision = json.loads(Path("docs/finished-output-lr-decision.json").read_text())
# supplied_native_options contains the existing paired rows, verified local
# model assets/manifest/cache and exact admitted native numerical/source profile.
options = dict(supplied_native_options)
options.update(decision["rate_overrides"], steps=128, batch=2, seed=7)
config = SessionConfig("music3", "caller-declared-configuration", **options)
```

This constructs a configuration only, without opening a model session. `steps`
sets the128-update schedule horizon; it is not replaced by the earlier measured
RAW endpoint. Use the already saved selected RAW artifact, rather than retraining
it. Exact checkpoint continuation keeps its original complete configuration,
model/data/source identity and RNG state; these values cannot rebind a different
checkpoint. New sessions explicitly disable model-weight EMA and use RAW weights
throughout; no checkpoint averaging or EMA comparison is proposed.

## Earlier response-error screen

The [archived guide](https://github.com/HyperGAN/music3-particle-sliders/blob/f23889b4fafd6a63efaff2ceeb7fc4c312f9f5d6/LEARNING-RATES.md) retains the
original preset, all eight rates and the teacher-response SSE results.
