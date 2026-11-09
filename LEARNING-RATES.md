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

<a id="measured-native-pg095-combination"></a>

## Historical teacher-response SSE screen (RAW128)

The following original measurement, presets and reports retain their source
identities. Its held-out response-error scores are diagnostic history, not
current acoustic intended-effect scores or selectors. No confirmation replay
or completed training is required by this documentation update.


The 2026-11-05 native 4,096-wide BF16 response experiment selected **G=0.000001**, actual **D=0.018000000000000002**, and actual bank **P=0.029999999999999995**, at a common horizon of 128 updates. This is the best tested G at the prospectively declared lower bound 1e-6, **unbracketed**, with D/P fixed. It is not a joint or global optimum. The inherited defaults above remain unchanged.

Development RAW at 128 was **1.0382026088532337**, above OFF=1. The separately unused two-example final measured **0.8928904545147202**, about 10.71% below OFF, with zero optimizer updates. Both examples were valid; none were excluded. An independent saved-tensor audit reproduced the operand-first FP64 score exactly. A separate CPU audit recomputed packet payload checksums, verified unchanged selected 128-update state and exact RAW network tensors, and confirmed `ema_decay=None` with no EMA shadows. Public RAW response/OFF equality comes from the source-qualified native worker. This measures numerical hidden responses; it makes no acoustic or subjective quality claim.

The unchanged [historical rate preset](presets/music3-native-pg095-h128.json)
retains the exact session literals and float hex values from that earlier
128-update response-error screen. It is a numerical teacher-match record,
not the current finished-audio selector and not an instruction to repeat128
updates. The current selected artifact is RAW32; the G value happens to agree.

Prepare the exact assets and numerical settings in [the native guide](NATIVE-CONTINUOUS.md) first. The measured scope is product `8d4e5d1e9fee333256afc52beaececef0217198e`, core `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, official PG0.9.5/continuous, native MiniMax-Music3 revision `bd348f9c49ea3c1b39f33ace3436f8fad435f24e`, strict BF16 eager profile, complete 128×4 bank, batch 2, seed 7, and horizon 128. This historical result does not transfer numerical scores to different data or profiles. Keep old packets on their original source pins; changed configuration does not rebind an existing checkpoint.

The [sanitized evidence](evidence/pg095-native-lr/summary.json) records all 8 completed 32-update screens, matched 64-update endpoints for cells 4/5/6/7, the selected 128-update endpoint and unused final. It includes no weights, private paths or corpus payload. Historical core5d/PG801 rates and packets retain their old pins.
