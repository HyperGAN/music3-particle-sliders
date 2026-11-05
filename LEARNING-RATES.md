# Learning rates

The current training configuration accepts optional `lr` (base/generator), `d_lr_mult` (critic), and `prior_lr_mult` (actual routed particle bank). Values must be finite positive numbers; booleans are rejected before model allocation. The existing shared ParticleGAN recipe creates all optimizers: D = `lr * d_lr_mult`, bank = `lr * prior_lr_mult`. No product GAN or optimizer math is duplicated.

Omitted controls retain the default `bispf_optimized` recipe: G **0.012**, D multiplier **1.5**, bank multiplier **2.5**. These are inherited defaults, not native calibration results. Selecting the supported `bispf` preset retains its existing base 0.00425, D multiplier 1, bank multiplier 2. Use the keys in the shared session configuration; the registered common training dispatcher executes the same recipe. Historical/source-only CLI cohorts retain their original interfaces.

Use `"lr": 0.00002, "d_lr_mult": 1.5, "prior_lr_mult": 2.5` in common session `options`. The numbers are an input example, not a tuned recommendation. The actual resolved shared recipe and configuration bind full-state resume; a changed rate requires a fresh experiment output/cohort. Older packets and closed HNDL source cohorts keep their original source/defaults. Current CPU optimizer/resume tests do not qualify pretrained/native learning rates, audio or semantic quality.

Native/wide-response source gates remain enforced. Native calibration compares identical initialized state, data, stochastic streams, precision, declared schedule horizons and completed-update budgets on the actual pretrained model. Select and independently confirm rates using only held-out raw teacher/student relative SSE: cast student, teacher and neutral to FP64 before subtraction, pool positions within each example, divide by that example's teacher-edit energy, then average examples equally. Keep the teacher-only valid/excluded set fixed across candidates; zero-energy examples remain undefined, and student failures never remove examples. See the [common LR protocol](https://github.com/HyperGAN/particle-sliders/blob/main/packages/particle-sliders-core/LEARNING-RATE-CONTRACT.md).

Preservation/no-edit drift, late/weak-field scores and native media outputs are optional diagnostics; they cannot select, reject or delay a rate. After the native mechanical proof, candidate endpoint net displacement is diagnostic too. Retain exact source/model/data/numerical identity, actual optimizer-rate, finite native-update, checkpoint-continuation and raw-response replay checks. Tiny CPU software checks are not native LR evidence.

## Measured native PG0.9.5 combination

The 2026-11-05 native 4,096-wide BF16 response experiment selected **G=0.000001**, actual **D=0.018000000000000002**, and actual bank **P=0.029999999999999995**, at a common horizon of 128 updates. This is the best tested G at the prospectively declared lower bound 1e-6, **unbracketed**, with D/P fixed. It is not a joint or global optimum. The inherited defaults above remain unchanged.

Development RAW at 128 was **1.0382026088532337**, above OFF=1. The separately unused two-example final measured **0.8928904545147202**, about 10.71% below OFF, with zero optimizer updates. Both examples were valid; none were excluded. An independent saved-tensor audit reproduced the operand-first FP64 score exactly. A separate CPU audit recomputed packet payload checksums, verified unchanged selected 128-update state and exact RAW network tensors, and confirmed `ema_decay=None` with no EMA shadows. Public RAW response/OFF equality comes from the source-qualified native worker. This measures numerical hidden responses; it makes no acoustic or subjective quality claim.

The [rate preset](presets/music3-native-pg095-h128.json) stores exact multipliers and actual float hex values. Use G times `d_lr_mult=18000.000000000004` and `prior_lr_mult=29999.999999999996`; these products reproduce the measured D/P bits. Do not substitute rounded multipliers. The preset contains session-option overrides, not a complete training configuration:

```python
import json
from pathlib import Path
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response

rates = json.loads(Path("presets/music3-native-pg095-h128.json").read_text())
options = dict(qualified_native_options)  # local model/manifest and strict profile
options.update(rates["session_option_overrides"], rows=paired_training_rows)
with open_response(SessionConfig("music3", "my-new-pg095-native-cohort", **options)) as run:
    for _ in range(128):
        run.step()
    run.export_inference("music3-raw.pt", selection="raw")
```

Prepare the exact assets and numerical settings in [the native guide](NATIVE-CONTINUOUS.md) first. The measured scope is product `8d4e5d1e9fee333256afc52beaececef0217198e`, core `3dd672a7f9b8c169daef90fd63e4bb42b9ddc423`, official PG0.9.5/continuous, native MiniMax-Music3 revision `bd348f9c49ea3c1b39f33ace3436f8fad435f24e`, strict BF16 eager profile, complete 128×4 bank, batch 2, seed 7, and horizon 128. Other data or numerical profiles need their own measurements. Keep old packets on their original source pins; a new rate/configuration uses a fresh cohort.

The [sanitized evidence](evidence/pg095-native-lr/summary.json) records all 8 completed 32-update screens, matched 64-update endpoints for cells 4/5/6/7, the selected 128-update endpoint and unused final. It includes no weights, private paths or corpus payload. Historical core5d/PG801 rates and packets retain their old pins.
