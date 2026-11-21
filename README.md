# Music 3 particle sliders

**Calculated memory:** Native BF16 LM response training: **32 GiB GPU / 32 GiB host** for batch 2,256 tokens; hidden-response inference **22 GiB GPU**. Finished audio is a separate, unqualified path. [Full memory ledger and limits](MEMORY.md). These are planning estimates, not measured peaks.

The current root package uses shared **core0.15.2 / ParticleGAN0.11.1**
at immutable source `ec1161f5d74481be3540cb0dfa63d0e8f8b54e90` (Python 3.11+).
Use the installed numerical-response lifecycle for train/resume/export/inference:

```sh
python -m pip install '.[response-fixture,test]'
particle-slider train --interface response --config examples/music3-response.json \
  --out new-response-run --export-format both --selection raw
```

Set `ema_decay` to `null` in JSON (`None` in Python). Train, evaluate, export
and infer with raw current weights; do not add EMA or checkpoint averaging.

See [current usage and exact source checks](NATIVE-CONTINUOUS.md),
[installation cohorts](INSTALL-COHORTS.md), and
[publication and qualification status](PUBLICATION-STATUS.md).
The example is a small random Qwen3 CPU fixture on the official release.
The [separate native source candidate](NATIVE-CONTINUOUS.md) explicitly selects
published PG0.11.1 with the explicit continuous input domain and a strict BF16 eager profile for the
genuine 4096-wide Music3 state. Native updates, exact continuation and public RAW
response replay completed in the archived native runtime. The completed
finished-audio comparison selects existing **cell6 G1e-6 RAW32** at fixed
D0.018000000000000002 / bank0.029999999999999995 and horizon128, with equal-case
estimated-piano attack-rise ON−OFF mean **+0.08936980775216061** across four cases.
[Exact options, all outcomes and scope](LEARNING-RATES.md). This frozen
piano-stem proxy can contain separation leakage/artifacts; it is not human or
general acoustic-quality validation. Reuse the saved search, RAW, audio and
scores; no extra confirmation, rerating or new training is required. The older
RAW128 teacher-SSE result0.89289 remains diagnostic history. The response API
retains its own hidden-state contract; no generic waveform API is implied.
No projection or reshape bypasses source/domain requirements.

The bounded [ordinary LoRA helper](docs/ordinary-lora-distillation.md) remains
available for explicit CPU approximation. The separately packaged
[archived HyperGAN/HNDL candidate](https://github.com/HyperGAN/music3-particle-sliders/blob/f23889b4fafd6a63efaff2ceeb7fc4c312f9f5d6/cohorts/hndl/README.md) documents its original pin.
Published historical controls and loading paths are unchanged; no weights are
retrained or migrated by the current dependency upgrade.

[Earlier CLI and reproduction instructions](https://github.com/HyperGAN/music3-particle-sliders/blob/f23889b4fafd6a63efaff2ceeb7fc4c312f9f5d6/README.md#historical-core060-cli-contract)
require the original product source and dependency environment.

## Scheduled native semantic strength

With the original semantic pipeline and its attached RAW slider, call
`music3.scheduled_semantic.generate(pipe, scale_adapter=slider.scaled,
audio_duration=seconds, step_gains=gains, **inputs)`. Pass one prefill
gain followed by one for each possible native audio frame. The batched
conditional and unconditional branches share each gain. Effective strengths
must lie in `[0,1]`. The call returns the semantic stage result and the
strengths used through native termination. Feed the frame hiddens into the
original acoustic decoder. Pretrained scheduled audio effects are pending.
