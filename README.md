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

### Activation-dependent native semantic strength

Use `generate_gated` to apply the shared CFP, CLABS, or matched-random rule to
the same saved RAW adapter. Each decision is made from the conditional row of
the live, pre-edit frame-feedback embedding before the cached Qwen3 forward.
That embedding already combines the previous frame's sampled semantic code
and RVQ residual codes in the original Music3 loop. Both classifier-free
guidance rows use the same strength for that forward. Prefill is OFF and
predicts only the warm-up code, so it does not advance the gate. There is one
decision per subsequent frame prediction, including a prediction that ends
generation at EOS. RVQ substeps and the acoustic decoder use their original
paths. The semantic result can be passed to that decoder unchanged.

Build the local core wheel as a source artifact. Music3 checks the standalone
steering module's bytes against the exact core revision before loading it.
The existing Music3 runtime keeps its original installed response core and
checkpoint identity.

```sh
python -m pip wheel --no-deps --wheel-dir wheels \
  'particle-sliders-core @ git+https://github.com/HyperGAN/particle-sliders.git@b1f780decc516808cfd3019d5e5b674457161f86#subdirectory=packages/particle-sliders-core'
```

```python
from music3.activation_gate import SemanticGatePlan
from music3.scheduled_semantic import generate_gated

plan = SemanticGatePlan(
    core_wheel=steering_wheel,
    method=calibration["method"],
    parameter=calibration["parameter"],
    calibration_sha256=calibration_record_sha256,
    direction=calibration.get("direction"),
    seed=calibration.get("seed"),
    clabs=calibration.get("clabs"),
)
semantic, receipt = generate_gated(
    pipe, scale_adapter=slider.scaled, audio_duration=seconds,
    strength=calibration["strength"], gate_plan=plan, **inputs,
)
```

The caller supplies the fitted unit direction and the active-rollout
threshold or bias. CLABS also takes the measured projection statistics,
explicit controller settings, and an independent seed. The matched-random
control uses the target rate and an independent seed. The receipt records
the calibration digest, resolved settings, projections, decisions, effective
strengths including the OFF prefill, and the realized firing rate and total
strength over cached predictions. Replay a fixed control with the same
prefill strength and retain the native sampler and EOS semantics.

For qualification, bind Music3's model, RAW adapter, tokenizer, CFG, frame
and acoustic settings, and the probe inputs to the calibration record.
Fit contrasts on matched neutral and target frame feedback, calibrate with
steering active on disjoint cases, and evaluate finished audio on further
held-out cases at the matched budget. Keep the intended-effect ON-minus-OFF
score and catastrophic-output verdicts separate. The archived RAW32
comparison and RAW64 fixed controls preserve their original evidence. The
RAW64 semantic receipts record frame hiddens for acoustic synthesis, with
no live feedback directions or adaptive budget calibration. Adaptive
Music3 gain remains unmeasured.
[Shared equations and calibration protocol](https://github.com/HyperGAN/particle-sliders/blob/b1f780decc516808cfd3019d5e5b674457161f86/docs/strength-schedules.md#activation-dependent-steering)

### Read back cached-frame dose

Use `SemanticDoseMatcher` to propose a zero-prefill constant and measure both
returned strength traces. Supply a separate core wheel built from
`34ed1856f224f9a32ab092e58d9803f0cfc45c38` for the shared interval-dose
functions. Music3 verifies that module's source bytes before loading it.

```python
from music3.semantic_dose import SemanticDoseMatcher

matcher = SemanticDoseMatcher(dose_wheel)
gate_strengths = receipt["applied_strengths"]
candidate = matcher.propose(gate_strengths, expected_eligible=receipt["eligible"])
# After the registered constant arm returns, use its actual strengths from generate().
comparison = matcher.readback(gate_strengths, constant_applied_strengths)
```

Run a feasible proposal with effective strengths `[0.0] + [candidate] * max_frames`
and the same request, seed and native sampler. The proposal uses a hypothesized
cached length and returns `None` if that clock cannot fit the dose within `[0,1]`.
Readback excludes the zero prefill and counts the cached EOS prediction when
present. On the rank-8, alpha-8 Music3 BF16 CUDA path, Torch 2.9.1+cu128
applies binary32 scalar coefficients before BF16 tensor rounding.
`exact` requires equal doses on the two returned clocks. `nearest` records the
closest representable constant for the returned control clock and its signed
`dose_error`. Treat that arm as a timing comparison only when the prospective
protocol explicitly accepts nearest controls. An `unmatched` result leaves the
timing comparison open and preserves the independent gate-versus-OFF result.
