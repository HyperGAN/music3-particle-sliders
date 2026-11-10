# Explicit cached fresh95 initialization

`music3_fresh95` is a separate opt-in library for Music3's original final-AR-hidden response lane. It keeps the original `music3` package, registrations, CLI and measured LR cohorts unchanged. It returns hidden responses; it does not render PCM or establish native audio availability. The existing [finished-output LR results](LEARNING-RATES.md) retain their original scope and values; they are not new measurements of this initialization profile.

## Installation and prerequisites

Use Python 3.12 and an environment already satisfying the original `music3-particle-sliders==0.3.0` guards: the exact core0.13.0 Git/wheel provenance and canonical Source closure, published ParticleGAN0.9.5, and the declared response/runtime dependencies. Install the addon as a genuine separate wheel from the repository root:

```sh
python -m pip install --no-deps ./fresh95
```

`--no-deps` preserves the prepared environment; it does not provide missing dependencies. The original native4096 lane additionally requires its exact original assets, BF16 eager runtime, numerical policy and whole loaded-checkpoint verification. Do not substitute an editable Source overlay, fabricated RECORD, newer core or different native runtime. This profile provides no standalone GPU-admitting CLI or model weights.

## Completed cache and owner API

The application supplies `config`, already completed original `PreparedAudioResponse` rows, their `teacher_origins`, a pristine original Qwen3 `component_factory`, and an `authenticate_and_admit` callback. The callback must authenticate the completed tensors against their whole-output/Source/base/token-packing origins and admit the requested operation. The library validates and binds these records; a file hash or model name alone does not authenticate how tensors were produced.

The factory receives a request containing the explicit operation/configuration and must return the original bare model. It must not attach an adapter or collect teachers. A WAV is not a prepared response cache. No teacher collector, tokenizer, acoustic decoder or token sampler is called by this API.

```python
from music3_fresh95 import open_training, open_inference, save_prepared_cache

# Owner-supplied: config, original_prepared_rows, teacher_origins,
# component_factory, authenticate_and_admit, complete_unpadded_ids.
cache = save_prepared_cache(
    "completed-cache.pt", config=config, prepared=original_prepared_rows,
    teacher_origins=teacher_origins, cache_id="explicit-new-cache",
    scope="completed-native", owner_admission=authenticate_and_admit,
)
with open_training(
    config=config, cache=cache, directory="new-fresh-run", mode="fresh",
    cohort_id="explicit-new-cohort", component_factory=component_factory,
    owner_admission=authenticate_and_admit, ema_decay=None,
) as owner:
    owner.update()
    full = owner.save("FULL1.pt")
    raw = owner.export("RAW1.pt")

with open_training(
    config=config, cache=cache, directory="new-resume-run", mode="resume",
    resume=full, cohort_id="explicit-new-cohort",
    component_factory=component_factory,
    owner_admission=authenticate_and_admit, ema_decay=None,
) as resumed:
    pass  # Next update continues the complete saved game and RNG state.

with open_inference(
    raw, component_factory=component_factory,
    owner_admission=authenticate_and_admit,
) as inference:
    hidden = inference.response(ids=complete_unpadded_ids, strength=1.0, seed=7)
```

This is an API pattern requiring authentic owner inputs, not a self-contained model-loading script. Each run directory must be new. The explicit configuration requires `lane`, `model_identity`, `steps`, `batch`, `seed`, `width`, `device`, `base_precision`, `checkpointing`, `max_sequence_length`, `pg_source_profile`, `penalty_input_domain`, `preset`, `ema_decay`, `lr`, `d_lr_mult`, and `prior_lr_mult`. The native lane also requires the original `model_dir`, `manifest`, `manifest_sha256`, and `native_numerical_profile`. CPU fixtures use width32/CPU/FP32; native responses retain full width4096/CUDA/BF16 and the explicit continuous domain. Whole contexts are retained. Response strength uses the original [0,1] capability. All rates are caller inputs; this addon changes no default.

## Fresh, FULL and RAW

Declare `fresh` or `resume` before component allocation. Fresh mode is a new cohort, not a way to reinitialize an old checkpoint. Public PG0.9.5 QR initializes the same original bias-free down Parameters at the original `kaiming_uniform_(a=1)` host variance `1/fan_in`; the same 128×4 bank receives R2Normal(0,1). Zero-UP, router/bridge and the original Linear critic constructor values are kept. The explicit new constructor stream uses the slider seed and critic seed+1. A durable policy receipt precedes optimizer creation; it makes no claim of equality to an old whole-factory RNG stream.

FULL restore accepts only this new profile's exact source/configuration/cohort/cache and restores the whole typed network, bank, critic, optimizers, cursors and CPU/CUDA RNG state. It skips fresh initialization and old updates. RAW inference restores only the added network and skips cache, critic/game/optimizer construction and fresh initialization. Old Music3 FULL/RAW cohorts are not migrated. Set `ema_decay=None`; all inference and exports use current raw weights, with upstream scalar optimizer statistics unchanged.

The owner preserves caller RNG and native observation state. Foreign markers, callbacks, attributes, flags or modes cause refusal instead of replacement. On cleanup failure it retains the unfinished ledger, blocks further work and removes the exclusive marker only after successful cleanup. Repair the reported foreign state and retry `close()`.

## Software evidence and limits

Four new installed synthetic CPU cases passed, with one tiny update total. They cover fresh initialization, strict FULL/RAW paths, identity/RNG preservation and retryable cleanup. The containing run later failed while writing its report; the four observations were adopted separately. This is not a claim of a successful complete harness report or an available blocked-call list. It does not qualify native pretrained execution, finished audio, acoustic quality, a new LR result, public defaults or released weights. Existing native zero-frame outcomes and all prior judgments remain unchanged.
