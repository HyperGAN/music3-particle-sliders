# Packaged Music 3 PG 0.9.1 integration

The product targets installed `particle-sliders-core` API **0.6.0** with
ParticleGAN **0.9.1**. Exact source, wheel and installed-file identities
belong to `core.lock.json`. Exact local package `9be81c1` passed 80 product
and 49 focused shared-state CPU tests. Its public source publication and
Ada review are pending; earlier candidate results are retained separately.

## Ownership and public API

The public package owns the complete audio workflow through
`particle_sliders.pg09_audio_cli`: `make_audio_game`, `run_training`,
`probe_export`, and the train/probe/plan dispatcher. Its packaged
`particle_sliders.pg09_audio` and `particle_sliders.music3_native` modules
supply the response adapter and native model binding. The shared
`particle_sliders.ConditionalPG09` game owns losses, optimizers, update
scheduling, prior regularization, checkpoint state and resume.

Product `music3.train` and `music3.infer` validate Music 3 arguments and
call the public workflow main. They do not implement model construction,
GAN updates or probe orchestration. The loader verifies the installed
distribution and locked files before accepting imported module origins.
It does not import repository-internal audio modules, alter the caller's
Python search path or inject checkout environment variables.

## Response and conditioning contract

The CPU cohort executes a genuine random Qwen3 causal model with hidden
width 32 and Music 3 audio-start token ID 151669. The packaged routed
adapter attaches to the actual native host and retains the same
participating 128×4 bank. Both presets passed response gradients to that
bank, bank updates and unchanged frozen weights. A separate width-4096
random geometry gate also passed both presets under the explicit
continuous profile. Its synthetic vocabulary does not qualify native
tokenization, pretrained execution or acoustic sampling.

Each row binds distinct neutral and positive token bytes and any shared
continuation history. The teacher is the frozen positive final-token
hidden state. The student uses neutral tokens at adapter scale +1. The
response remains the native hidden state, without projection, truncation
or a logits substitute. The conditional critic receives frozen neutral
context, token lengths, history length and adapter sign. Exact row, target
and condition identities belong to the shared checkpoint contract.

The generator objective is the public PG 0.9.1 adversarial loss
plus its public prior regularizer. The critic objective is the public
adversarial loss plus penalty. The product adds no MSE, NMSE, caption loss
or alternate teacher objective. Recipe construction, optimizer behavior,
constraints, sampled rows and RNG remain central package responsibilities.
The CPU default is `bispf_optimized`, with a fixed 4-update horizon;
`--steps` is the absolute completion target within that horizon.

## New checkpoint lineage and scale contract

PG 0.9.1 starts a new cohort. There is no migration of PG 0.9.0 training
checkpoints, inference exports or optimizer state. The earlier PG42
candidate is superseded by the central Adam/native-loader repair. This
fresh cohort uses the final local wheel identified in `core.lock.json`.

`checkpoint.pt` is the full shared training checkpoint. Resume must retain
its source/runtime identity, fixed recipe horizon, model and critic,
optimizers, controller, prior, conditioning, targets and RNG.
`adapter-export.pt` is the shared inference export and does not resume
training. It is not a conventional native or ComfyUI LoRA. The packaged
workflow owns artifact schemas and loading; the product does not reconstruct
state or rewrite source provenance to admit an older export.

The new route is unipolar. Finite probe scales in [0,1] are accepted; +1 is
the trained treatment and exact frozen-neutral replay at 0 is required.
Negative and nonfinite scales are rejected. Export reload parity, probe
RNG preservation and fresh-process continuation passed on that exact
wheel. Hidden-state probes alone do not establish acoustic quality.

## Native-runtime gates

The pretrained Music 3 LM response width is 4096. The final package's
source/domain guard must admit the actual native response under its exact
reviewed runtime profile. A smaller response or different source profile
must not be substituted to bypass a guard. Profile names accepted by a
product CLI are not evidence that the final native path is qualified.

`--live` calls the public package's plan route with staged model, verified
asset manifest and fixed train/heldout prompts. It emits a plan only. The
coordinator separately owns admission, physical GPU lease, guardian and
resource bounds before any native execution.

Only the LM/particle product binding is targeted. Transformer and
condition-encoder training remain refused. Native pretrained smoke,
exact fresh-process continuation/export reload on that runtime, bounded
acoustic renders and listening review remain pending. Historical
`winning_formulation()` metadata, released LoRAs and PG 0.9.0 CPU receipts
do not qualify this new package or checkpoint cohort.
