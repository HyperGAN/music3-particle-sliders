# Music3 neutral response-session proposal

The new opt-in package registers `music3.lifecycle:BACKEND` under
`particle_sliders.response_backends` and `music3.lifecycle:INFERENCE` under
`particle_sliders.inference_backends`. No image backend is registered. The
dependency remains on the existing PG0.9.1 line: exact reviewed core0.8.2 source
`f6807a91546e43e198645a3f8925d2dc7c9e03b4`, including the independently cleared
reversible checkpoint-ownership update. This is a new source cohort.
The historical core0.6.0/PG0.9.1 `core.lock.json`, `requirements.txt`, six CLI
scripts, loaders, saved results and checkpoint cohorts remain unchanged. PR5
`471a7c9` and independent PR3 `2f1ff5a` remain preserved; no merge is proposed.

## Exact shared evidence and environment

`music3/response-core.lock.json` is an unchanged copy of the owner's canonical
091 lock fragment, including its full38-file source map and minimal audio
module closure. Registration verifies installed source ownership, all38 source
bytes, PG/core versions, imported origins and available installation provenance
before importing the core. It permits no checkout or editable fallback.

Use only the exact owner-reviewed wheel in a separate environment:

- Canonical lock SHA256: `12e2310b2680173b5a6c3c220270d3d79cb9029d5660bf23f7d6e6f06fd986f3`.
- Core wheel SHA256: `50a6a7d06240254357569fd3cacbb5cf5e9d3c1771f642dfb719733b5711f32a`.
- Core wheel: `task-2/evidence/checkpoint-ownership-packaging/091/dist/particle_sliders_core-0.8.2-py3-none-any.whl`.
- PG0.9.1 release wheel SHA256: `b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec`.
- Companion source archive SHA256: `6bad2ec9cd5ccac54c9b313004bf5b3937cefdd6b4207746b2586f75bb091127`.
- Companion source archive: `task-2/evidence/checkpoint-ownership-packaging/091/reviewed-source.tar`.
- Shared packaging receipt: `task-2/evidence/checkpoint-ownership-packaging/receipt.json`.
- Corrected metadata: `task-2/evidence/checkpoint-ownership-packaging/metadata-correction.json`.

The exact wheel, all38 source entries and companion archive were checked as
inert bytes in this task. Nothing was installed or imported as runtime code.
All four guides, including CHECKPOINT-OWNERSHIP.md, were copied byte-for-byte
from the verified archive to `docs/shared-response/`. They are also included
in these new sdists; the previous guide-distribution omission is fixed. Their
original source-proposal status prose is retained. The parent reports independent
installed artifact clearance of the60 selected ownership checks (30 per PG line),
in addition to the earlier response-contract review. Shared artifact evidence
does not qualify this Music3 product source. See the
[ownership adoption](checkpoint-ownership-adoption.md) and its source receipt.

The new `pyproject.toml` declares core0.8.2 as a dependency and uses separate
registry entry points. That version constraint alone is insufficient provenance;
the runtime canonical map and exact local wheel receipt are mandatory. Do not
resolve an unpublished package from an arbitrary index or combine this proposal
with the old CLI environment. The optional response fixture declares the native
Transformers5.15.0 version present in the historical Music3 runtime receipt.
No native dependency install has occurred in this task.

## Owned training/provider/observation path

The product factory owns actual random Qwen3 construction, the shared native
Music3 ParticleNetwork and its participating128x4 bank, frozen paired teachers,
the immutable AudioProvider, full neutral-context conditions, existing
AudioResponseCritic and ConditionalPG09 construction. Provider identity binds
complete token tuples, continuation lengths, frozen targets and conditions.
Fetches clone stored rows; caller mutation cannot rewrite training identity.
The actual bank is passed by identity to the shared game, which owns unique
optimizer/prior ownership, GAN updates, private RNG, coverage and checkpoints.
No GAN mathematics, optimizer law or local lifecycle is copied.

The response descriptor explicitly declares `audio_ar_hidden`,
`native_final_ar_hidden`, `continuous_vector`, native shape `[1,32]` for the
tiny fixture, and `routed_particle_gan`. Trainables stay FP32. Student forwards
receive neutral tokens; positive teachers remain frozen, with identical suffix
history where declared. There is no diffusion scheduler, token generation or
waveform decode in this response arm. Native packing code remains behind its
unsupported native-width gate for future separately reviewed source changes.

The shared response session owns train/resume/evaluate, raw/EMA export, invalid
restore preflight and adapter observation. Binding.observation additionally
restores the native model's persistent/nonpersistent buffer registrations,
buffer values, RoPE cache counters, modes, gradient flags and existing gradient
objects/values. It encloses provider bounds/fetch and response forward, including
exceptions. No frozen model weight clone is taken merely to observe a response.
Setup/update cleanup uses the adapter's shared reversible checkpoint handles
before restoring the product's own LoRA forwards and the common native cleanup
for modes, gradient flags and checkpoint settings. A conflict retains the live
adapter and retry ledger in CleanupPending; no broader restoration runs while
block cleanup refuses. The common session close policy is unchanged: resolve
the conflicting owner, then call the error's retry_cleanup() or Binding.close()
directly. LoRA forward replacement conflicts are also refused without overwrite.

## Data-independent inference

Binding supplies the shared InferenceSpec. The v2 factory constructs only the
same base and adapter and returns InferenceBinding with the actual network,
native identity, numerical callback, cleanup and native observation hook.
It never calls the training factory or constructs AudioProvider, teacher rows,
critic, game or optimizers. Artifact configuration contains only model geometry,
base seed/precision, sequence budget and exact source profile. Training paths,
rows, update budget and EMA state are absent. New response IDs are caller input;
the artifact contains no default training prompt.

Use the shared installed lifecycle and CLI in the admitted environment:

```python
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.inference import open_inference

config = SessionConfig('music3', 'new-music3-response-cohort',
    dummy=True, width=32, steps=4, checkpointing=True, ema_decay=.9)
with open_response(config) as run:
    run.train(stop_after=2, on_step=hypergan_observer)
    run.checkpoint('new-run/training.pt')
    run.export('new-run/adapter.pt', selection='ema')
    run.export_inference('new-run/adapter-v2.pt', selection='ema')
with open_inference('new-run/adapter-v2.pt', options={'device': 'cpu'},
                    operation='response') as native:
    hidden = native.response(ids=[61, 67, 71], strength=.5, seed=11)
```

`hypergan_observer` is a caller-provided HyperGAN run observer. HyperGAN owns
presentation through on_step, immutable manifest and artifact seams. There is no
new product dashboard. Future qualified acoustic sampling must be a separate
product callback operating on the same adapter.

```sh
particle-slider capabilities --interface response --config examples/music3-response.json
particle-slider train --interface response --config examples/music3-response.json \
  --out new-response-run --export-format both --selection ema
particle-slider infer --artifact new-response-run/adapter-v2.pt \
  --runtime examples/music3-response-runtime.json --mode response \
  --native examples/music3-response-request.json --strength .5 --out new-hidden.pt
```

These are future validation examples; no command above has run in this task.

## Executable gates and remaining work

This new proposal accepts only the reviewed official PG0.9.1 `release` / `auto`
pair and a CPU FP32 random-native fixture below width170. Common response
preflight then verifies actual installed PG bytes and advertised native shape
before model allocation. The product additionally refuses every wide/native
request, including explicit ALZ overlay requests. Its recovery/qualification
is pending; this proposal does not weaken that hold. Native Music3 remains
4096-wide/BF16 in its descriptor and is never projected, truncated, reshaped or
silently rebound to a smaller model. PG0.9.3 is not selected or downgraded.

No rendering capability is advertised. Image references, PIL/latent output
types, transformer/encoder training, unsupported signed/extrapolated strengths,
training-data overrides during v2 inference and staged caller-assembled models
are refused. Valid response strengths remain finite [0,1].

The earlier image-only API, unconditional default-profile preflight, missing
audio closure and omitted bare product names are resolved in the supplied
reviewed shared contract. Remaining gates are the scheduled Music3 installed
conformance slot, native continuous-profile recovery/upstream selector work,
native tokenizer/history and pretrained acoustic qualification, and separate
matched-weight HNDL mapping/parity of existing critic/router/LoRA definitions.
No new concrete shared runtime gap was found by source inspection in this pass.
See the [scheduled-slot test plan](response-session-test-plan.md). Its first
authored product selection is `tests/test_response_proposal.py`, with22
parameterized cases pending execution.

This source pass ran only AST/data parsing, artifact byte hashing and git diff
checks. No tests, installs, native dependency imports/model loads, GPU work,
publication, merge, deployment or active pin changes occurred.
