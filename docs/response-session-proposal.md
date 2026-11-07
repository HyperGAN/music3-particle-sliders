# Current Music3 response sessions — ParticleGAN 0.9.5

The root 0.4 package owns `music3.native` and `music3.response_adapter` and
registers the Music3 numerical-response backend and independent
native-only inference backend. It pins model-independent shared core **0.14.0**, commit
`1ccf67ed041d2ba1646e51b3048651cfb83018b2`, with official **ParticleGAN 0.9.5**
(source `1b8355b184b21bb7d81104ee590813c262f6df0a`). Python 3.11+ is required.
The default source profile is `release-0.9.5` with penalty input domain `auto`.
Aliases such as `release` and recovered continuous overlays are refused. A
[separate exact native source candidate](../NATIVE-CONTINUOUS.md) requires the
explicit current continuous source/domain pair and strict numerical profile.

`music3/response-core.lock.json` binds the complete packaged generic core source/data inventory,
the exact core wheel, and the upstream release manifest. Registration checks
distribution ownership, file bytes, versions, import origins and available
installation provenance. Editable/source-checkout fallbacks are refused.
`pyproject.toml` and `requirements.txt` select the same immutable source pin.

Install in a new isolated environment, then run the supported installed CLI:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install '.[response-fixture,test]'
.venv/bin/particle-slider capabilities --interface response --config examples/music3-response.json
.venv/bin/particle-slider train --interface response --config examples/music3-response.json \
  --out new-response-run --export-format both --selection raw
.venv/bin/particle-slider infer --artifact new-response-run/adapter-v2.pt \
  --runtime examples/music3-response-runtime.json --mode response \
  --native examples/music3-response-request.json --strength .5 --out new-hidden.pt
```

The example is an explicit tiny random Qwen3 CPU software fixture. It does not
load pretrained weights. Current training uses shared `ConditionalPG095` with
Music3's actual routed particle bank, frozen paired teachers and native hidden
response geometry. The core owns all GAN losses, updates, optimizer/prior
ownership, RNG, resume and exports. No GAN formula is copied into this product.

Set `ema_decay=null` in JSON (`None` in Python) and use raw current weights for
training, evaluation, selection, confirmation, export and inference. Do not
maintain model-weight EMA or compare averaged checkpoints. Historical EMA
conformance evidence and optional compatibility APIs retain their original scope.

Response sessions support exact same-cohort resume, raw export, numerical
responses, frozen-base observation and ordinary-LoRA approximation. Native-only
inference builds only the base and adapter; it creates no teacher, critic, game
or optimizer. On cleanup conflicts core0.12 leaves the session open: resolve the
external owner, then retry `session.close()`. Failed handles retain their retry
ledger, and successful sibling cleanups are not repeated.

## Native boundaries

The default `release-0.9.5/auto` path refuses flat AR vectors of width >=170.
Music3's genuine width4096 requires the explicit `release-0.9.5/continuous`
pair and strict native policy. No projection or geometry conversion is used.
This new source cohort has CPU software checks; pretrained GPU updates and
exact native continuation remain unqualified. Token and waveform generation
are not response-session operations. Transformer/encoder hosts, signed
or extrapolated strengths, image references and media outputs remain refused.
Accepted response strengths are finite values in [0,1].

The historical published controls, model identities, source locks, checkpoints,
recipes and ComfyUI loader are unchanged. A dependency upgrade does not retrain
or migrate those artifacts. See [installation cohorts](../INSTALL-COHORTS.md)
and [the archived PG0.9.1 response proposal](../cohorts/legacy-response/response-session-proposal.md).
The independent HNDL cohort remains separately pinned; it does not qualify wide
native audio or replace this canonical installation.

Historical PG094/PG801 packets require their original source environment. No checkpoint conversion or old-rate transfer is performed. [Prior guide](response-session-pg094.md).
