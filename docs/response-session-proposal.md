# Current Music3 response sessions — ParticleGAN 0.9.4

The root package registers the Music3 numerical-response backend and independent
native-only inference backend. It pins shared core **0.12.0**, commit
`ddb41423bcefc1fe76cd94d0f9b367353b3ea137`, with official **ParticleGAN 0.9.4**
(source `330eb94ee748cdcad131b0275d8de75fd5a7f96f`). Python 3.11+ is required.
The exact source profile is `release-0.9.4` with penalty input domain `auto`.
Aliases such as `release` and recovered continuous overlays are refused.

`music3/response-core.lock.json` binds all 574 packaged core source/data files,
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
  --out new-response-run --export-format both --selection ema
.venv/bin/particle-slider infer --artifact new-response-run/adapter-v2.pt \
  --runtime examples/music3-response-runtime.json --mode response \
  --native examples/music3-response-request.json --strength .5 --out new-hidden.pt
```

The example is an explicit tiny random Qwen3 CPU software fixture. It does not
load pretrained weights. Current training uses shared `ConditionalPG09` with
Music3's actual routed particle bank, frozen paired teachers and native hidden
response geometry. The core owns all GAN losses, updates, optimizer/prior
ownership, RNG, resume and exports. No GAN formula is copied into this product.

Response sessions support exact same-cohort resume, raw/EMA export, numerical
responses, frozen-base observation and ordinary-LoRA approximation. Native-only
inference builds only the base and adapter; it creates no teacher, critic, game
or optimizer. On cleanup conflicts core0.12 leaves the session open: resolve the
external owner, then retry `session.close()`. Failed handles retain their retry
ledger, and successful sibling cleanups are not repeated.

## Native boundaries

Official PG0.9.4 still does not admit flat continuous AR vectors of width >=170.
Music3's genuine hidden width is 4096. Native/pretrained training, wider random
fixtures, token generation, acoustic decoding and waveform rendering remain
refused before model allocation. The factory does not project, truncate or
reshape these responses into an image field. Transformer/encoder hosts, signed
or extrapolated strengths, image references and media outputs remain refused.
Accepted response strengths are finite values in [0,1].

The historical published controls, model identities, source locks, checkpoints,
recipes and ComfyUI loader are unchanged. A dependency upgrade does not retrain
or migrate those artifacts. See [installation cohorts](../INSTALL-COHORTS.md)
and [the archived PG0.9.1 response proposal](../cohorts/legacy-response/response-session-proposal.md).
The independent HNDL cohort remains separately pinned; it does not qualify wide
native audio or replace this canonical installation.
