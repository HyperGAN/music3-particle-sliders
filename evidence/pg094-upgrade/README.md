# Music3 official ParticleGAN 0.9.4 upgrade

Validated 2026-11-04. This is a new numerical-response source cohort; no model
weights were downloaded, trained on GPU, migrated or published.

## Exact source

- Root package: Music3 0.2.0, Python >=3.11, immutable VCS dependency on
  `HyperGAN/particle-sliders@ddb41423bcefc1fe76cd94d0f9b367353b3ea137`.
- Shared core: 0.12.0. All 574 packaged source/data files in the locally built
  wheel matched an exact git archive of that commit before installation.
- Official ParticleGAN: 0.9.4, upstream source
  `330eb94ee748cdcad131b0275d8de75fd5a7f96f`, profile `release-0.9.4` / `auto`.
- Canonical response lock SHA256:
  `dd3b1937bf1dd6c8712064a2ad1c42e3f098a026a2d85c7a69cb5013be5f45eb`.

[Runtime](runtime.json) records installed versions, exact core wheel hash and
CPU-only CLI output. [Packaging](packaging.json) records product wheel/sdist
hashes and verifies all 15 product Python/data files match the installed wheel,
source and sdist. [pip check](pip-check.txt) passed. Dependencies were reused
read-only from a reviewed local Python3.12 CPU environment into a fresh isolated
venv; this is not a clean network dependency-resolution receipt.

## Checks

[Installed tests](installed-tests.txt): **52 passed** using `python -I`, importlib
mode and no source-checkout fallback. These cover both official presets, exact
full versus split resume, frozen native weights, routed particle gradients,
zero-strength replay, EMA/native-only inference, cleanup ownership and retries,
ordinary-LoRA approximation, VCS provenance, profile rejection and official
width170/4096 rejection before native allocation. Re-checksummed packets with
old PG version/profile identities cannot restore into the current cohort.

The ordinary suite also runs these current tests and explicitly skips historical
core0.6.0 CLI tests when that exact old core is absent. The legacy tests are kept
for their original locked environment; they are not relabeled PG0.9.4 evidence.

The three initial failures were assertions that failed cleanup marks a session
closed. Core0.12 `particle_sliders/lifecycle.py:752` closes the binding first and
sets `closed=True` only after successful cleanup. The tests now verify retained
ownership, resolution, retry and successful final closure under that actual
public contract.

The installed `particle-slider` session CLI completed a fixture update and
produced a finite [1,32] numerical response through native-only EMA inference.
The newly merged unified dispatcher also passed allocation-free plan/check,
trained two updates with a periodic checkpoint, then resumed to four updates
with unchanged semantic request identity and both EMA export formats. Reporting
was explicitly disabled. The reproducible evidence-only profile is
`dispatcher-profile.json` with `dispatcher-config.json`; this fixture status
does not admit the real native model. No audio file or pretrained quality claim
is produced by either route.

## Preserved boundaries

Music3 native width4096, flat vectors >=170, native/pretrained training, acoustic
or token rendering, transformer/encoder hosts, image inputs, nonfinite/signed
or extrapolated strengths, and unofficial continuous profile aliases remain
refused. No projection, truncation or image-field reshape bypasses those gates.

The root historical `core.lock.json`, `requirements-legacy.txt`, `configs/`,
legacy train/infer/shared modules and independent HNDL cohort are unchanged
relative to parent PR7. The previous core0.8.2 response lock is archived byte for
byte (SHA256 `12e2310b2680173b5a6c3c220270d3d79cb9029d5660bf23f7d6e6f06fd986f3`).
The separate `feat/unified-training-command-20261104` product branch is neither
modified nor included. Canonical core `ddb41423` includes the newly merged
shared execution APIs, whose compatibility is exercised by the evidence-only
fixture above.
