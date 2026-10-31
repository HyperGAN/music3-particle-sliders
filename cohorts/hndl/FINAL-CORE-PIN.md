# Final coordinated core pin

Active modern candidate only: exact core
`a794e3b4a8813b012f2c4faa3da82b78d4a90214`, core 0.12.0, official PG 0.9.4,
HyperGAN develop `5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL 0.10.0.
The canonical core wheel SHA256 is
`22e47ced4af8cb874d818db2f4006738f5f978cfcff324693931b6ab23f21c8a`.
The product lock SHA256 is
`9c354ffa98523fa9ed22f3936aea8922a120e530ce171db4ba8c7ed2e2707887`.

The lock was regenerated once from that exact wheel, recursively covering all 51
Python/data/graph/license entries in the core's two package namespaces. Three
source hashes changed from 4abc: `pg09.py`, `lifecycle.py`, `inference.py`.
Versions, HNDL graph identity, bank 128×4, rank 8, and alpha 8, product binding, training
objectives and native refusal gates are unchanged. The stable core 0.8.2 cohort
and all historical locks, notes, artifacts and metadata remain untouched.

## Installation and identity

Use a separate clean CPU venv. The qualified run starts with pip only, installs
the verified official Torch 2.6.0+cpu archive and public build/test dependencies,
then resolves the built product wheel with exact public VCS dependencies:

```bash
python3.12 -m venv /tmp/music3-a794-cpu
PUBLIC_PY=/tmp/music3-a794-cpu/bin/python
"$PUBLIC_PY" -m pip install --index-url https://pypi.org/simple \
  'setuptools==84.2.1' 'build==1.6.1' 'pytest==9.1.1' 'safetensors==0.8.0' \
  'torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl#sha256=59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91'
"$PUBLIC_PY" -m build --no-isolation cohorts/hndl --outdir /tmp/music3-a794-dist
"$PUBLIC_PY" -m pip install --index-url https://pypi.org/simple --no-build-isolation \
  'particle-sliders-core[hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@a794e3b4a8813b012f2c4faa3da82b78d4a90214#subdirectory=packages/particle-sliders-core' \
  'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@5f368a3770cf5bdae0448995c9f76e29af8049f7' \
  /tmp/music3-a794-dist/music3_particle_sliders_hndl-0.1.0-py3-none-any.whl
"$PUBLIC_PY" -I -m pip check
```

The actual run used the identical hash-verified cached official Torch archive to
avoid redownloading it. No dependency trees or symlinks were reused. Product and
core imports resolve to installed distributions; checkouts are absent from the
Python import path. Repository tests are explicit test inputs, not product import
fallbacks. Sdist installation builds from the archive's extracted temporary tree.
The owner-machine-only offline replay remains separately labeled and now expects
the exact a794 canonical wheel; its stable-lane inputs are unchanged.

The package remains source-only version 0.1.0. A new source/core identity produces
different wheel/sdist digests and checkpoint signatures. Exact public VCS builds
can have different archive bytes from the canonical wheel while retaining the
same 51 locked source/data bytes; this is admitted only with exact VCS provenance.
Old 4abc packets are preserved as old cohorts and never edited to gain admission.

## Current capability and test scope

CPU random native Qwen3 width 32 only; one to four steps; both released PG presets;
exact OFF, raw export reload and split resume. This low-level product API has no
session EMA interface; `ema_decay` is refused. It does not claim native EMA or
session rollback capabilities it does not expose. Product cleanup failure/retry,
invalid-state preflight and source admission remain covered.

Tests are run from `/tmp` with `python -I`, `--import-mode=importlib`, CUDA hidden
and HF offline. Supply `MUSIC3_OLD_PACKET_DIR` for actual preserved old-preview
packet refusal cases; otherwise those two cases are explicitly skipped. The
qualified run provides them and counts no skip as passed:

```bash
PRODUCT_REPO=/absolute/path/to/music3-particle-sliders
cd /tmp
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 \
MUSIC3_OLD_PACKET_DIR=/absolute/path/to/preserved-4abc-preview \
"$PUBLIC_PY" -I -m pytest -c /dev/null --import-mode=importlib \
  "$PRODUCT_REPO/cohorts/hndl/tests" -q
```

The old packets were generated before migration from installed product 9ae3eb4
and core 4abc0eb, after one real shared update: `game.state_dict()` and
`game.export_state_dict()`, saved with `torch.save`. Their source manifest and
SHA256 receipts are preserved privately; no tensor/adapter files are published.
`evidence/core-a794/` records final results, package identities, exact commands
and installed source/provenance checks. Initial review receipts remain unchanged.

No new GPU job, native width 4096/audio admission, pretrained loading, model weight
publication, deployment, workflow/access change or release is part of this pin.
Direct VCS dependencies prohibit PyPI upload. Root owns Claude actual-diff and
RETURN review, fixes and eventual routing; this task does not merge independently.
