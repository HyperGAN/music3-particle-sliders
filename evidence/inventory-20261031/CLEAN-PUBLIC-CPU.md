# Final bounded polish and clean public-source CPU qualification

Re-review base: `68ab1f7d3e2f3f065eef5539117d80e71cfba193`.

The offline replay is now explicitly **owner-machine-only**. Its absolute
defaults and unpinned read-only dependency symlinks do not establish portability
or clean resolution. `HNDL_CORE_WHEEL` and `HYPERGAN_CHECKOUT` are overridable.
This labels the existing evidence honestly without introducing a new lock scheme.

The only runtime polish is explicit closed/pending state. After any close, the
slider cannot enter a strength context and the owner cannot expose its game.
When a foreign forward conflicts, modes/gradient flags are still restored,
`cleanup_pending` remains true, and the foreign forward and retry ledger remain
reachable. The regression asserts game access and adapter response both raise.
Resolving ownership and retrying clears the pending ledger; the closed adapter
cannot resume use. A raw model reference is for repair/release, not permission to
execute a host with an unresolved foreign forward.

## Separate clean public-source lane

Completed on Linux x86_64 / CPython3.12.15, Torch2.6.0+cpu. A new ordinary venv
started with **pip25.0.1 only** and system site packages disabled. No dependency
symlinks, copied package trees, owner-built core wheel or local HyperGAN checkout
were used. The existing official Torch CPU wheel was reinstalled as an archive,
after checking SHA256
`59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91`.
Its public source is
`https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl`.
Using that verified cached download avoided transferring the same178MB again.

Pip fetched/build-installed the exact public VCS sources:

- core `4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, package subdirectory
  `packages/particle-sliders-core`, extra `hypergan-hndl`;
- HyperGAN `5f368a3770cf5bdae0448995c9f76e29af8049f7`;
- remaining runtime dependencies from `https://pypi.org/simple`, including
  PG0.9.4, HNDL0.10.0 and Transformers5.22.1.

Installed VCS provenance contains both exact requested revisions/commit IDs.
The canonical core source map and actual HyperGAN/HNDL verifier pass without
relaxing any hash or gate. All package distributions resolve inside the new
venv; its entire site-packages tree contains no symlinks. The candidate product
was built/installed from this repository, not imported from its source directory.

**24 repository HNDL tests passed in4.93s; `pip check` passed.** This includes the
new unusable-conflict assertions and existing matched-weight graphs, gradients,
zero strength, split resume, export and width refusal. The stable product's
source/pins are unchanged; its previous37 selected tests remain separate evidence.

The exact completed commands and public-source reproduction recipe are in
`clean-public/commands.txt`. `resolution.json` retains the resolved package names,
versions, public download URLs/hashes and VCS revisions from pip's install report;
`freeze.txt`, `runtime.json`, initial package list and literal results are adjacent.
The full unabridged pip report/log remain in the task workspace under
`/ml2/codex-storage/future-workspaces/2026-10-31/task-11/evidence/clean-public-*`.
Transitive versions are a dated resolution snapshot, not a promised cross-platform
lock. The verified platform is CPython3.12/Linux x86_64 CPU only.

Equivalent public-input recipe on that platform, from the product repository
root (the completed run used the identical hash-verified cached Torch archive):

```bash
PRODUCT_REPO="$PWD"
python3.12 -m venv /tmp/music3-public-cpu
PUBLIC_PY=/tmp/music3-public-cpu/bin/python
"$PUBLIC_PY" -m pip install --index-url https://pypi.org/simple \
  'setuptools==84.2.1' 'build==1.6.1' 'pytest==9.1.1' 'safetensors==0.8.0' \
  'torch @ https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl#sha256=59e78aa0c690f70734e42670036d6b541930b8eabbaa18d94e090abf14cc4d91'
"$PUBLIC_PY" -m pip install --index-url https://pypi.org/simple --no-build-isolation \
  'particle-sliders-core[hypergan-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231#subdirectory=packages/particle-sliders-core' \
  'hypergan @ git+https://github.com/HyperGAN/HyperGAN.git@5f368a3770cf5bdae0448995c9f76e29af8049f7' \
  "$PRODUCT_REPO/cohorts/hndl"
cd /tmp
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 "$PUBLIC_PY" -I -m pytest \
  -c /dev/null --import-mode=importlib "$PRODUCT_REPO/cohorts/hndl/tests" -q
"$PUBLIC_PY" -I -m pip check
```

No default/historical cohort, formulation or shared-core file was changed. No
pretrained model, GPU/audio qualification, PyPI upload, merge, deployment or release
is added. Native width4096 and the remaining acoustic/tokenizer/listening gates
remain unchanged. Direct VCS requirements still prohibit PyPI upload.
