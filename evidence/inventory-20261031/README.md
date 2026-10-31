# Music3 inventory maintenance receipt — 2026-10-31

Public base verified: `11b683ad7912b421ea3cd13642dda9356a73e2aa`.
Reused installer repair: `6643aea7bea3fb2276be562cdb626316cd44a2ea`.
The original inventory/receipts remain in coordinator task3. No historical
source lock, checkpoint identity or model identity was rewritten.

Verified fixes: default requirements and packaged lifecycle use the same exact
S082 core; wheel installs include the historical release catalog; VCS admission
checks the exact package subdirectory; ordinary LoRA distillation delegates the
fit to shared core and records approximation/source/calibration/factor provenance.

The separate `cohorts/hndl` package executes actual shared HNDL projections on
genuine random Qwen3 q/k/v/o hosts and the existing AR readout architecture via
the verified HyperGAN builder. Its exact core0.12 candidate is
`4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231`, PG0.9.4, HyperGAN develop
`5f368a3770cf5bdae0448995c9f76e29af8049f7`, HNDL0.10.0. Its source map is
bound to the exact owner-built wheel, not a version-only assertion.

Latest review follow-up: [REVIEW-FIXES.md](REVIEW-FIXES.md). It supersedes the
initial copied-suite commands with fresh venvs and repository test paths.

Completed checks on installed wheels, Python3.12.15 / Torch2.14.1+cpu:

- Retained core0.8.2/PG0.9.1 lane: **37 passed** (22 existing +15 new).
  Actual particle gradients, frozen weights, zero replay, checkpoint restore,
  ordinary factor strict load/on/off, bounded collection, failure cleanup,
  source provenance, response/inference ownership and resume are exercised.
- Separate real HyperGAN/HNDL lane: **24 passed** after review fixes. Matched-weight whole Qwen3
  responses and particle VJPs, AR critic outputs and first/second derivatives,
  both PG presets, exact split resume, export response reload, state rejection,
  cleanup conflict retry, partial construction rollback and native width refusal.
- Source/config/history contracts: **7 passed**.
- An actual trained S082/PG0.9.1 checkpoint is rejected by the new game with no
  state mutation. Wheel ZIP checks, wheel/sdist package data and unchanged
  historical lock bytes pass. Both isolated environments pass `pip check`.

Commands are recorded in `commands.txt` and `scripts/reproduce_inventory_cpu.sh`.
The latest run creates fresh virtual environments, explicitly installs both exact
core wheels, builds clean HyperGAN at its exact commit, and runs repository tests
from `/tmp` using `python -I`; product and core imports resolve to installed wheels.
The source/config tests intentionally read repository cards. Dependency binaries
reuse selected read-only CPU runtimes, with versions/origins recorded; this is
not a fresh pretrained dependency resolution or Python-version matrix. Initial
copied-suite and failed attempts remain as historical evidence in the task workspace.

External raw logs, wheels, runtime metadata, full diff and exact remote receipt
are under `/ml2/codex-storage/future-workspaces/2026-10-31/task-11/evidence/`.
No generated tensor/adapter/model file is committed or published.

Remaining gates: the core0.12 candidate is not publicly fetchable yet; its
separate candidate dependency records an intended immutable source. Native
width4096 is still refused before allocation. Pretrained/tokenizer execution,
GPU/acoustic sampling, listening quality, native distillation and new ComfyUI
exports are unrun. The HNDL candidate is a bounded CPU integration, not a new
default or a native release. No mathematical gate or loss was changed.

Release impact: source-only changes; existing package version0.2.0 retained and
new candidate distribution0.1.0 remains unpublished. Both manifests contain direct
VCS dependencies and cannot be uploaded to PyPI. No workflow, package/model
release, deployment, security/driver/permission change or GPU job is added.
Parent owns Claude/Ada review and merge. No independent review/merge is claimed.
