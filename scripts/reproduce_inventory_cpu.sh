#!/usr/bin/env bash
# Offline reproduction in the selected executor, using read-only CPU dependency
# runtimes and exact provided wheels. No pretrained files or copied test suites.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COORDINATOR_DIR="${COORDINATOR_DIR:-/ml2/codex-storage/future-workspaces/2026-10-31/task-3}"
RUN_DIR="${RUN_DIR:-$REPO_DIR/../evidence/review-repro}"
STABLE_SOURCE_PYTHON="${STABLE_SOURCE_PYTHON:-$COORDINATOR_DIR/.venv082/bin/python}"
HNDL_SOURCE_PYTHON="${HNDL_SOURCE_PYTHON:-$COORDINATOR_DIR/.venv-joint/bin/python}"
STABLE_CORE_WHEEL="${STABLE_CORE_WHEEL:-/ml2/codex-storage/future-workspaces/2026-10-21/task-2/evidence/checkpoint-ownership-packaging/091/dist/particle_sliders_core-0.8.2-py3-none-any.whl}"
HNDL_CORE_WHEEL="$COORDINATOR_DIR/evidence/joint-wheels/particle_sliders_core-0.12.0-py3-none-any.whl"

test ! -e "$RUN_DIR/stable"
test ! -e "$RUN_DIR/hndl"
mkdir -p "$RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2

"$STABLE_SOURCE_PYTHON" -m venv "$RUN_DIR/stable"
"$HNDL_SOURCE_PYTHON" -m venv "$RUN_DIR/hndl"
STABLE_PYTHON="$RUN_DIR/stable/bin/python"
HNDL_PYTHON="$RUN_DIR/hndl/bin/python"
STABLE_SITE="$("$STABLE_PYTHON" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
HNDL_SITE="$("$HNDL_PYTHON" -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"

"$STABLE_SOURCE_PYTHON" -I "$REPO_DIR/scripts/materialize_cpu_dependencies.py" "$STABLE_SITE" \
    torch transformers build pytest safetensors pyyaml > "$RUN_DIR/stable-dependencies.json"
"$HNDL_SOURCE_PYTHON" -I "$REPO_DIR/scripts/materialize_cpu_dependencies.py" "$HNDL_SITE" \
    torch transformers build pytest safetensors pyyaml foldlitedb > "$RUN_DIR/hndl-dependencies.json"

"$STABLE_PYTHON" -I - "$STABLE_CORE_WHEEL" "$HNDL_CORE_WHEEL" <<'PY'
import hashlib, sys
from pathlib import Path
for path, expected in zip(sys.argv[1:], [
    '50a6a7d06240254357569fd3cacbb5cf5e9d3c1771f642dfb719733b5711f32a',
    '9afad3afc1f6c653a574074dc14f0da1f1c43975c31ca45c4eac049df0b0b4ad',
]):
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
PY

"$STABLE_PYTHON" -m pip install --no-index --no-deps \
    "$COORDINATOR_DIR/evidence/wheels/setuptools-84.2.1-py3-none-any.whl" \
    "$COORDINATOR_DIR/evidence/wheels/particlegan-0.9.1-py3-none-any.whl" "$STABLE_CORE_WHEEL"
"$HNDL_PYTHON" -m pip install --no-index --no-deps \
    "$COORDINATOR_DIR/evidence/wheels/setuptools-84.2.1-py3-none-any.whl" \
    "$COORDINATOR_DIR/evidence/wheels/particlegan-0.9.4-py3-none-any.whl" \
    "$COORDINATOR_DIR/evidence/wheels/hndl-0.10.0-py3-none-any.whl" "$HNDL_CORE_WHEEL"

git clone --no-hardlinks "$COORDINATOR_DIR/repos/hypergan-develop-20261031" "$RUN_DIR/HyperGAN"
git -C "$RUN_DIR/HyperGAN" checkout --detach 5f368a3770cf5bdae0448995c9f76e29af8049f7
test -z "$(git -C "$RUN_DIR/HyperGAN" status --porcelain)"
"$HNDL_PYTHON" -m pip wheel --no-index --no-build-isolation --no-deps \
    "$RUN_DIR/HyperGAN" -w "$RUN_DIR/runtime-wheels"
"$HNDL_PYTHON" -m pip install --no-index --no-deps "$RUN_DIR/runtime-wheels/hypergan-2.0.0-py3-none-any.whl"

"$STABLE_PYTHON" -m build --no-isolation "$REPO_DIR" --outdir "$RUN_DIR/stable-dist"
"$HNDL_PYTHON" -m build --no-isolation "$REPO_DIR/cohorts/hndl" --outdir "$RUN_DIR/hndl-dist"
"$STABLE_PYTHON" -m pip install --no-index --no-deps "$RUN_DIR/stable-dist/music3_particle_sliders-0.2.0-py3-none-any.whl"
"$HNDL_PYTHON" -m pip install --no-index --no-deps "$RUN_DIR/hndl-dist/music3_particle_sliders_hndl-0.1.0-py3-none-any.whl"

cd /tmp
"$STABLE_PYTHON" -I -m pytest -c /dev/null --import-mode=importlib \
    "$REPO_DIR/tests/test_response_proposal.py" "$REPO_DIR/tests/test_inventory_contracts.py" -q \
    | tee "$RUN_DIR/stable-tests.txt"
"$HNDL_PYTHON" -I -m pytest -c /dev/null --import-mode=importlib "$REPO_DIR/cohorts/hndl/tests" -q \
    | tee "$RUN_DIR/hndl-tests.txt"
"$STABLE_PYTHON" -I -m pytest -c /dev/null --import-mode=importlib "$REPO_DIR/tests/test_configs.py" \
    "$REPO_DIR/tests/test_surface.py::test_lock_declares_exact_installed_wheel_and_new_lineage" \
    "$REPO_DIR/tests/test_surface.py::test_product_has_no_loss_optimizer_or_training_loop_copy" -q \
    | tee "$RUN_DIR/source-tests.txt"
"$STABLE_PYTHON" -I "$REPO_DIR/scripts/check_inventory_artifacts.py" "$RUN_DIR/stable-dist" "$RUN_DIR/hndl-dist" \
    --stable-site "$STABLE_SITE" --hndl-site "$HNDL_SITE" \
    | tee "$RUN_DIR/packaging.json"
"$STABLE_PYTHON" -I -m pip check | tee "$RUN_DIR/stable-pip-check.txt"
"$HNDL_PYTHON" -I -m pip check | tee "$RUN_DIR/hndl-pip-check.txt"
