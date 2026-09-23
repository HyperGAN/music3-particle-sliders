"""The product binds the shared stamp and does not reimplement the game."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from music3.surface import declare, load_stamp
from particle_sliders import winning_formulation

ROOT = Path(__file__).resolve().parents[1]


def _product_sources() -> list[Path]:
    roots = [ROOT / "music3", ROOT / "scripts", ROOT / "__init__.py"]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(root.rglob("*.py"))
    return files

FORBIDDEN_SNIPPETS = (
    "class RoutedMLP",
    "class GlobalMixErrorCritic",
    "class GradRegularizer",
    "class GradientPenalty",
    "class VICRegLikeLoss",
    "def locked_shared_recipe",
    "from particle_sliders import RoutedMLP",
    "from particle_sliders.reference import",
    "import conceptmod",
    "from conceptmod",
    "PARTICLE_SLIDERS_ROOT",
)


def test_stamp_is_the_shared_singleton():
    stamp = load_stamp()
    assert stamp is winning_formulation()
    assert stamp.architecture_id == "gmix"
    assert stamp.formulation_id == "particle-gmix-1600-v2"
    assert stamp.formulation_provisional is True
    assert stamp.spec["lm_target"] == "faithful_plus_neu"
    assert stamp.spec["critic"] == "gmix"


def test_declare_requires_and_allows_music_batch():
    stamp = load_stamp()
    seen = {}

    def _require(actual):
        seen["actual"] = actual
        return type(stamp).require(stamp, actual)

    stamp.require = _require  # type: ignore[method-assign]
    try:
        declared = declare(stamp, g_lr=1e-4, adv_batch=32)
    finally:
        del stamp.require
    assert seen["actual"]["g_lr"] == pytest.approx(1e-4)
    assert seen["actual"]["adv_batch"] == 32
    assert declared["critic"] == "gmix"
    assert declared["parts"] == 128


def test_require_rejects_a_local_fork():
    stamp = load_stamp()
    forked = stamp.as_dict()
    forked["aux_weights"] = dict(forked["aux_weights"], cover_weight=1.0)
    with pytest.raises(ValueError, match="drift"):
        stamp.require(forked)


def test_product_sources_do_not_copy_the_game():
    offenders = []
    for path in _product_sources():
        text = path.read_text(encoding="utf-8")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                offenders.append(f"{path.relative_to(ROOT)}: {snippet}")
    assert offenders == []
    surface = (ROOT / "music3" / "surface.py").read_text(encoding="utf-8")
    assert "from particle_sliders import winning_formulation" in surface
    assert "stamp.require" in surface or ".require(" in surface


def test_core_lock_matches_the_installed_package():
    import particle_sliders

    lock = json.loads((ROOT / "core.lock.json").read_text(encoding="utf-8"))
    assert lock["commit"] == "a119ca1ecd3d5d6c437065839d22739b04f2f4d8"
    assert lock["version"] == particle_sliders.__version__ == "0.3.0"
    package = Path(particle_sliders.__file__).resolve().parent
    for name in ("__init__.py", "formulation.py", "reference.py", "recipe.py"):
        digest = hashlib.sha256((package / name).read_bytes()).hexdigest()
        assert digest == lock["files"][name]
    pin = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert lock["commit"] in pin
    assert "subdirectory=packages/particle-sliders-core" in pin
