"""Load the reviewed shared audio integration without installing or copying it.

Set PARTICLE_SLIDERS_SHARED_CHECKOUT and PARTICLE_SLIDERS_CORE_CHECKOUT for
portable clones. Every locked file is checked before imports; only source roots
are added to sys.path. Native Transformers/torch/ParticleGAN stay in the caller's
interpreter. The optional continuous PG profile must be selected by the caller.
"""
from __future__ import annotations

import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace


LOCK_PATH = Path(__file__).resolve().parents[1] / "shared.lock.json"
DEFAULT_SHARED_CHECKOUT = Path("/mnt/nvme/sliders/audio-adapters-pg09-20261017")
DEFAULT_CORE_CHECKOUT = Path("/mnt/nvme/sliders/audio-core-pin-86fc4e0-20261017")
_SHARED_PREFIXES = ("conceptmod", "analysis")
_SCRIPT_MODULES = (
    "inspect_audio_pg09_assets", "pg09_audio_cpu_fixtures",
    "qualify_audio_pg09_live", "run_audio_pg09_smoke",
)


def _verify_files(root: Path, record: dict) -> None:
    for relative, expected in record["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing locked shared source: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"shared source hash differs: {path}")


def _managed_module(name: str) -> bool:
    prefixes = ("particle_sliders", "concept_slider_core", *_SHARED_PREFIXES)
    return name in _SCRIPT_MODULES or any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def _verify_imports(shared_root: Path, core_root: Path, lock: dict) -> dict:
    """Reject cached foreign modules as well as files outside the source lock."""
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if name in ("particle_sliders", "concept_slider_core") or name.startswith(("particle_sliders.", "concept_slider_core.")):
            root, record = core_root, lock["core"]
        elif name in _SCRIPT_MODULES or any(name == prefix or name.startswith(prefix + ".") for prefix in _SHARED_PREFIXES):
            root, record = shared_root, lock["shared"]
        else:
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            raise ValueError(f"shared module has no verified source origin: {name}")
        path = Path(origin).resolve()
        if not path.is_relative_to(root) or path.relative_to(root).as_posix() not in record["files"]:
            raise ValueError(f"foreign or unlocked shared module already imported: {name} ({path})")
        for package_path in getattr(module, "__path__", ()):
            if not Path(package_path).resolve().is_relative_to(root):
                raise ValueError(f"foreign shared package search path: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["files"][path.relative_to(root).as_posix()]:
            raise ValueError(f"imported shared source hash differs: {path}")
        imported[name] = dict(path=str(path), sha256=digest)
    return imported


def bootstrap(shared_checkout=None, core_checkout=None) -> SimpleNamespace:
    """Return the original shared modules after verifying both source trees.

    Copies of the exact source trees are supported; a .git directory is not
    required. The commit references in shared.lock.json describe the reviewed
    local integration, whose upstream publication and merge remain pending.
    """
    shared_root = Path(shared_checkout or os.environ.get(
        "PARTICLE_SLIDERS_SHARED_CHECKOUT", DEFAULT_SHARED_CHECKOUT)).expanduser().resolve()
    core_root = Path(core_checkout or os.environ.get(
        "PARTICLE_SLIDERS_CORE_CHECKOUT", DEFAULT_CORE_CHECKOUT)).expanduser().resolve()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    _verify_files(shared_root, lock["shared"])
    _verify_files(core_root, lock["core"])
    _verify_imports(shared_root, core_root, lock)
    if version("particlegan") != "0.9.0":
        raise RuntimeError("shared Music3 integration requires particlegan==0.9.0")
    roots = (core_root / "packages/particle-sliders-core/src", shared_root, shared_root / "scripts")
    previous_path = sys.path[:]
    previous_env = dict(os.environ)
    previous_modules = set(sys.modules)
    try:
        sys.path[:0] = [str(path) for path in roots if str(path) not in sys.path]
        core = importlib.import_module("particle_sliders")
        audio = importlib.import_module("conceptmod.textsliders.pg09_audio")
        bridge = importlib.import_module("conceptmod.textsliders.music3_particle_bridge")
        fixtures = importlib.import_module("pg09_audio_cpu_fixtures")
        worker = importlib.import_module("qualify_audio_pg09_live")
        imported_modules = _verify_imports(shared_root, core_root, lock)
    except BaseException:
        sys.path[:] = previous_path
        for name in sorted(sys.modules.keys() - previous_modules, key=len, reverse=True):
            if _managed_module(name):
                sys.modules.pop(name, None)
        raise
    finally:
        # Historical tokenizer helpers set cache/device defaults during import.
        # Reusing their functions must preserve the native caller environment.
        for name in os.environ.keys() - previous_env.keys():
            del os.environ[name]
        os.environ.update(previous_env)
    return SimpleNamespace(
        core=core, audio=audio, bridge=bridge, fixtures=fixtures, worker=worker,
        shared_checkout=shared_root, core_checkout=core_root,
        worker_path=shared_root / "scripts/qualify_audio_pg09_live.py",
        smoke_path=shared_root / "scripts/run_audio_pg09_smoke.py",
        fixture_path=shared_root / "scripts/pg09_audio_cpu_fixtures.py",
        inspect_path=shared_root / "scripts/inspect_audio_pg09_assets.py",
        receipt=dict(
            shared_commit=lock["shared"]["commit"], core_commit=lock["core"]["commit"],
            shared_checkout=str(shared_root), core_checkout=str(core_root),
            source_lock_sha256=hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
            particlegan_version="0.9.0", integration_target=lock["integration_target"],
            imported_modules=imported_modules,
        ),
        lock=lock,
    )


shared_modules = bootstrap
