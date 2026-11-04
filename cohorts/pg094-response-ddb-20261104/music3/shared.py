"""Load the locked public wheel APIs without consulting a source checkout.

The caller installs reviewed wheels in its interpreter. Bootstrap verifies that
installation before importing it; it never installs packages or changes Python's
search path. This is a new source lineage, not a compatibility loader for old
checkpoints bound to repository-internal audio modules.
"""
from __future__ import annotations

import hashlib
import importlib
from importlib import metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
from types import SimpleNamespace


LOCK_PATH = Path(__file__).resolve().parents[1] / "core.lock.json"
API_WORKFLOWS_MODULE = "particle_sliders.pg09_audio_cli"
_PACKAGE = "particle_sliders"
_API_MODULES = {
    "core": _PACKAGE,
    "audio": "particle_sliders.pg09_audio",
    "native": "particle_sliders.music3_native",
    "cli": API_WORKFLOWS_MODULE,
}
_METADATA_ROOT = "particle_sliders_core-0.6.0.dist-info"
_WHEEL_PATH = f"{_METADATA_ROOT}/WHEEL"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_module(name: str) -> bool:
    return name == _PACKAGE or name.startswith(_PACKAGE + ".")


def _relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("wheel lock paths must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"invalid wheel lock path: {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"noncanonical wheel lock path: {value!r}")
    return path


def _wheel_metadata(contents: bytes) -> tuple[bytes, str]:
    """Normalize one Generator value, preserving header order and other bytes."""
    if not contents.endswith(b"\n\n") or b"\r" in contents:
        raise ValueError("malformed WHEEL metadata header terminator or line endings")
    retained, generators = [], []
    for line in contents[:-2].split(b"\n"):
        # No folding, bodies, controls, empty values or ambiguous separators.
        match = re.fullmatch(rb"([A-Za-z0-9][A-Za-z0-9-]*): ([!-~](?:[ -~]*[!-~])?)", line)
        if match is None:
            raise ValueError("malformed WHEEL metadata header")
        name, value = match.groups()
        if name == b"Generator":
            generators.append(value.decode("ascii"))
            retained.append(b"Generator: <build-generator>")
        else:
            retained.append(line)
    if len(generators) != 1:
        raise ValueError("WHEEL metadata must contain exactly one canonical Generator header")
    return b"\n".join(retained) + b"\n\n", generators[0]


def _wheel_metadata_contract(lock: dict) -> str:
    contract = lock.get("wheel_metadata")
    if not isinstance(contract, dict) or not isinstance(contract.get("reviewed_text"), str):
        raise ValueError("core.lock.json requires the exact reviewed WHEEL metadata")
    try:
        reviewed = contract["reviewed_text"].encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("reviewed WHEEL metadata must contain ASCII headers") from exc
    if hashlib.sha256(reviewed).hexdigest() != lock["files"][_WHEEL_PATH]:
        raise ValueError("reviewed WHEEL metadata differs from the raw wheel file lock")
    normalized, _ = _wheel_metadata(reviewed)
    digest = hashlib.sha256(normalized).hexdigest()
    if contract.get("generator_independent_sha256") != digest:
        raise ValueError("reviewed WHEEL metadata semantic SHA-256 differs")
    return digest


def _load_lock() -> tuple[dict, str]:
    contents = LOCK_PATH.read_bytes()
    lock = json.loads(contents)
    if not isinstance(lock, dict):
        raise ValueError("core.lock.json must contain an object")
    for field, expected in {
        "distribution": "particle-sliders-core", "version": "0.6.0",
        "particlegan_version": "0.9.1",
    }.items():
        if lock.get(field) != expected:
            raise ValueError(f"core.lock.json requires {field}={expected!r}")
    if not isinstance(lock.get("commit"), str) or not _COMMIT.fullmatch(lock["commit"]):
        raise ValueError("core.lock.json requires an immutable full source commit")
    if not isinstance(lock.get("wheel_sha256"), str) or not _SHA256.fullmatch(lock["wheel_sha256"]):
        raise ValueError("core.lock.json requires a wheel SHA-256")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("core.lock.json requires a nonempty installed-file hash map")
    metadata_roots = set()
    for relative, expected in files.items():
        path = _relative_path(relative)
        if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
            raise ValueError(f"invalid locked SHA-256: {relative}")
        if path.parts[0].endswith(".dist-info"):
            metadata_roots.add(path.parts[0])
        elif path.parts[0] != _PACKAGE:
            raise ValueError(f"file outside the locked public distribution: {relative}")
    if metadata_roots != {_METADATA_ROOT}:
        raise ValueError("core.lock.json must lock the expected wheel .dist-info directory")
    for relative in (
        "particle_sliders/__init__.py", "particle_sliders/pg09.py",
        "particle_sliders/pg09_audio.py", "particle_sliders/music3_native.py",
        "particle_sliders/pg09_audio_cli.py",
        f"{_METADATA_ROOT}/METADATA", f"{_METADATA_ROOT}/WHEEL",
    ):
        if relative not in files:
            raise ValueError(f"missing required wheel lock entry: {relative}")
    # Pip rewrites RECORD and creates the other files during installation. Their
    # bytes are not stable wheel inputs; direct_url is checked against the pin.
    generated = {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}
    if any(f"{_METADATA_ROOT}/{name}" in files for name in generated):
        raise ValueError("lock stable wheel files, not installer-generated metadata")
    _wheel_metadata_contract(lock)
    return lock, hashlib.sha256(contents).hexdigest()


def _installed_files(dist, lock: dict, *, allow_wheel_generator_change=False) -> tuple[Path, dict[str, Path]]:
    root = Path(dist.locate_file("")).resolve()
    if not root.is_dir():
        raise ValueError("locked distribution has no installed directory")
    records = dist.files
    if records is None:
        raise ValueError("distribution has no installed file record; editable/source installs are unsupported")
    registered = {str(path) for path in records}
    paths = {}
    for relative, expected in lock["files"].items():
        if relative not in registered:
            raise ValueError(f"locked file is not owned by the installed distribution: {relative}")
        path = Path(dist.locate_file(relative)).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"missing or external installed wheel file: {relative}")
        actual = _sha256(path)
        if relative == _WHEEL_PATH:
            normalized, _ = _wheel_metadata(path.read_bytes())
            if hashlib.sha256(normalized).hexdigest() != lock["wheel_metadata"]["generator_independent_sha256"]:
                raise ValueError(f"installed WHEEL metadata differs outside Generator: {path}")
            if actual != expected and allow_wheel_generator_change:
                paths[relative] = path
                continue
        if actual != expected:
            raise ValueError(f"installed wheel hash differs: {path}")
        paths[relative] = path
    for relative in registered:
        path = PurePosixPath(relative)
        if path.parts and path.parts[0] == _PACKAGE and "__pycache__" not in path.parts:
            if relative not in paths:
                raise ValueError(f"installed public package file is absent from the lock: {relative}")
    return root, paths


def _repository_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("installed Git repository URL must be a string")
    url = url.rstrip("/")
    return url[:-4] if url.endswith(".git") else url


def _verify_installation_origin(dist, lock: dict, root: Path) -> dict:
    path = Path(dist.locate_file(f"{_METADATA_ROOT}/direct_url.json")).resolve()
    if not path.exists():
        return {"kind": "wheel", "direct_url": None, "archive_hash_verified": False}
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("installed direct_url.json is outside the locked distribution")
    direct = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(direct, dict) or "dir_info" in direct:
        raise ValueError("editable and source-tree installs are unsupported; install the locked distribution")
    provenance = {"path": str(path), "sha256": _sha256(path)}
    if "vcs_info" in direct:
        vcs = direct["vcs_info"]
        repository = lock.get("repository")
        if "archive_info" in direct or not isinstance(vcs, dict):
            raise ValueError("installed Git provenance is malformed")
        if vcs.get("vcs") != "git" or vcs.get("commit_id") != lock["commit"]:
            raise ValueError("installed Git source commit differs from the lock")
        if vcs.get("requested_revision") != lock["commit"]:
            raise ValueError("installed Git requirement must request the exact locked commit")
        if not isinstance(repository, str) or _repository_url(repository) != "https://github.com/HyperGAN/particle-sliders":
            raise ValueError("Git installation requires the locked public particle-sliders repository")
        if _repository_url(direct.get("url")) != _repository_url(repository):
            raise ValueError("installed Git repository differs from the lock")
        if "subdirectory" in lock and direct.get("subdirectory") != lock["subdirectory"]:
            raise ValueError("installed Git package subdirectory differs from the lock")
        return {"kind": "pinned-vcs", **provenance, "repository": repository, "commit": lock["commit"]}
    archive = direct.get("archive_info")
    if not isinstance(archive, dict):
        raise ValueError("installed direct_url.json does not identify a wheel archive")
    hashes = archive.get("hashes", {})
    if not isinstance(hashes, dict):
        raise ValueError("installed archive hashes must be an object")
    digests = []
    if "sha256" in hashes:
        digests.append(hashes["sha256"])
    if "hash" in archive:
        legacy = archive["hash"]
        if not isinstance(legacy, str) or not legacy.startswith("sha256="):
            raise ValueError("installed archive hash must use SHA-256")
        digests.append(legacy.split("=", 1)[1])
    if not digests or any(digest != lock["wheel_sha256"] for digest in digests):
        raise ValueError("installed archive SHA-256 differs from the locked wheel")
    return {"kind": "wheel", **provenance, "archive_sha256": lock["wheel_sha256"], "archive_hash_verified": True}


def _module_path(name: str, paths: dict[str, Path]) -> Path:
    relative = name.replace(".", "/")
    if relative + ".py" in paths:
        return paths[relative + ".py"]
    if relative + "/__init__.py" in paths:
        return paths[relative + "/__init__.py"]
    raise ValueError(f"public API module is not locked in the wheel: {name}")


def _verify_origin(name: str, origin, search_paths, paths: dict[str, Path]) -> Path:
    expected = _module_path(name, paths)
    if not isinstance(origin, str) or Path(origin).resolve() != expected:
        raise ValueError(f"foreign or source-tree public module: {name} ({origin})")
    if search_paths is not None:
        actual = tuple(Path(path).resolve() for path in search_paths)
        if expected.name != "__init__.py" or actual != (expected.parent,):
            raise ValueError(f"foreign public package search path: {name}")
    elif expected.name == "__init__.py":
        raise ValueError(f"locked public package has no package search path: {name}")
    return expected


def _verify_imports(paths: dict[str, Path], lock: dict) -> dict:
    imported = {}
    relative_paths = {path: relative for relative, path in paths.items()}
    for name, module in tuple(sys.modules.items()):
        if not _package_module(name):
            continue
        path = _verify_origin(name, getattr(module, "__file__", None), getattr(module, "__path__", None), paths)
        spec = getattr(module, "__spec__", None)
        if spec is None:
            raise ValueError(f"public module has no verified import specification: {name}")
        _verify_origin(name, spec.origin, spec.submodule_search_locations, paths)
        digest = _sha256(path)
        if digest != lock["files"][relative_paths[path]]:
            raise ValueError(f"imported public source hash differs: {path}")
        imported[name] = {"path": str(path), "sha256": digest}
    return imported


def _import_public(name: str, paths: dict[str, Path]):
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ValueError(f"locked public API is not installed: {name}")
    _verify_origin(name, spec.origin, spec.submodule_search_locations, paths)
    return importlib.import_module(name)


def bootstrap(*legacy_args, **legacy_kwargs) -> SimpleNamespace:
    """Return verified installed public modules and JSON-safe source receipts.

    Historical source-checkout arguments are rejected. Install the wheel named
    by core.lock.json in a separate interpreter instead.
    """
    if legacy_args or legacy_kwargs:
        raise ValueError("source-checkout bootstrap arguments are unsupported; install the locked wheel")
    lock, lock_sha256 = _load_lock()
    try:
        dist = metadata.distribution(lock["distribution"])
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("install the locked particle-sliders-core wheel in this interpreter") from exc
    if dist.version != lock["version"]:
        raise ValueError(f"installed {lock['distribution']} must be version {lock['version']}")
    name = dist.metadata.get("Name", "")
    if re.sub(r"[-_.]+", "-", name).lower() != lock["distribution"]:
        raise ValueError("installed distribution metadata name differs from the wheel lock")
    root = Path(dist.locate_file("")).resolve()
    installation_provenance = _verify_installation_origin(dist, lock, root)
    root, paths = _installed_files(
        dist, lock, allow_wheel_generator_change=installation_provenance["kind"] == "pinned-vcs",
    )
    _verify_imports(paths, lock)
    try:
        pg_version = metadata.version("particlegan")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("the locked shared API requires particlegan==0.9.1") from exc
    if pg_version != lock["particlegan_version"]:
        raise RuntimeError("the locked shared API requires particlegan==0.9.1")
    previous_env = dict(os.environ)
    previous_modules = set(sys.modules)
    try:
        loaded = {key: _import_public(name, paths) for key, name in _API_MODULES.items()}
        imported = _verify_imports(paths, lock)
    except BaseException:
        for name in sorted(set(sys.modules) - previous_modules, key=len, reverse=True):
            if _package_module(name):
                module = sys.modules.pop(name)
                parent_name, _, child_name = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and getattr(parent, child_name, None) is module:
                    delattr(parent, child_name)
        raise
    finally:
        # Native helpers must not change the caller's cache or device defaults.
        for name in set(os.environ) - set(previous_env):
            del os.environ[name]
        os.environ.update(previous_env)
    wheel_contents = paths[_WHEEL_PATH].read_bytes()
    wheel_normalized, wheel_generator = _wheel_metadata(wheel_contents)
    wheel_sha256 = hashlib.sha256(wheel_contents).hexdigest()
    receipt = {
        "distribution": lock["distribution"], "version": dist.version,
        "core_commit": lock["commit"], "wheel_sha256": lock["wheel_sha256"],
        "particlegan_version": pg_version, "source_lock_sha256": lock_sha256,
        "installation_root": str(root), "installation_provenance": installation_provenance,
        "wheel_metadata": {
            "path": str(paths[_WHEEL_PATH]), "sha256": wheel_sha256,
            "reviewed_sha256": lock["files"][_WHEEL_PATH], "generator": wheel_generator,
            "generator_independent_sha256": hashlib.sha256(wheel_normalized).hexdigest(),
            "generator_differs": wheel_sha256 != lock["files"][_WHEEL_PATH],
        },
        "imported_modules": imported,
        "core_source": {"path": str(paths["particle_sliders/pg09.py"]), "sha256": lock["files"]["particle_sliders/pg09.py"]},
        "audio_source": imported[_API_MODULES["audio"]],
        "native_source": imported[_API_MODULES["native"]],
        "workflows_source": imported[_API_MODULES["cli"]],
    }
    return SimpleNamespace(**loaded, workflows=loaded["cli"], receipt=receipt, lock=lock)


shared_modules = bootstrap
