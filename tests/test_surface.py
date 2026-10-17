"""Verify installed shared APIs, source identity and thin product entry points."""
import ast
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest

from music3 import shared as shared_loader

ROOT = Path(__file__).resolve().parents[1]


def _isolated(code, tmp_path, *args):
    """Expose only this product; the shared package must come from installation."""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code), str(ROOT), *args],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
    )


def test_verified_installed_modules_and_environment():
    environment, search_path = dict(os.environ), sys.path[:]
    shared = shared_loader.bootstrap()
    assert dict(os.environ) == environment
    assert sys.path == search_path
    assert shared.core.ConditionalPG09.__module__ == "particle_sliders.pg09"
    assert shared.audio.Music3ARResponseAdapter.__module__ == "particle_sliders.pg09_audio"
    assert shared.audio.AudioResponseCritic.__module__ == "particle_sliders.pg09_audio"
    assert shared.native.ParticleNetwork.__module__ == "particle_sliders.music3_native"
    assert shared.cli.__name__ == "particle_sliders.pg09_audio_cli"
    lock = shared.lock
    installed = distribution(lock["distribution"])
    assert installed.version == lock["version"] == "0.6.0"
    for name in ("distribution", "version", "wheel_sha256", "particlegan_version"):
        assert shared.receipt[name] == lock[name]
    assert shared.receipt["core_commit"] == lock["commit"]
    assert shared.receipt["particlegan_version"] == "0.9.1"
    imported = shared.receipt["imported_modules"]
    assert {"particle_sliders.pg09", "particle_sliders.pg09_audio",
            "particle_sliders.music3_native", "particle_sliders.pg09_audio_cli"} <= imported.keys()
    paths = {Path(installed.locate_file(relative)).resolve(): (relative, digest)
             for relative, digest in lock["files"].items()}
    for name, record in imported.items():
        assert name == "particle_sliders" or name.startswith("particle_sliders.")
        path = Path(record["path"]).resolve()
        assert path in paths, f"unlocked installed source: {name}: {path}"
        relative, digest = paths[path]
        assert path == Path(installed.locate_file(relative)).resolve()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"] == digest
    import torch
    assert not torch.cuda.is_initialized()


def test_tampered_installed_source_lock_rejected_without_path_or_environment_mutation(tmp_path, monkeypatch):
    lock = json.loads(shared_loader.LOCK_PATH.read_text())
    lock["files"]["particle_sliders/pg09.py"] = "0" * 64
    altered = tmp_path / "core.lock.json"
    altered.write_text(json.dumps(lock))
    monkeypatch.setattr(shared_loader, "LOCK_PATH", altered)
    environment, search_path = dict(os.environ), sys.path[:]
    with pytest.raises((ValueError, RuntimeError), match="(?i)(hash|digest|mismatch|differs)"):
        shared_loader.bootstrap()
    assert dict(os.environ) == environment
    assert sys.path == search_path


@pytest.mark.parametrize("name", ["particle_sliders", "particle_sliders.pg09_audio"])
def test_foreign_cached_module_rejected_in_fresh_process(name, tmp_path):
    result = _isolated("""
        import sys
        from types import ModuleType
        sys.path.insert(0, sys.argv[1])
        foreign = ModuleType(sys.argv[2])
        foreign.__file__ = '/tmp/foreign-particle-sliders.py'
        foreign.__path__ = ['/tmp/foreign-particle-sliders']
        sys.modules[sys.argv[2]] = foreign
        from music3.shared import bootstrap
        try:
            bootstrap()
        except (ValueError, RuntimeError) as error:
            assert 'foreign' in str(error), str(error)
        else:
            raise AssertionError('foreign cached shared module accepted')
    """, tmp_path, name)
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_apis_import_source_free_in_isolated_process(tmp_path):
    result = _isolated("""
        import hashlib
        import importlib.abc
        from importlib.metadata import distribution, version
        import os
        from pathlib import Path
        import sys

        forbidden = {
            'conceptmod', 'analysis', 'pg09_audio_cpu_fixtures',
            'qualify_audio_pg09_live', 'run_audio_pg09_smoke',
            'inspect_audio_pg09_assets',
        }
        class NoHistoricalSource(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in forbidden:
                    raise AssertionError('historical source import: ' + fullname)

        sys.meta_path.insert(0, NoHistoricalSource())
        sys.path.insert(0, sys.argv[1])
        environment, search_path = dict(os.environ), sys.path[:]
        from music3.shared import bootstrap
        shared = bootstrap()
        assert environment == dict(os.environ)
        assert search_path == sys.path
        assert version('particlegan') == shared.lock['particlegan_version'] == '0.9.1'
        installed = distribution(shared.lock['distribution'])
        expected = {
            Path(installed.locate_file(relative)).resolve(): digest
            for relative, digest in shared.lock['files'].items()
        }
        for name, record in shared.receipt['imported_modules'].items():
            assert name == 'particle_sliders' or name.startswith('particle_sliders.')
            path = Path(record['path']).resolve()
            assert path in expected, (name, path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'] == expected[path]
            assert not path.is_relative_to(Path(sys.argv[1]).resolve())
        assert shared.cli.__name__ == 'particle_sliders.pg09_audio_cli'
        assert not any(name.split('.')[0] in forbidden for name in sys.modules)
        import torch
        assert not torch.cuda.is_initialized()
    """, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_product_has_no_loss_optimizer_or_training_loop_copy():
    forbidden_classes = {"RoutedMLP", "ConditionalPG09", "AudioResponseCritic", "ParticleNetwork"}
    for path in (ROOT / "music3").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.name not in forbidden_classes
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ""])
                assert not any(name.split(".")[0] in {"conceptmod", "analysis"} for name in names)
            if isinstance(node, ast.Call):
                value = ast.unparse(node.func)
                assert value not in {"torch.optim.Adam", "torch.optim.AdamW", "torch.nn.functional.mse_loss"}
                assert not value.endswith((".backward", ".step"))
    for name in ("train.py", "infer.py"):
        tree = ast.parse((ROOT / "music3" / name).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "shared.cli.main"]
        assert calls and all(call.args or call.keywords for call in calls), name


def test_lock_declares_exact_installed_wheel_and_new_lineage():
    lock = json.loads((ROOT / "core.lock.json").read_text())
    assert lock["distribution"] == "particle-sliders-core"
    assert lock["version"] == "0.6.0"
    assert lock["particlegan_version"] == "0.9.1"
    assert re.fullmatch(r"[0-9a-f]{40}", lock["commit"])
    assert re.fullmatch(r"[0-9a-f]{64}", lock["wheel_sha256"])
    assert {"particle_sliders/pg09.py", "particle_sliders/pg09_audio.py",
            "particle_sliders/music3_native.py", "particle_sliders/pg09_audio_cli.py"} <= lock["files"].keys()
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in lock["files"].values())
    assert lock.get("source_checkout_only") is not True
    requirements = (ROOT / "requirements.txt").read_text()
    assert "particlegan==0.9.1" in requirements
    assert lock["commit"] in requirements


def test_public_surface_default_and_legacy_path_rejection():
    from music3.surface import load_game_class
    shared = shared_loader.bootstrap()
    assert load_game_class() is shared.core.ConditionalPG09
    with pytest.raises(ValueError, match="source-checkout"):
        load_game_class(shared_checkout="/tmp/old-source")


@pytest.fixture
def rebuilt_distribution(tmp_path):
    """Temporary installed-file fixture; no public fetch/build is implied."""
    lock = json.loads(shared_loader.LOCK_PATH.read_text())
    installed = distribution(lock["distribution"])
    target = tmp_path / "rebuilt-installation"
    for relative in lock["files"]:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Path(installed.locate_file(relative)).read_bytes())
    info = target / "particle_sliders_core-0.6.0.dist-info"
    direct = {
        "url": lock["repository"], "subdirectory": lock["subdirectory"],
        "vcs_info": {"vcs": "git", "commit_id": lock["commit"], "requested_revision": lock["commit"]},
    }
    (info / "direct_url.json").write_text(json.dumps(direct))
    records = [*lock["files"], "particle_sliders_core-0.6.0.dist-info/direct_url.json",
               "particle_sliders_core-0.6.0.dist-info/RECORD"]
    (info / "RECORD").write_text("".join(relative + ",,\n" for relative in records))
    wheel = info / "WHEEL"
    reviewed = wheel.read_bytes()
    wheel.write_bytes(reviewed.replace(b"setuptools (79.0.1)", b"setuptools (99.0.0)"))
    assert wheel.read_bytes() != reviewed
    return target, lock, reviewed


def _rebuilt_bootstrap(target, tmp_path):
    return _isolated("""
        import json
        import sys
        sys.path.insert(0, sys.argv[1])
        sys.path.insert(0, sys.argv[2])
        from music3.shared import bootstrap
        print(json.dumps(bootstrap().receipt))
    """, tmp_path, str(target))


def test_exact_pinned_vcs_rebuild_accepts_only_generator_difference(rebuilt_distribution, tmp_path):
    target, lock, reviewed = rebuilt_distribution
    result = _rebuilt_bootstrap(target, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["installation_provenance"]["kind"] == "pinned-vcs"
    assert receipt["wheel_sha256"] == lock["wheel_sha256"]
    actual = receipt["wheel_metadata"]
    assert actual["generator"] == "setuptools (99.0.0)"
    assert actual["generator_differs"] is True
    assert actual["reviewed_sha256"] == hashlib.sha256(reviewed).hexdigest()
    assert actual["sha256"] != actual["reviewed_sha256"]
    assert actual["generator_independent_sha256"] == lock["wheel_metadata"]["generator_independent_sha256"]


@pytest.mark.parametrize("change", [
    "archive", "no-direct-url", "wrong-commit", "branch-revision",
    "wrong-repository", "wrong-subdirectory", "local-directory",
])
def test_generator_exception_requires_exact_vcs_provenance(rebuilt_distribution, tmp_path, change):
    target, lock, _ = rebuilt_distribution
    path = target / "particle_sliders_core-0.6.0.dist-info/direct_url.json"
    direct = json.loads(path.read_text())
    if change == "archive":
        direct = {"url": "file:///tmp/reviewed.whl", "archive_info": {"hashes": {"sha256": lock["wheel_sha256"]}}}
    elif change == "no-direct-url":
        path.unlink()
    elif change == "wrong-commit":
        direct["vcs_info"]["commit_id"] = "0" * 40
    elif change == "branch-revision":
        direct["vcs_info"]["requested_revision"] = "main"
    elif change == "wrong-repository":
        direct["url"] = "https://github.com/foreign/particle-sliders.git"
    elif change == "wrong-subdirectory":
        direct["subdirectory"] = "other-package"
    elif change == "local-directory":
        direct = {"url": "file:///tmp/source", "dir_info": {"editable": False}}
    if change != "no-direct-url":
        path.write_text(json.dumps(direct))
    result = _rebuilt_bootstrap(target, tmp_path)
    assert result.returncode != 0
    assert "ValueError" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("change", [
    "version", "purelib", "tag", "added-header", "reordered-headers", "relocated-generator", "body", "crlf",
    "missing-generator", "duplicate-generator", "folded-generator", "empty-generator",
    "control-generator", "nonascii-generator", "malformed-separator",
])
def test_vcs_rebuild_rejects_every_other_wheel_change(rebuilt_distribution, tmp_path, change):
    target, _, _ = rebuilt_distribution
    path = target / "particle_sliders_core-0.6.0.dist-info/WHEEL"
    contents = path.read_bytes()
    generator = b"Generator: setuptools (99.0.0)\n"
    replacements = {
        "version": (b"Wheel-Version: 1.0", b"Wheel-Version: 2.0"),
        "purelib": (b"Root-Is-Purelib: true", b"Root-Is-Purelib: false"),
        "tag": (b"Tag: py3-none-any", b"Tag: cp311-none-any"),
        "added-header": (b"\n\n", b"\nBuild: 2\n\n"),
        "reordered-headers": (b"Root-Is-Purelib: true\nTag: py3-none-any", b"Tag: py3-none-any\nRoot-Is-Purelib: true"),
        "relocated-generator": (
            generator + b"Root-Is-Purelib: true\nTag: py3-none-any",
            b"Root-Is-Purelib: true\nTag: py3-none-any\n" + generator.rstrip(b"\n"),
        ),
        "body": (b"\n\n", b"\n\nunexpected body\n"),
        "crlf": (b"\n", b"\r\n"),
        "missing-generator": (generator, b""),
        "duplicate-generator": (generator, generator + generator),
        "folded-generator": (generator, b"Generator: setuptools\n (99.0.0)\n"),
        "empty-generator": (generator, b"Generator: \n"),
        "control-generator": (generator, b"Generator: setuptools\x00\n"),
        "nonascii-generator": (generator, b"Generator: setuptools\xff\n"),
        "malformed-separator": (generator, b"Generator=setuptools (99.0.0)\n"),
    }
    before, after = replacements[change]
    assert before in contents
    path.write_bytes(contents.replace(before, after))
    result = _rebuilt_bootstrap(target, tmp_path)
    assert result.returncode != 0
    assert "WHEEL metadata" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("relative", ["particle_sliders/pg09.py", "particle_sliders_core-0.6.0.dist-info/METADATA"])
def test_vcs_generator_exception_keeps_other_file_hashes_exact(rebuilt_distribution, tmp_path, relative):
    target, _, _ = rebuilt_distribution
    path = target / relative
    path.write_bytes(path.read_bytes() + b"\n")
    result = _rebuilt_bootstrap(target, tmp_path)
    assert result.returncode != 0
    assert "installed wheel hash differs" in result.stderr, result.stdout + result.stderr


@pytest.mark.parametrize("change", ["reviewed-text", "semantic-digest", "missing-contract"])
def test_wheel_semantic_contract_is_bound_to_exact_reviewed_bytes(tmp_path, monkeypatch, change):
    lock = json.loads(shared_loader.LOCK_PATH.read_text())
    if change == "reviewed-text":
        lock["wheel_metadata"]["reviewed_text"] = lock["wheel_metadata"]["reviewed_text"].replace("79.0.1", "99.0.0")
    elif change == "semantic-digest":
        lock["wheel_metadata"]["generator_independent_sha256"] = "0" * 64
    else:
        del lock["wheel_metadata"]
    altered = tmp_path / "core.lock.json"
    altered.write_text(json.dumps(lock))
    monkeypatch.setattr(shared_loader, "LOCK_PATH", altered)
    with pytest.raises(ValueError, match="(?i)reviewed WHEEL"):
        shared_loader.bootstrap()
