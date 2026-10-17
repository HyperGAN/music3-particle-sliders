"""Verify real shared-code execution and fail-closed source dependencies."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from music3.shared import bootstrap, _verify_files

ROOT = Path(__file__).resolve().parents[1]


def test_verified_shared_modules_and_environment():
    before = dict(os.environ)
    shared = bootstrap()
    assert dict(os.environ) == before
    assert shared.core.ConditionalPG09.__module__ == "particle_sliders.pg09"
    assert shared.audio.Music3ARResponseAdapter.__module__ == "conceptmod.textsliders.pg09_audio"
    assert shared.audio.AudioResponseCritic.__module__ == "conceptmod.textsliders.pg09_audio"
    assert shared.receipt["core_commit"] == "3e94b13eb5d6cd1ceee3f018f019a68603f35696"
    assert shared.receipt["shared_commit"] == "9262a930cd351459e93ac33cc542fa379c12fe52"
    for key in ("shared", "core"):
        root = shared.shared_checkout if key == "shared" else shared.core_checkout
        _verify_files(root, shared.lock[key])
    import torch
    assert not torch.cuda.is_initialized()


def test_tampered_source_rejected_before_import(tmp_path):
    (tmp_path / "x.py").write_text("bad")
    with pytest.raises(ValueError, match="hash differs"):
        _verify_files(tmp_path, {"files": {"x.py": hashlib.sha256(b"good").hexdigest()}})
    with pytest.raises(ValueError, match="missing locked"):
        _verify_files(tmp_path, {"files": {"../escape.py": "bad"}})


def test_foreign_import_rejected():
    code = """
import sys, types
sys.modules['particle_sliders'] = types.SimpleNamespace(__file__='/tmp/foreign.py')
from music3.shared import bootstrap
try:
    bootstrap()
except ValueError as e:
    assert 'foreign' in str(e)
else:
    raise AssertionError('foreign module accepted')
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_product_has_no_loss_optimizer_or_training_loop_copy():
    for path in (ROOT / "music3").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.name not in {"RoutedMLP", "ConditionalPG09", "AudioResponseCritic", "ParticleNetwork"}
            if isinstance(node, ast.Call):
                value = ast.unparse(node.func)
                assert value not in {"torch.optim.Adam", "torch.optim.AdamW", "torch.nn.functional.mse_loss"}
                assert not value.endswith(".backward")
    train = (ROOT / "music3/train.py").read_text()
    assert "shared.worker.main()" in train
    assert "game.step()" not in train
    infer = (ROOT / "music3/infer.py").read_text()
    assert "game.load_export_state_dict(payload)" in infer
    assert "adapter.sample(p, scale=s)" in infer


def test_lock_declares_integration_sources_not_a_guessed_public_release():
    lock = json.loads((ROOT / "core.lock.json").read_text())
    assert lock["source_checkout_only"] is True
    assert lock["entry"] == "ConditionalPG09"
    assert lock["particlegan_version"] == "0.9.0"
    assert "git+" not in (ROOT / "requirements.txt").read_text()
