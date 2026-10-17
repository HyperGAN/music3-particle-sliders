"""Explicit PG0.9.1 continuous profile: native-width CPU geometry, no audio."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from music3.shared import bootstrap
from test_train_infer import equal

ROOT = Path(__file__).resolve().parents[1]
PROFILE = os.environ.get("PARTICLE_SLIDERS_CONTINUOUS_PROFILE")
RUNNER = textwrap.dedent('''
    import importlib.abc,runpy,sys
    overlay=sys.argv.pop(1)
    sys.path.insert(0,overlay)
    class NoHistoricalSource(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in {'conceptmod','analysis','pg09_audio_cpu_fixtures','qualify_audio_pg09_live'}:
                raise AssertionError('historical source import: '+fullname)
    sys.meta_path.insert(0,NoHistoricalSource())
    script=sys.argv.pop(1)
    sys.argv[0]=script
    runpy.run_path(script,run_name='__main__')
''')


def test_default_release_rejects_native_width_before_model_forward():
    with pytest.raises(ValueError, match="width >=170|released PG09"):
        bootstrap().audio.require_audio_width(4096)


@pytest.mark.skipif(not PROFILE, reason="explicit separately staged continuous profile required")
@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_installed_native_width_exact_fresh_resume_under_explicit_profile(preset, tmp_path):
    import torch
    profile = Path(PROFILE).resolve()
    assert (profile / "particlegan/__init__.py").is_file()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    full, first, resumed = [tmp_path / name for name in ("full", "first", "resumed")]
    for out, until, extra in ((full, 3, []), (first, 1, []), (resumed, 3, ["--resume", str(first / "checkpoint.pt")])):
        args = ["--dummy", "--width", "4096", "--preset", preset, "--steps", str(until),
                "--save_dir", str(out), "--pg_source_profile", "alz-continuous-v1", "--penalty_input_domain", "continuous", *extra]
        result = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(profile),
                                 str(ROOT / "scripts/train_music3.py"), *args],
                                cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
    a = torch.load(full / "qualification.pt", weights_only=True)
    equal(a, torch.load(resumed / "qualification.pt", weights_only=True))
    config = a["game"]["config"]
    assert config["pg_version"] == "0.9.1"
    assert config["pg_source_profile"] == "alz-continuous-v1"
    assert config["penalty_input_domain"] == config["resolved_penalty_domain"] == "continuous"
    assert config["response_shape"] == [4096]
    assert "geometry_only" in config["backend"]["model_identity"]["token_policy"]
    assert all(metric["d_pen"] > 0 for metric in a["metrics"])
