"""Product CLI regression tests use actual shared PG09/random native Qwen3."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from music3.infer import infer, parse_args as parse_infer, _scales
from music3.train import parse_args, train
from music3.shared import bootstrap

ROOT = Path(__file__).resolve().parents[1]


def run(script, args, check=True):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                            cwd=ROOT, text=True, capture_output=True, timeout=90)
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("script", ["train_music3.py", "train_lm_slider_music3.py", "train_lora_music3.py", "train_lora_music3_particle.py", "train_encoder_music3.py", "infer_music3.py"])
def test_help_does_not_need_model_or_shared_import(script):
    result = run(script, ["--help"])
    assert "MiniMaxAI/MiniMax-Music3" in result.stdout
    assert "--dummy" in result.stdout


@pytest.mark.parametrize("script", ["train_lora_music3.py", "train_encoder_music3.py"])
def test_unqualified_hosts_fail_instead_of_training_a_fake_host(script, tmp_path):
    out = tmp_path / "unsupported"
    result = run(script, ["--dummy", "--save_dir", str(out)], check=False)
    assert result.returncode != 0 and "only the LM/particle binding" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_fresh_process_resume_and_real_export_reload(preset, tmp_path):
    import torch
    full, first, resumed = [tmp_path / name for name in ("full", "first", "resumed")]
    for out, steps, extra in ((full, 3, []), (first, 1, []), (resumed, 3, ["--resume", str(first / "checkpoint.pt")])):
        run("train_music3.py", ["--dummy", "--preset", preset, "--steps", str(steps), "--save_dir", str(out), *extra])
    a = torch.load(full / "qualification.pt", weights_only=True)
    b = torch.load(resumed / "qualification.pt", weights_only=True)
    equal(a, b)
    receipt = json.loads((full / "product-integration.json").read_text())
    assert receipt["status"] == "bounded_training_software_checks_passed"
    assert receipt["frozen_base_unchanged"] and receipt["actual_bank_weights_changed"]
    assert receipt["bank_response_gradient_l1"] > 0
    assert receipt["zero_replay_exact"] and receipt["export_replay_exact"]
    assert receipt["peak_allocated_bytes"] == 0 and not receipt["audio_rendered"]
    for metric in a["metrics"]:
        assert metric["g_loss"] == pytest.approx(metric["g_adv"] + metric["prior_regularization"])
        assert metric["d_loss"] == pytest.approx(metric["d_adv"] + metric["d_pen"])
        assert metric["d_pen"] > 0
    probes = tmp_path / "probes"
    run("infer_music3.py", ["--dummy", "--preset", preset, "--weights", str(full / "adapter-export.pt"), "--out_dir", str(probes)])
    values = torch.load(probes / "hidden-probes.pt", weights_only=True)
    equal(values["probes"][0]["responses"], values["neutral"])
    equal(values["probes"][-1]["responses"], a["probes"])
    card = json.loads((probes / "probe-card.json").read_text())
    assert card["scales"] == [0, .5, 1]
    assert card["rng_preserved"] and card["zero_replay_exact"]
    assert not card["rendered_audio"] and not card["quality_qualified"]
    assert not list(probes.glob("*.wav"))
    rejected = run("train_music3.py", ["--dummy", "--resume", str(full / "adapter-export.pt"), "--save_dir", str(tmp_path / "bad-resume")], check=False)
    assert rejected.returncode != 0


def test_real_shared_objective_neutral_condition_and_transactional_reload(monkeypatch):
    import torch
    shared = bootstrap()
    adapter, model, fixed, game = shared.fixtures.make_audio("music3", checkpointing=True, critic_class=shared.audio.AudioResponseCritic)
    assert type(game) is shared.core.ConditionalPG09
    assert adapter.particles is adapter.slider.particles
    assert sum(p is adapter.particles for group in game.opt_g.param_groups for p in group["params"]) == 1
    seen = []
    original = adapter._hidden
    def traced(ids):
        seen.append(tuple(ids))
        return original(ids)
    adapter._hidden = traced
    def forbidden(*args, **kwargs):
        raise AssertionError("historical MSE/bridge training reached")
    monkeypatch.setattr(torch.nn.functional, "mse_loss", forbidden)
    monkeypatch.setattr(shared.bridge, "update", forbidden)
    game.step()
    assert set(seen) == {p.row.neutral_ids for p in fixed}
    state = game.state_dict()
    bad = deepcopy(state)
    bad["config"]["backend"]["model_identity"]["seed"] = 999
    with pytest.raises(ValueError):
        game.load_state_dict(bad)
    equal(state, game.state_dict())
    conditions = shared.audio.audio_conditions(fixed)
    assert torch.equal(conditions[0, :32], fixed[0].neutral[0])
    assert conditions[0, -2:].tolist() == [2, 1]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("raw", ["-1,0,1", "0,1.1", "nan", "inf", "-inf", "", "0,-0.1"])
def test_unqualified_or_invalid_strengths_rejected(raw):
    with pytest.raises(ValueError, match="finite scales"):
        _scales(raw)


def test_infer_requires_export_and_does_not_accept_training_or_legacy_weights(tmp_path):
    with pytest.raises(ValueError, match="weights"):
        infer(parse_infer(["--dummy"]))
    import torch
    weights = tmp_path / "wrong.pt"
    torch.save({"schema": "historical"}, weights)
    with pytest.raises(ValueError, match="shared inference export"):
        infer(parse_infer(["--dummy", "--weights", str(weights), "--out_dir", str(tmp_path / "bad")]))


def test_historical_flags_and_silent_dummy_truncation_rejected(tmp_path):
    for args in (["--dummy", "--g_lr", "0.01"], ["--dummy", "--loss", "nmse"], ["--dummy", "--steps", "8"], ["--dummy", "--prompts_file", "custom.yaml"]):
        with pytest.raises(SystemExit):
            train(parse_args(args))
    with pytest.raises(RuntimeError, match="native Music3 acoustic"):
        infer(parse_infer([]))


def test_live_plan_never_dispatches_and_release_rejects_native_width(tmp_path, monkeypatch):
    shared = bootstrap()
    config = tmp_path / "native/language_model"
    config.mkdir(parents=True)
    (config / "config.json").write_text('{"hidden_size":4096}')
    files = [tmp_path / name for name in ("manifest.json", "train.yaml", "heldout.yaml")]
    for path in files:
        path.write_text('{}')
    def forbidden():
        raise AssertionError("native worker dispatched")
    monkeypatch.setattr(shared.worker, "main", forbidden)
    with pytest.raises(ValueError, match="released PG09"):
        train(parse_args(["--live", "--save_dir", "/mnt/nvme/sliders/qualification/runs/music3-product-audit-unadmitted", "--model_dir", str(config.parent), "--manifest", str(files[0]), "--prompts_file", str(files[1]), "--heldout_prompts", str(files[2])]))


def test_native_plan_preserves_pinned_import_roots_in_fresh_process(tmp_path, monkeypatch):
    import os
    shared = bootstrap()
    config = tmp_path / "native/language_model"
    config.mkdir(parents=True)
    (config / "config.json").write_text('{"hidden_size":32}')
    files = [tmp_path / name for name in ("manifest.json", "train.yaml", "heldout.yaml")]
    for path in files:
        path.write_text('{}')
    def forbidden():
        raise AssertionError("native worker dispatched")
    monkeypatch.setattr(shared.worker, "main", forbidden)
    plan = train(parse_args(["--live", "--save_dir", "/mnt/nvme/sliders/qualification/runs/music3-product-audit-plan-test",
                             "--model_dir", str(config.parent), "--manifest", str(files[0]),
                             "--prompts_file", str(files[1]), "--heldout_prompts", str(files[2])]))
    assert not plan["executed"] and plan["status"] == "PROPOSED_NOT_ADMITTED"
    env = dict(os.environ, **plan["environment"])
    env["CUDA_VISIBLE_DEVICES"] = ""  # Read-only import check, never a dispatch.
    result = subprocess.run([plan["command"][0], plan["command"][1], "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    code = "import particle_sliders.pg09 as p; import conceptmod.textsliders.pg09_audio as a; import torch; print(p.__file__); print(a.__file__); assert not torch.cuda.is_initialized()"
    result = subprocess.run([plan["command"][0], "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert str(shared.core_checkout) in result.stdout and str(shared.shared_checkout) in result.stdout
    assert not plan["native_runtime"]["cuda_initialized"]


def test_native_plan_cannot_inject_a_whole_site_packages_overlay(tmp_path, monkeypatch):
    import particlegan
    shared = bootstrap()
    config = tmp_path / "native/language_model"
    config.mkdir(parents=True)
    (config / "config.json").write_text('{"hidden_size":4096}')
    files = [tmp_path / name for name in ("manifest.json", "train.yaml", "heldout.yaml")]
    for path in files:
        path.write_text('{}')
    # Source/profile validation has separate real-runtime coverage. Exercise the
    # plan's additional path guard without changing any installed source.
    monkeypatch.setattr(shared.audio, "require_audio_width", lambda *a, **kw: None)
    monkeypatch.setattr(particlegan, "__file__", "/tmp/fake-site-packages/particlegan/__init__.py")
    with pytest.raises(ValueError, match="no site-packages injection"):
        train(parse_args(["--live", "--pg_source_profile", "alz-continuous-v1", "--penalty_input_domain", "continuous",
                          "--save_dir", "/mnt/nvme/sliders/qualification/runs/music3-product-audit-overlay-test",
                          "--model_dir", str(config.parent), "--manifest", str(files[0]),
                          "--prompts_file", str(files[1]), "--heldout_prompts", str(files[2])]))
