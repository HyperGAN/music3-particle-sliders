"""Source-free product qualification against the exact installed PG0.9.1 API."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from music3.infer import infer, parse_args as parse_infer, _scales
from music3.train import parse_args, train
from music3.shared import bootstrap

ROOT = Path(__file__).resolve().parents[1]
RUNNER = textwrap.dedent('''
    import importlib.abc,runpy,sys
    class NoHistoricalSource(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'conceptmod','analysis','pg09_audio_cpu_fixtures','qualify_audio_pg09_live','inspect_audio_pg09_assets','run_audio_pg09_smoke'}:
                raise AssertionError('historical source imported: '+fullname)
    sys.meta_path.insert(0,NoHistoricalSource())
    script=sys.argv.pop(1)
    sys.argv[0]=script
    runpy.run_path(script,run_name='__main__')
''')


def run(script, args, cwd, check=True, extra_env=None):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for key in ("PYTHONPATH", "PYTHONHOME", "PARTICLE_SLIDERS_SHARED_CHECKOUT", "PARTICLE_SLIDERS_CORE_CHECKOUT"):
        environment.pop(key, None)
    if extra_env:
        environment.update(extra_env)
    result = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(ROOT / "scripts" / script), *args],
                            cwd=cwd, env=environment, text=True, capture_output=True, timeout=120)
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
def test_all_entrypoint_help_without_model_or_source_import(script, tmp_path):
    result = run(script, ["--help"], tmp_path)
    assert "MiniMaxAI/MiniMax-Music3" in result.stdout and "--dummy" in result.stdout


@pytest.mark.parametrize("script", ["train_lora_music3.py", "train_encoder_music3.py"])
def test_unqualified_hosts_fail_before_output_or_load(script, tmp_path):
    out = tmp_path / "unsupported"
    result = run(script, ["--dummy", "--save_dir", str(out)], tmp_path, check=False)
    assert result.returncode != 0 and "only the LM/particle binding" in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("preset", ["bispf", "bispf_optimized"])
def test_installed_both_presets_fresh_process_resume_and_actual_export_probe(preset, tmp_path):
    import torch
    full, first, resumed = [tmp_path / name for name in ("full", "first", "resumed")]
    for script, out, until, extra in (("train_music3.py", full, 3, []),
                                      ("train_lm_slider_music3.py", first, 1, []),
                                      ("train_lora_music3_particle.py", resumed, 3, ["--resume", str(first / "checkpoint.pt")])):
        run(script, ["--dummy", "--preset", preset, "--steps", str(until), "--save_dir", str(out), *extra], tmp_path)
    a = torch.load(full / "qualification.pt", weights_only=True)
    b = torch.load(resumed / "qualification.pt", weights_only=True)
    equal(a, b)
    packet = a["game"]
    assert packet["config"]["pg_version"] == "0.9.1"
    assert packet["config"]["pg_release_wheel_sha256"] == "b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec"
    sources = packet["config"]["implementation_sources"]
    assert "particle_sliders.pg09_audio" in sources
    assert not any(name.startswith("conceptmod.") for name in sources)
    for metric in a["metrics"]:
        assert metric["g_loss"] == pytest.approx(metric["g_adv"] + metric["prior_regularization"])
        assert metric["d_loss"] == pytest.approx(metric["d_adv"] + metric["d_pen"])
        assert metric["d_pen"] > 0
    receipt = json.loads((full / "product-integration.json").read_text())
    assert receipt["lineage"] == "music3-packaged-pg091-v1"
    assert receipt["source_receipt"]["particlegan_version"] == "0.9.1"
    assert not receipt["old_checkpoint_migration"] and not receipt["native_audio_qualified"]
    out = tmp_path / "probes"
    run("infer_music3.py", ["--dummy", "--weights", str(full / "adapter-export.pt"), "--out_dir", str(out)], tmp_path)
    probes = torch.load(out / "hidden-probes.pt", weights_only=True)
    assert probes["off_exact"]
    assert not probes["audio_rendered"] and not probes["quality_qualified"]
    equal(probes["samples"]["1.0"], a["probes"])
    assert not list(out.glob("*.wav"))
    result = run("train_music3.py", ["--dummy", "--resume", str(full / "adapter-export.pt"), "--save_dir", str(tmp_path / "bad-resume")], tmp_path, check=False)
    assert result.returncode != 0


def test_public_game_neutral_condition_bank_and_gan_only(monkeypatch):
    import torch
    shared = bootstrap()
    runtime = shared.cli.make_audio_game("music3", checkpointing=True)
    adapter, model, fixed, game = runtime.adapter, runtime.model, runtime.rows, runtime.game
    assert type(game) is shared.core.ConditionalPG09
    assert model.__class__.__name__ == "Qwen3ForCausalLM"
    assert adapter.particles is adapter.slider.particles
    assert sum(p is adapter.particles for group in game.opt_g.param_groups for p in group["params"]) == 1
    seen = []
    original = adapter._hidden
    def traced(ids):
        seen.append(tuple(ids))
        return original(ids)
    adapter._hidden = traced
    def forbidden(*args, **kwargs):
        raise AssertionError("MSE entered shared GAN-only objective")
    monkeypatch.setattr(torch.nn.functional, "mse_loss", forbidden)
    owned = {id(p) for p in adapter.network.parameters()}
    frozen = {name:p.detach().clone() for name,p in model.named_parameters() if id(p) not in owned}
    bank = adapter.particles.detach().clone()
    for _ in range(3):
        game.step()
    assert set(seen) == {p.row.neutral_ids for p in fixed}
    assert not torch.equal(bank, adapter.particles)
    assert all(torch.equal(frozen[name], p) for name,p in model.named_parameters() if name in frozen)
    conditions = shared.audio.audio_conditions(fixed)
    assert torch.equal(conditions[0,:fixed[0].neutral.shape[-1]],fixed[0].neutral[0])
    assert conditions[0,-1].item()==1
    value=adapter.generated(fixed[0])
    gradient=torch.autograd.grad((value*torch.linspace(-1,1,value.shape[-1])).sum(),adapter.particles)[0]
    assert gradient.abs().sum()>0
    assert torch.equal(adapter.sample(fixed[0],scale=0),fixed[0].neutral)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change", ["version", "model", "source"])
def test_wrong_source_lineage_rejected_without_state_or_rng_mutation(change):
    import torch
    runtime=bootstrap().cli.make_audio_game("music3")
    runtime.game.step()
    before=runtime.game.state_dict()
    bad=deepcopy(before)
    if change=="version":
        bad["config"]["pg_version"]="0.9.0"
    elif change=="model":
        bad["config"]["backend"]["model_identity"]["new_unqualified_identity"]=True
    else:
        bad["config"]["implementation_sources"]["conceptmod.textsliders.pg09_audio"]="old-source"
    rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        runtime.game.load_state_dict(bad)
    equal(before,runtime.game.state_dict())
    assert torch.equal(rng,torch.get_rng_state())


@pytest.mark.parametrize("raw", ["-1,0,1","0,1.1","nan","inf","-inf","","0,-0.1"])
def test_unqualified_or_invalid_strengths_rejected(raw):
    with pytest.raises(ValueError,match="finite scales"):
        _scales(raw)


def test_missing_export_and_historical_options_rejected(tmp_path):
    with pytest.raises(ValueError,match="weights"):
        infer(parse_infer(["--dummy"]))
    for args in (["--dummy","--g_lr",".01"],["--dummy","--loss","nmse"],["--dummy","--steps","8"],
                 ["--dummy","--shared_checkout","/tmp/oldsource"],["--dummy","--core_checkout","/tmp/oldcore"]):
        with pytest.raises(SystemExit):
            train(parse_args(args))
    with pytest.raises(RuntimeError,match="native Music3 acoustic"):
        infer(parse_infer([]))


def test_native_plan_routes_public_api_without_loading_or_dispatch(tmp_path, monkeypatch):
    import hashlib
    import torch
    shared = bootstrap()
    root = tmp_path / "native"
    (root / "language_model").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    (root / "language_model/config.json").write_text('{"hidden_size":32}')
    (root / "tokenizer/fixture.json").write_text('{"fixture":true}')
    entries = []
    for path in sorted(root.rglob("*.json")):
        entries.append(dict(filename=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(files=entries, fixture_only=True)))
    files = []
    for name, count in (("train",4),("heldout",2)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(dict(rows=[dict(neutral=f"piano room {i}",positive=f"loud piano room {i}",lyrics="morning light") for i in range(count)])))
        files.append(path)
    def forbidden(*args, **kwargs):
        raise AssertionError("native plan constructed a model or executed training")
    monkeypatch.setattr(shared.cli, "make_audio_game", forbidden)
    monkeypatch.setattr(shared.native, "Music3Backend", forbidden)
    out = tmp_path / "planned-output"
    argv = ["--live", "--model_dir", str(root), "--manifest", str(manifest),
            "--prompts_file", str(files[0]), "--heldout_prompts", str(files[1]), "--save_dir", str(out),
            "--preset", "bispf", "--seed", "19"]
    plan = train(parse_args(argv))
    assert plan["status"] == "PLAN_ONLY_NOT_ADMITTED" and not plan["dispatch_allowed"]
    assert plan["output_absent"] and not out.exists()
    assert not plan["native_audio_rendered"] and not torch.cuda.is_initialized()
    (root / "tokenizer/fixture.json").write_text("changed")
    with pytest.raises(ValueError, match="hash differs"):
        train(parse_args(argv))
    assert not out.exists()


def test_native_resume_cannot_bypass_new_cohort_producer_gate(tmp_path):
    with pytest.raises(SystemExit):
        train(parse_args(["--live", "--resume", str(tmp_path / "old-checkpoint.pt")]))


def test_wrong_audio_family_rejected_before_factory_or_output(tmp_path, monkeypatch):
    import torch
    shared = bootstrap()
    path = tmp_path / "wrong-family.pt"
    torch.save(dict(schema=shared.cli.EXPORT_SCHEMA, factory=dict(family="yue2"), export={}), path)
    def forbidden(*args, **kwargs):
        raise AssertionError("wrong-family export allocated a model")
    monkeypatch.setattr(shared.cli, "make_audio_game", forbidden)
    out = tmp_path / "wrong-family-output"
    with pytest.raises(ValueError, match="audio family differs"):
        infer(parse_infer(["--dummy", "--weights", str(path), "--out_dir", str(out)]))
    assert not out.exists() and not torch.cuda.is_initialized()


@pytest.mark.parametrize("corruption", ["nonfinite", "decomposition", "clock"])
def test_prior_metric_history_rejected_before_live_restore_or_update(tmp_path, corruption):
    import torch
    shared = bootstrap()
    source = shared.cli.make_audio_game("music3")
    packet = shared.cli.run_training(source, until=1, out=tmp_path / "source")
    broken = deepcopy(packet)
    if corruption == "nonfinite":
        broken["metrics"][0]["g_loss"] = float("nan")
    elif corruption == "decomposition":
        broken["metrics"][0]["g_loss"] += 1
    else:
        broken["metrics"][0]["step"] = 2
    path = tmp_path / "broken-checkpoint.pt"
    torch.save(broken, path)
    receiver = shared.cli.make_audio_game("music3")
    before = deepcopy(receiver.game.state_dict())
    rng = torch.get_rng_state().clone()
    out = tmp_path / "rejected"
    with pytest.raises((ValueError, TypeError)):
        shared.cli.run_training(receiver, until=3, out=out, resume=path)
    equal(before, receiver.game.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert not out.exists() and not torch.cuda.is_initialized()
