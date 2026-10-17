"""Transactional public Music3 loads and RNG-independent export probes on CPU."""
from copy import deepcopy

import pytest
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn

from music3.shared import bootstrap


@pytest.fixture(scope="module")
def shared_api():
    return bootstrap()


@pytest.fixture
def audio_run(shared_api):
    return shared_api.cli.make_audio_game("music3")


def _binding(forward):
    """Bound-method identities are stable even when attribute reads recreate them."""
    return id(getattr(forward, "__self__", None)), id(getattr(forward, "__func__", forward))


def _inventory(model):
    return (
        tuple((name, id(module), type(module)) for name, module in model.named_modules()),
        tuple((name, id(module), _binding(module.forward))
              for name, module in model.named_modules() if isinstance(module, nn.Linear)),
        tuple((name, tuple(value.shape), value.dtype, value.device)
              for name, value in model.state_dict().items()),
    )


@pytest.mark.parametrize("malformation", ["nan", "inf", "missing", "extra", "shape", "dtype"])
def test_matching_metadata_malformed_native_load_is_transactional(shared_api, audio_run, tmp_path, malformation):
    run = audio_run
    assert type(run.model).__name__ == "Qwen3ForCausalLM"
    assert type(run.model).__module__.startswith("transformers.models.qwen3.")
    assert run.model.config.hidden_size == 32
    assert type(run.adapter.slider) is shared_api.native.ParticleNetwork

    good = tmp_path / "good.safetensors"
    run.adapter.slider.save(good, {"concept": "CPU qualification"})
    with safe_open(str(good), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    assert metadata and "conceptmod" in metadata
    state = load_file(str(good), device="cpu")
    key = next(name for name, value in state.items() if value.is_floating_point() and value.numel() > 1)
    if malformation in ("nan", "inf"):
        state[key].view(-1)[0] = float(malformation)
    elif malformation == "missing":
        del state[key]
    elif malformation == "extra":
        state["unexpected.weight"] = torch.zeros(1)
    elif malformation == "shape":
        state[key] = state[key].reshape(-1)[:-1].clone()
    else:
        state[key] = state[key].to(torch.float64)
    malformed = tmp_path / f"{malformation}.safetensors"
    save_file(state, str(malformed), metadata=metadata)

    # Never deepcopy a host already modified by routed forward closures.
    target = type(run.model)(run.model.config).eval().requires_grad_(False)
    inventory = _inventory(target)
    weights = {name: value.detach().clone() for name, value in target.state_dict().items()}
    rng = torch.get_rng_state().clone()
    with pytest.raises((ValueError, RuntimeError)):
        shared_api.native.ParticleNetwork.load(target, malformed)
    assert _inventory(target) == inventory
    actual = target.state_dict()
    assert actual.keys() == weights.keys()
    assert all(torch.equal(weights[name], value) for name, value in actual.items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.cuda.is_initialized()


def test_public_probe_rebuilds_genuine_fixture_without_rng_changes_and_export_cannot_train(shared_api, audio_run, tmp_path):
    run = audio_run
    assert type(run.model).__name__ == "Qwen3ForCausalLM"
    assert type(run.model).__module__.startswith("transformers.models.qwen3.")
    assert run.model.config.hidden_size == 32 and run.model.config.vocab_size > 151669
    assert not run.model.training
    for prepared in tuple(run.rows) + tuple(run.heldout):
        assert prepared.row.neutral_ids[-1] == prepared.row.positive_ids[-1] == 151669
        assert prepared.row.scale == 1.0
        assert prepared.neutral.shape == prepared.target.shape == (1, 32)

    out = tmp_path / "trained"
    packet = shared_api.cli.run_training(run, until=1, out=out)
    config = packet["game"]["config"]
    assert config["pg_version"] == "0.9.1"
    assert config["pg_release_wheel_sha256"] == "b01b7957169cb3de45887e22b4d525fd54071577e1de58089707cf05fe6992ec"
    assert "particle_sliders.pg09_audio" in config["implementation_sources"]
    assert not any(name.startswith("conceptmod.") for name in config["implementation_sources"])

    rng = torch.get_rng_state().clone()
    probes = shared_api.cli.probe_export(out / "adapter-export.pt")
    assert torch.equal(torch.get_rng_state(), rng)
    assert set(probes["samples"]) == {"0.0", "0.5", "1.0"}
    assert probes["off_exact"]
    assert not probes["audio_rendered"] and not probes["quality_qualified"]
    assert len(probes["samples"]["1.0"]) == len(packet["probes"])
    assert all(torch.equal(actual, expected)
               for actual, expected in zip(probes["samples"]["1.0"], packet["probes"]))
    assert all(torch.equal(actual, prepared.neutral)
               for actual, prepared in zip(probes["samples"]["0.0"], run.heldout))

    receiver = shared_api.cli.make_audio_game("music3")
    receiver.game.load_export_state_dict(packet["export"])
    network = deepcopy(receiver.adapter.network.state_dict())
    completed = receiver.game.completed_steps
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="inference export"):
        receiver.game.step()
    assert receiver.game.completed_steps == completed
    assert all(torch.equal(network[name], value)
               for name, value in receiver.adapter.network.state_dict().items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.cuda.is_initialized()


def test_cpu_factory_preserves_rng_threads_and_never_dispatches_cuda_seed(shared_api, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fixture dispatched an all-device CUDA seed")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        rng = torch.get_rng_state().clone()
        monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
        run = shared_api.cli.make_audio_game("music3")
        assert type(run.model).__name__ == "Qwen3ForCausalLM"
        assert torch.equal(rng, torch.get_rng_state())
        assert torch.get_num_threads() == 2
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(previous_threads)
