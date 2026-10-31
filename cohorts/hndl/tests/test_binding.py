"""Real installed CPU HNDL graphs and genuine random Qwen3 hosts; no model weights."""
from copy import deepcopy

import pytest
import torch

from music3_hndl import build_cpu_game
from music3_hndl.binding import Music3HNDLCritic
from particle_sliders.pg09 import _hash
from particle_sliders.pg09_audio import AudioResponseCritic


def copy_affine(destination, source):
    with torch.no_grad():
        destination.weight.copy_(source.weight)
        if source.bias is not None:
            destination.bias.copy_(source.bias)


def test_real_hndl_critic_matches_ar_readout_outputs_and_penalty_derivatives():
    old, new = AudioResponseCritic(32, 36), Music3HNDLCritic(32, 36)
    copy_affine(new.graph['hidden'], old.net[0])
    copy_affine(new.graph['logits'], old.net[2])
    value = torch.randn(2, 32, requires_grad=True)
    other = value.detach().clone().requires_grad_()
    condition = torch.randn(2, 36)
    expected, actual = old(value, condition), new(other, condition)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
    g0 = torch.autograd.grad(expected.sum(), value, create_graph=True)[0]
    g1 = torch.autograd.grad(actual.sum(), other, create_graph=True)[0]
    torch.testing.assert_close(g1, g0, atol=2e-7, rtol=2e-5)
    d0 = torch.autograd.grad(g0.square().sum(), old.net[0].weight)[0]
    d1 = torch.autograd.grad(g1.square().sum(), new.graph['hidden'].weight)[0]
    torch.testing.assert_close(d1, d0, atol=2e-7, rtol=3e-5)
    assert new.identity['profile'] == 'hypergan-hndl-v2'
    assert new.identity['hypergan_commit'] == '5f368a3770cf5bdae0448995c9f76e29af8049f7'


def test_matched_weight_hndl_qwen3_matches_routed_native_outputs_and_bank_vjps():
    from transformers import Qwen3ForCausalLM
    from particle_sliders.music3_native import ParticleNetwork
    from particle_sliders.pg09_audio import Music3ARResponseAdapter
    with build_cpu_game(checkpointing=False) as run:
        model = Qwen3ForCausalLM(run.model.config).eval().requires_grad_(False)
        model.load_state_dict(run.model.state_dict())
        old = ParticleNetwork(model)
        adapter = Music3ARResponseAdapter(model, old, model_identity={'fixture': 'matched-weight comparison'})
        with torch.no_grad():
            run.slider.particles.copy_(old.particles)
        for lora in old.network.unet_loras:
            projection = run.slider.projections[lora.lora_name]
            torch.nn.init.normal_(lora.lora_up.weight, std=.2)
            copy_affine(projection.lora_down, lora.lora_down)
            copy_affine(projection.lora_up, lora.lora_up)
            bridge = old.bridges[lora.lora_name.replace('.', '-')]
            for family, sequential in [('router', bridge.router), ('bridge', bridge.net)]:
                for index in (0, 2, 4, 6):
                    copy_affine(projection.graph[family + str(index)], sequential[index])
        ids = (11, 13, 17)
        expected = adapter._response(ids, scale=.7)
        actual = run.adapter._response(ids, scale=.7)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        grad0, = torch.autograd.grad(expected.square().sum(), old.particles)
        grad1, = torch.autograd.grad(actual.square().sum(), run.slider.particles)
        torch.testing.assert_close(grad1, grad0, atol=3e-6, rtol=3e-5)
        assert grad1.abs().sum() > 0
        assert torch.equal(adapter._response(ids, scale=0.), run.adapter._response(ids, scale=0.))
        assert sum(p is run.slider.particles for p in run.adapter.network.parameters()) == 1
        assert not any(p.requires_grad for p in run.model.parameters())


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_shared_training_resume_export_and_frozen_ownership(preset):
    rng = torch.get_rng_state().clone()
    with build_cpu_game(preset=preset) as full:
        assert torch.equal(torch.get_rng_state(), rng)
        base = _hash(full.model.state_dict())
        particle_before = full.slider.particles.detach().clone()
        full.game.step()
        first = deepcopy(full.game.state_dict())
        full.game.step()
        final = deepcopy(full.game.state_dict())
        assert not torch.equal(full.slider.particles, particle_before)
        assert _hash(full.model.state_dict()) == base
        assert all(not p.requires_grad and p.grad is None for p in full.model.parameters())
        export = deepcopy(full.game.export_state_dict())
        expected = full.adapter._response((11, 13, 17), scale=.5).detach().clone()
    with build_cpu_game(preset=preset) as resumed:
        resumed.game.load_state_dict(first)
        resumed.game.step()
        assert _hash(resumed.game.state_dict()) == _hash(final)
        before = _hash(resumed.game.state_dict())
        broken = deepcopy(first)
        broken['config']['pg_version'] = '0.9.1'
        with pytest.raises((RuntimeError, ValueError)):
            resumed.game.load_state_dict(broken)
        assert _hash(resumed.game.state_dict()) == before
    with build_cpu_game(preset=preset) as receiver:
        receiver.game.load_export_state_dict(export)
        assert torch.equal(receiver.adapter._response((11, 13, 17), scale=.5), expected)
        before = _hash(receiver.adapter.network.state_dict())
        with pytest.raises(RuntimeError, match='inference export'):
            receiver.game.step()
        assert _hash(receiver.adapter.network.state_dict()) == before


def test_forward_owner_conflict_can_be_resolved_and_cleanup_retried():
    run = build_cpu_game()
    host = run.model.model.layers[0].self_attn.q_proj
    ours = host.forward
    external = lambda *args: None
    host.forward = external
    with pytest.raises(RuntimeError, match='owner changed'):
        run.close()
    assert host.forward is external
    host.forward = ours
    run.close()
    run.close()
    assert 'forward' not in vars(host)
    assert '_pg09_scale_checkpoint' not in vars(run.model.model.layers[0])


def test_partial_hndl_attachment_failure_restores_native_model(monkeypatch):
    from music3_hndl import binding
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, vocab_size=256, max_position_embeddings=256))
    state = _hash(model.state_dict())
    modes = [m.training for m in model.modules()]
    gradients = [p.requires_grad for p in model.parameters()]
    builder, identity = binding.verified_builder()
    calls = []
    def fail_third(*args, **kwargs):
        calls.append(True)
        if len(calls) == 3:
            raise RuntimeError('injected HNDL construction failure')
        return builder(*args, **kwargs)
    monkeypatch.setattr(binding, 'verified_builder', lambda: (fail_third, identity))
    with pytest.raises(RuntimeError, match='injected HNDL'):
        binding.Music3HNDLNetwork(model)
    assert state == _hash(model.state_dict())
    assert modes == [m.training for m in model.modules()]
    assert gradients == [p.requires_grad for p in model.parameters()]
    assert all('forward' not in vars(m) for m in model.modules() if isinstance(m, torch.nn.Linear))


def test_candidate_canonical_source_lock_cannot_be_rewritten(tmp_path, monkeypatch):
    from music3_hndl import source
    from pathlib import Path
    lock = Path(source.__file__).with_name('core.lock.json').read_bytes()
    (tmp_path / 'core.lock.json').write_bytes(lock + b'\n')
    monkeypatch.setattr(source, 'Path', lambda _: tmp_path / 'source.py')
    with pytest.raises(RuntimeError, match='canonical source lock bytes'):
        source.verify_core()


def test_initial_and_restored_zero_scale_equal_unmodified_base_outside_context():
    from music3_hndl.binding import Music3HNDLNetwork
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, vocab_size=256, max_position_embeddings=256)).eval().requires_grad_(False)
    ids = torch.tensor([[11, 13, 17]])
    with torch.no_grad():
        expected = model.model(input_ids=ids, use_cache=False).last_hidden_state.clone()
    slider = Music3HNDLNetwork(model)
    try:
        for projection in slider.projections.values():
            torch.nn.init.normal_(projection.lora_up.weight, std=.2)
        assert slider.scale == 0. and all(p.multiplier == 0. for p in slider.projections.values())
        with torch.no_grad():
            assert torch.equal(model.model(input_ids=ids, use_cache=False).last_hidden_state, expected)
            with slider.scaled(1.):
                assert not torch.equal(model.model(input_ids=ids, use_cache=False).last_hidden_state, expected)
            assert torch.equal(model.model(input_ids=ids, use_cache=False).last_hidden_state, expected)
    finally:
        slider.close()


@pytest.mark.parametrize('after_mutation', [False, True])
def test_partial_forward_assignment_failure_rolls_back_all_patches(after_mutation):
    from music3_hndl.binding import Music3HNDLNetwork
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, vocab_size=256, max_position_embeddings=256))
    primary = RuntimeError('injected forward setter failure')
    class Linear(torch.nn.Linear):
        def __setattr__(self, name, value):
            if name == 'forward' and getattr(self, 'fail_next_forward', False):
                object.__setattr__(self, 'fail_next_forward', False)
                if after_mutation:
                    super().__setattr__(name, value)
                raise primary
            super().__setattr__(name, value)
    original = model.model.layers[0].self_attn.k_proj
    replacement = Linear(original.in_features, original.out_features, bias=original.bias is not None)
    replacement.load_state_dict(original.state_dict())
    replacement.fail_next_forward = True
    model.model.layers[0].self_attn.k_proj = replacement
    modes = [m.training for m in model.modules()]
    flags = [p.requires_grad for p in model.parameters()]
    state = _hash(model.state_dict())
    with pytest.raises(RuntimeError) as caught:
        Music3HNDLNetwork(model)
    assert caught.value is primary
    assert all('forward' not in vars(m) for m in model.modules() if isinstance(m, torch.nn.Linear))
    assert modes == [m.training for m in model.modules()]
    assert flags == [p.requires_grad for p in model.parameters()]
    assert state == _hash(model.state_dict())


def test_checkpoint_close_failure_still_closes_slider_and_retains_retry(monkeypatch):
    run = build_cpu_game()
    close = run.adapter.close_checkpointing
    primary = RuntimeError('injected checkpoint close failure')
    def fail():
        raise primary
    monkeypatch.setattr(run.adapter, 'close_checkpointing', fail)
    with pytest.raises(RuntimeError) as caught:
        run.close()
    assert caught.value is primary
    assert all('forward' not in vars(p.org_forward.__self__) for p in run.slider.projections.values())
    assert run.adapter._checkpointing  # Failed shared handles remain reachable.
    monkeypatch.setattr(run.adapter, 'close_checkpointing', close)
    primary.retry_cleanup()
    assert not run.adapter._checkpointing


def test_context_body_error_is_preserved_when_both_cleanup_owners_fail():
    from types import SimpleNamespace
    from music3_hndl.binding import OwnedGame
    primary = RuntimeError('primary context error')
    checkpoint_error, slider_error = RuntimeError('checkpoint owner'), RuntimeError('slider owner')
    attempts = []
    def checkpoint_close():
        attempts.append('checkpoint')
        raise checkpoint_error
    def slider_close():
        attempts.append('slider')
        raise slider_error
    adapter = SimpleNamespace(close_checkpointing=checkpoint_close)
    slider = SimpleNamespace(close=slider_close)
    with pytest.raises(RuntimeError) as caught:
        with OwnedGame(None, slider, adapter, None):
            raise primary
    assert caught.value is primary
    assert attempts == ['checkpoint', 'slider']
    assert primary.cleanup_errors == (checkpoint_error, slider_error)
    adapter.close_checkpointing = lambda: attempts.append('checkpoint retry')
    slider.close = lambda: attempts.append('slider retry')
    primary.retry_cleanup()
    assert attempts[-2:] == ['checkpoint retry', 'slider retry']


def test_multiple_native_forward_conflicts_keep_all_errors_and_retry_ledgers():
    run = build_cpu_game()
    hosts = [run.model.model.layers[0].self_attn.q_proj, run.model.model.layers[0].self_attn.k_proj]
    forwards = [host.forward for host in hosts]
    for host in hosts:
        host.forward = lambda *args: None
    with pytest.raises(RuntimeError, match='owner changed') as caught:
        run.close()
    assert len(caught.value.cleanup_errors) == 2
    assert len(run.slider._forwards) == 2
    for host, ours in zip(hosts, forwards):
        host.forward = ours
    caught.value.retry_cleanup()
    assert not run.slider._forwards
    assert all('forward' not in vars(host) for host in hosts)


def test_factory_primary_error_survives_checkpoint_cleanup_failure(monkeypatch):
    from music3_hndl import binding
    primary = RuntimeError('primary critic construction failure')
    cleanup_error = RuntimeError('injected checkpoint cleanup failure')
    observed = []
    adapter_class = binding.Music3ARResponseAdapter
    def make_adapter(*args, **kwargs):
        adapter = adapter_class(*args, **kwargs)
        observed.append(adapter)
        close = adapter.close_checkpointing
        attempts = []
        def fail_once():
            attempts.append(True)
            if len(attempts) == 1:
                raise cleanup_error
            close()
        adapter.close_checkpointing = fail_once
        return adapter
    def fail_critic(*args):
        raise primary
    monkeypatch.setattr(binding, 'Music3ARResponseAdapter', make_adapter)
    monkeypatch.setattr(binding, 'Music3HNDLCritic', fail_critic)
    with pytest.raises(RuntimeError) as caught:
        binding.build_cpu_game()
    assert caught.value is primary and primary.cleanup_errors == (cleanup_error,)
    adapter = observed[0]
    assert adapter._checkpointing
    assert all('forward' not in vars(p.org_forward.__self__) for p in adapter.slider.projections.values())
    primary.retry_cleanup()
    assert not adapter._checkpointing


@pytest.mark.parametrize('dependency', ['hndl', 'hypergan'])
def test_shared_verified_builder_rejects_changed_dependency_sources_before_model_allocation(monkeypatch, dependency):
    import hndl
    import hypergan.hndl_networks
    import transformers
    from pathlib import Path
    from particle_sliders import fibo_hndl
    target = (Path(hndl.__file__) if dependency == 'hndl'
              else Path(hypergan.hndl_networks.__file__))
    original_sha = fibo_hndl._sha
    monkeypatch.setattr(fibo_hndl, '_sha',
        lambda path: '0' * 64 if Path(path) == target else original_sha(path))
    def forbidden(*args, **kwargs):
        raise AssertionError('unverified dependency allocated a model')
    monkeypatch.setattr(transformers, 'Qwen3ForCausalLM', forbidden)
    with pytest.raises(ValueError, match='source closure|builder closure'):
        build_cpu_game()


@pytest.mark.parametrize('options', [
    {'width': 4096}, {'width': 170}, {'width': 64}, {'steps': 5},
    {'steps': True}, {'seed': -1}, {'checkpointing': 1}, {'preset': 'other'},
])
def test_native_width_and_invalid_options_are_rejected_before_model_allocation(monkeypatch, options):
    import transformers
    def forbidden(*args, **kwargs):
        raise AssertionError('refused request allocated a model')
    monkeypatch.setattr(transformers, 'Qwen3ForCausalLM', forbidden)
    with pytest.raises((RuntimeError, ValueError)):
        build_cpu_game(**options)
    assert not torch.cuda.is_initialized()
