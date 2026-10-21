"""New-cohort installed-product conformance, authored pending a resource slot.

Run only this module in the exact isolated response environment. Historical
product tests belong to the old core0.6.0 environment. Tiny random Qwen3 and
two-linear ownership fixtures exercise wiring, never pretrained audio quality.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.metadata import entry_points
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from music3 import lifecycle as product
from music3.response_source import CORE_COMMIT, LOCK, LOCK_SHA256
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import CheckpointError, SessionConfig, UnsupportedOperation
from particle_sliders.pg09_audio import ARLastHiddenResponseAdapter
from particle_sliders.pg09_backends import checkpoint_with_adapter_scale
from particle_sliders.response import open_response


def equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            equal(left, right)
    else:
        assert actual == expected


def config(**overrides):
    options = dict(dummy=True, width=32, steps=2, batch=2, seed=7,
                   checkpointing=True, ema_decay=.9)
    options.update(overrides)
    return SessionConfig('music3', 'music3-ownership-new-test-cohort', **options)


def ownership(module):
    return {name: vars(module)[name] for name in ('forward', '_pg09_scale_checkpoint') if name in vars(module)}


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.7))
        self.scale = 1.

    @contextmanager
    def scaled(self, scale):
        old = self.scale
        self.scale = scale
        try:
            yield
        finally:
            self.scale = old


def cleanup_fixture():
    model = nn.Module()
    model.config = SimpleNamespace(hidden_size=32)
    model.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
    model.eval().requires_grad_(False)
    adapter = ARLastHiddenResponseAdapter(model, Scale(),
        model_identity={'fixture': 'music3-product-cleanup'}, checkpoint_modules=model.blocks)
    return model, adapter


def test_exact_new_lock_and_separate_registrations():
    assert CORE_COMMIT == 'f6807a91546e43e198645a3f8925d2dc7c9e03b4'
    assert hashlib.sha256(LOCK.read_bytes()).hexdigest() == LOCK_SHA256
    assert len(entry_points(group='particle_sliders.response_backends', name='music3')) == 1
    assert len(entry_points(group='particle_sliders.inference_backends', name='music3')) == 1
    assert not entry_points(group='particle_sliders.image_backends', name='music3')


@pytest.mark.parametrize('changed', ['forward', '_pg09_scale_checkpoint'])
@pytest.mark.parametrize('conflict_index', [0, 1])
def test_product_cleanup_retains_conflict_closes_siblings_then_retries(monkeypatch, changed, conflict_index):
    from particle_sliders import pg09_backends
    model, adapter = cleanup_fixture()
    block = model.blocks[conflict_index]
    wrapped = ownership(block)
    setattr(block, changed, (lambda value: value) if changed == 'forward' else False)
    external = ownership(block)
    restored, attempts = [], []
    original = pg09_backends.AdapterScaleCheckpoint.close
    def record(handle):
        attempts.append(handle._module)
        return original(handle)
    monkeypatch.setattr(pg09_backends.AdapterScaleCheckpoint, 'close', record)
    close = product._owned_cleanup(adapter, lambda: restored.append(True))
    try:
        with pytest.raises(product.CleanupPending) as pending:
            close()
        assert pending.value.adapter is adapter
        assert ownership(block) == external and restored == []
        assert ownership(model.blocks[1 - conflict_index]) == {}
        assert attempts == [model.blocks[1], model.blocks[0]]
        assert len(adapter._checkpointing) == 1
        vars(block).update(wrapped)
        attempts.clear()
        pending.value.retry_cleanup()
        assert attempts == [block] and restored == [True]
        assert adapter._checkpointing == []
        assert all(ownership(module) == {} for module in model.blocks)
    finally:
        vars(block).update(wrapped if adapter._checkpointing else ownership(block))
        adapter.close_checkpointing()


def test_nested_same_handle_and_duplicate_wrapper_refusal_preserve_owner():
    model, adapter = cleanup_fixture()
    handle = adapter._checkpointing[0]
    block = model.blocks[0]
    wrapper = ownership(block)
    with handle:
        with handle:
            with pytest.raises(ValueError, match='already checkpointed'):
                checkpoint_with_adapter_scale(block, adapter.network)
        assert ownership(block) == wrapper
    assert ownership(block) == {}
    close = product._owned_cleanup(adapter, lambda: None)
    close()
    close()
    assert all(ownership(module) == {} for module in model.blocks)


def test_distinct_outer_wrapper_survives_product_cleanup():
    model, adapter = cleanup_fixture()
    # This block belongs to a distinct caller scope, not the product ledger.
    outer = nn.Linear(2, 2)
    with checkpoint_with_adapter_scale(outer, adapter.network):
        wrapper = ownership(outer)
        product._owned_cleanup(adapter, lambda: None)()
        assert ownership(outer) == wrapper
        assert all(ownership(module) == {} for module in model.blocks)
    assert ownership(outer) == {}


@pytest.mark.parametrize('changed', ['block_forward', 'block_marker', 'lora_forward'])
def test_real_response_binding_closes_handles_before_native_restore_and_can_retry(changed):
    run = open_response(config())
    block = run.game.backend.model.model.layers[0]
    wrapped = ownership(block)
    original = dict(run.game.backend._checkpointing[0]._saved)
    host = run.game.backend.model.model.layers[0].self_attn.q_proj
    lora_forward = host.forward
    replacement = lambda *args, **kwargs: None
    if changed == 'lora_forward':
        host.forward = replacement
    elif changed == 'block_forward':
        block.forward = replacement
    else:
        block._pg09_scale_checkpoint = False
    cleaned = False
    try:
        with pytest.raises(product.CleanupPending) as pending:
            run.close()
        assert run.closed  # Common session policy is unchanged.
        # Failed block cleanup never reaches broader LoRA/native restoration;
        # a LoRA conflict instead preserves that later owner's exact replacement.
        assert host.forward is (replacement if changed == 'lora_forward' else lora_forward)
        if changed == 'lora_forward':
            host.forward = lora_forward
            assert ownership(block) == original
        else:
            vars(block).update(wrapped)
        pending.value.retry_cleanup()
        run.binding.close()  # Product callback remains idempotent after resolution.
        assert ownership(block) == original
        assert 'forward' not in vars(host)
        cleaned = True
    finally:
        if not cleaned:
            if run.game.backend._checkpointing:
                vars(block).update(wrapped)
            if changed == 'lora_forward':
                host.forward = lora_forward
            run.binding.close()


@pytest.mark.parametrize('preset', ['bispf', 'bispf_optimized'])
def test_exact_full_split_response_resume_and_observation(preset, tmp_path):
    cfg = config(preset=preset)
    with open_response(cfg) as full:
        full.train()
        expected = full.checkpoint()
    with open_response(cfg) as first:
        first.train(stop_after=1)
        first.checkpoint(tmp_path / 'training.pt')
    with open_response(cfg, resume=tmp_path / 'training.pt') as resumed:
        resumed.train()
        equal(resumed.checkpoint(), expected)
        before, rng = resumed.checkpoint(), torch.get_rng_state().clone()
        equal(resumed.evaluate(strength=.5, seed=17), resumed.evaluate(strength=.5, seed=17))
        equal(resumed.checkpoint(), before)
        assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('selection', ['raw', 'ema'])
def test_native_only_response_is_separate_from_training_and_matches_selection(monkeypatch, selection, tmp_path):
    cfg = config()
    ids = [61, 67, 71]
    with open_response(cfg) as run:
        run.train(stop_after=1)
        packet = run.export_inference(selection=selection)
        run.export(tmp_path / 'adapter-v1.pt', selection=selection)
        assert 'rows' not in packet['specification']['configuration']
        with run._observation(11):
            if selection == 'ema':
                run.game.network.load_state_dict(run.ema.values)
            expected = run.game.backend._response(tuple(ids), scale=.5).detach().clone()
    with open_response(cfg, artifact=tmp_path / 'adapter-v1.pt', operation='response') as legacy_matching:
        assert legacy_matching.inference_only
    def forbidden(*args, **kwargs):
        raise AssertionError('native-only inference entered training construction')
    from particle_sliders import pg09, pg09_audio
    monkeypatch.setattr(product, 'prepare', forbidden)
    monkeypatch.setattr(product, 'AudioProvider', forbidden)
    monkeypatch.setattr(pg09.ConditionalPG09, '__init__', forbidden)
    monkeypatch.setattr(pg09_audio, 'AudioResponseCritic', forbidden)
    with open_inference(packet, options={'device': 'cpu'}, operation='response') as native:
        equal(native.response(ids=ids, strength=.5, seed=11), expected)
        assert not any(hasattr(native, name) for name in ('game', 'rows', 'critic', 'optimizer'))
        with pytest.raises(UnsupportedOperation):
            native.train()
        with pytest.raises(UnsupportedOperation):
            native.render()
    with pytest.raises(ValueError, match='no training data'):
        open_inference(packet, options={'rows': []}, operation='response')


def test_factory_provider_failure_releases_real_shared_checkpoint_handles(monkeypatch):
    observed = []
    original = product._build_native
    def build(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append(result[0])
        return result
    def failing(*args, **kwargs):
        raise RuntimeError('injected provider construction failure')
    monkeypatch.setattr(product, '_build_native', build)
    monkeypatch.setattr(product, 'AudioProvider', failing)
    with pytest.raises(RuntimeError, match='provider construction'):
        open_response(config())
    adapter = observed[0]
    assert adapter._checkpointing == []
    assert all(ownership(block) == {} for block in adapter.model.model.layers)
    assert all('forward' not in vars(module) for module in adapter.model.model.layers[0].self_attn.children())


@pytest.mark.parametrize('options,open_request', [
    ({'dummy': False, 'width': 4096}, {}),
    ({'dummy': True, 'width': 4096}, {}),
    ({'pg_source_profile': 'alz-continuous-v1', 'penalty_input_domain': 'continuous'}, {}),
    ({'host': 'transformer'}, {}),
    ({}, {'operation': 'render'}),
    ({}, {'strength': -.5}),
])
def test_native_width_profile_host_and_render_refusals_precede_factory(monkeypatch, options, open_request):
    def forbidden(*args, **kwargs):
        raise AssertionError('refused request allocated a native factory')
    monkeypatch.setattr(product, '_build_native', forbidden)
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(config(**options), **open_request)
    assert not torch.cuda.is_initialized()


def test_invalid_response_restore_is_transactional():
    with open_response(config()) as run:
        before = run.checkpoint()
        broken = deepcopy(before)
        broken['manifest']['config']['cohort'] = 'different-cohort'
        with pytest.raises(CheckpointError):
            run.restore(broken)
        equal(run.checkpoint(), before)
