"""Installed CPU checks for ordinary LoRA factors and immutable source identity."""
from copy import deepcopy
from importlib.resources import files
import json

import pytest
import torch

from music3 import lifecycle as product
from music3.cards import release_card
from music3.distill import distill_lora
from music3 import response_source
from particle_sliders.lifecycle import SessionConfig
from particle_sliders.response import open_response
from particle_sliders.native_lora import LoRANetwork
from particle_sliders.pg09 import _hash


def config(**options):
    return SessionConfig('music3', 'music3-inventory-cpu', dummy=True, width=32,
                         steps=2, batch=2, seed=7, **options)


def test_release_catalog_is_available_without_source_checkout():
    catalog = json.loads(files('music3').joinpath('data/sample-cards.json').read_text())
    card = release_card()
    assert len(card['controls']) == len(catalog['controls']) == 16
    assert card['controls'][0]['native'].endswith('female_step1000.safetensors')


@pytest.mark.parametrize('checkpointing', [False, True])
def test_native_adapter_gradient_ownership_and_exact_zero(checkpointing):
    with open_response(config(checkpointing=checkpointing)) as run:
        adapter = run.game.backend
        base = deepcopy(adapter.model.state_dict())
        original_particles = adapter.particles.detach().clone()
        run.train()
        assert not torch.equal(original_particles, adapter.particles)
        gradient, = torch.autograd.grad(adapter._response((11, 13, 17), scale=1.).sum(), adapter.particles)
        assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
        assert _hash(base) == _hash(adapter.model.state_dict())
        assert all(not p.requires_grad and p.grad is None for p in adapter.model.parameters())
        ids = torch.tensor([[11, 13, 17]])
        with torch.no_grad(), adapter.network.scaled(0.):
            frozen = adapter.model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         use_cache=False).last_hidden_state[:, -1].float()
        assert torch.equal(frozen, adapter._response((11, 13, 17), scale=0.))


def test_distilled_factors_load_as_ordinary_native_lora_and_preserve_teacher():
    with open_response(config(checkpointing=True)) as run:
        run.train()
        adapter = run.game.backend
        before, rng = _hash(run.checkpoint()), torch.get_rng_state().clone()
        forwards = [lora.org_forward.__self__.forward for lora in adapter.slider.network.unet_loras]
        factors, provenance = distill_lora(adapter, [[11, 13, 17], [29, 31, 37]], max_rows=5)
        assert _hash(run.checkpoint()) == before and torch.equal(torch.get_rng_state(), rng)
        assert forwards == [lora.org_forward.__self__.forward for lora in adapter.slider.network.unet_loras]
        assert provenance['approximation'] and not provenance['native_audio_qualified']
        assert provenance['factor_sha256'] == _hash(factors)
        assert all(record['rows'] == 5 for record in provenance['targets'].values())
        assert all('bridge' not in name and 'particle' not in name for name in factors)
        for lora in adapter.slider.network.unet_loras:
            assert torch.equal(factors[lora.lora_name + '.lora_up.weight'], lora.lora_up.weight)
        from transformers import Qwen3ForCausalLM
        model = Qwen3ForCausalLM(adapter.model.config).eval().requires_grad_(False)
        model.load_state_dict(adapter.model.state_dict())
        ids = torch.tensor([[11, 13, 17]])
        with torch.no_grad():
            base = model.model(input_ids=ids, use_cache=False).last_hidden_state
        ordinary = LoRANetwork(model, rank=8, alpha=8., multiplier=0., delimiter='-',
                               target_replace=['Qwen3Attention'], prefix='lora_te', train_method='full')
        ordinary.load_state_dict(factors, strict=True)
        with torch.no_grad():
            off = model.model(input_ids=ids, use_cache=False).last_hidden_state
            for lora in ordinary.unet_loras:
                lora.multiplier = 1.
            on = model.model(input_ids=ids, use_cache=False).last_hidden_state
        assert torch.equal(base, off)
        assert torch.isfinite(on).all() and not torch.equal(on, off)
        provenance['model_identity']['width'] = -1
        assert adapter.model_identity['width'] == 32


def test_distillation_failure_removes_its_hooks_and_restores_observation(monkeypatch):
    from particle_sliders import distillation
    with open_response(config(checkpointing=True)) as run:
        adapter = run.game.backend
        before, rng = _hash(run.checkpoint()), torch.get_rng_state().clone()
        hooks = {id(m): dict(m._forward_pre_hooks) for m in adapter.model.modules()}
        def fail(*args, **kwargs):
            raise RuntimeError('injected shared fit failure')
        monkeypatch.setattr(distillation, 'fit_routed_down', fail)
        with pytest.raises(RuntimeError, match='injected shared fit'):
            distill_lora(adapter, [[11, 13, 17]])
        assert hooks == {id(m): dict(m._forward_pre_hooks) for m in adapter.model.modules()}
        assert _hash(run.checkpoint()) == before and torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('contexts,options', [
    ([], {}), ([[True]], {}), ([[256]], {}), ([[1] * 257], {}),
    ([[1]], {'max_rows': 513}), ([[1]], {'max_rows': True}),
    ([[1]], {'ridge_fraction': float('nan')}),
])
def test_distillation_rejects_invalid_calibration_without_teacher_mutation(contexts, options):
    adapter, _, close = product._build_native({'dummy': True, 'width': 32})
    try:
        before = _hash(adapter.slider.state_dict())
        with pytest.raises(ValueError):
            distill_lora(adapter, contexts, **options)
        assert _hash(adapter.slider.state_dict()) == before
    finally:
        close()


@pytest.mark.parametrize('subdirectory', [None, 'other-package', 'packages/particle-sliders-core'])
def test_exact_vcs_provenance_includes_package_subdirectory(monkeypatch, subdirectory):
    actual = response_source.metadata.distribution('particle-sliders-core')
    distribution = response_source.metadata.distribution
    direct = dict(url='https://github.com/HyperGAN/particle-sliders', vcs_info=dict(
        vcs='git', commit_id=response_source.CORE_COMMIT, requested_revision=response_source.CORE_COMMIT))
    if subdirectory is not None:
        direct['subdirectory'] = subdirectory
    class Dist:
        def __getattr__(self, name):
            return getattr(actual, name)
        def read_text(self, name):
            return json.dumps(direct) if name == 'direct_url.json' else actual.read_text(name)
    monkeypatch.setattr(response_source.metadata, 'distribution',
                        lambda name: Dist() if name == 'particle-sliders-core' else distribution(name))
    if subdirectory == 'packages/particle-sliders-core':
        response_source.verify_response_core()
    else:
        with pytest.raises(RuntimeError, match='VCS provenance'):
            response_source.verify_response_core()
