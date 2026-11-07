"""Product-owned Music3 effective-scale restoration over tiny CPU hosts."""
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn

from music3 import native


class Qwen3Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)

    def forward(self, value):
        return self.q(value) + self.k(value)


def fixture():
    torch.manual_seed(5)
    host = nn.Sequential(Qwen3Attention()).eval().requires_grad_(False)
    slider = native.ParticleNetwork(host)
    assert len(slider.network.unet_loras) == 2
    with torch.no_grad():
        for layer in slider.network.unet_loras:
            layer.lora_up.weight.fill_(.2)
    slider.network.set_lora_slider(2.5)
    for layer, multiplier in zip(slider.network.unet_loras, (.25, -.75)):
        layer.multiplier = multiplier
    return host, slider, torch.randn(1, 3, 4)


@pytest.mark.parametrize('failed_body', [False, True])
def test_nested_scaled_scope_restores_heterogeneous_effective_outputs(failed_body):
    host, slider, values = fixture()
    original = host(values).detach().clone()
    with slider.scaled(.5):
        outer = host(values).detach().clone()
        assert slider.network.lora_scale == .5
        assert all(layer.multiplier == .5 for layer in slider.network.unet_loras)
        def inner():
            with slider.scaled(-1.):
                assert all(layer.multiplier == -1. for layer in slider.network.unet_loras)
                assert not torch.equal(host(values), outer)
                if failed_body:
                    raise ValueError('body failure')
        if failed_body:
            with pytest.raises(ValueError, match='body failure'):
                inner()
        else:
            inner()
        assert slider.network.lora_scale == .5
        assert torch.equal(host(values), outer)
    assert slider.network.lora_scale == 2.5
    assert [layer.multiplier for layer in slider.network.unet_loras] == [.25, -.75]
    assert torch.equal(host(values), original)
    assert not torch.cuda.is_initialized()


def test_partial_entry_failure_restores_scalar_and_every_multiplier(monkeypatch):
    host, slider, values = fixture()
    original = host(values).detach().clone()
    def fail(scale):
        slider.network.set_lora_slider(scale)
        slider.network.unet_loras[0].multiplier = 99.
        raise RuntimeError('partial entry failure')
    monkeypatch.setattr(slider, 'set_scale', fail)
    with pytest.raises(RuntimeError, match='partial entry failure'):
        with slider.scaled(.5):
            pytest.fail('failed entry reached body')
    assert slider.network.lora_scale == 2.5
    assert [layer.multiplier for layer in slider.network.unet_loras] == [.25, -.75]
    assert torch.equal(host(values), original)
    assert slider.network._activation_stack == []


def test_foreign_owner_cannot_enter_or_set_active_scope():
    _, slider, _ = fixture()
    def compete():
        with pytest.raises(RuntimeError, match='owned by another'):
            with slider.scaled(-1.):
                pytest.fail('foreign owner entered')
        with pytest.raises(RuntimeError, match='owned by another'):
            slider.set_scale(-1.)
    with slider.scaled(.5):
        with ThreadPoolExecutor(max_workers=1) as workers:
            workers.submit(compete).result(timeout=5)
        assert slider.network.lora_scale == .5
        assert all(layer.multiplier == .5 for layer in slider.network.unet_loras)
    assert slider.network.lora_scale == 2.5
    assert [layer.multiplier for layer in slider.network.unet_loras] == [.25, -.75]
