"""Owned Music3 adapters over generic core helpers; tiny CPU hosts only."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from music3 import native
from music3.response_adapter import Music3ARResponseAdapter
from particle_sliders.pg09_audio import AudioRow


class Qwen3Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, value):
        return self.proj(value).tanh()


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.attention = Qwen3Attention()

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return SimpleNamespace(last_hidden_state=self.attention(self.embedding(input_ids)))


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()
        self.config = SimpleNamespace(hidden_size=8)
        self.eval().requires_grad_(False)


def fixture():
    torch.manual_seed(7)
    host = Host()
    slider = native.ParticleNetwork(host)
    # A nonzero synthetic delta exercises the shared bank, rather than only
    # the production zero-initialized up matrices' first-update boundary.
    with torch.no_grad():
        for lora in slider.network.unet_loras:
            lora.lora_up.weight.fill_(.1)
    adapter = Music3ARResponseAdapter(host, slider, model_identity={'fixture': 'owned-Music3-CPU'})
    return host, slider, adapter


def inventory(host):
    return [(name, id(module), 'forward' in vars(module), vars(module).get('forward'),
             tuple(parameter.requires_grad for parameter in module.parameters(recurse=False)))
            for name, module in host.named_modules()]


def test_product_response_owns_native_dispatch_and_keeps_core_tensor_math():
    host, slider, adapter = fixture()
    assert type(slider).__module__ == 'music3.native'
    assert type(adapter).__module__ == 'music3.response_adapter'
    base = deepcopy(host.state_dict())
    row = adapter.prepare(AudioRow((1, 2, 3), (4, 5, 3), history_tokens=1))
    assert torch.equal(row.neutral, adapter.sample(row, scale=0.))
    value = adapter.generated(row)
    gradient, = torch.autograd.grad(value.square().sum(), slider.particles)
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    assert all(torch.equal(value, host.state_dict()[key]) for key, value in base.items())
    assert all(not parameter.requires_grad for parameter in host.parameters())
    assert not torch.cuda.is_initialized()


def test_product_adapter_roundtrip_preserves_current_raw_values_and_rng(tmp_path):
    host, slider, adapter = fixture()
    path = tmp_path / 'raw.safetensors'
    slider.save(path, {'concept': 'CPU fixture'})
    target = Host()
    target.load_state_dict(host.state_dict())
    before = torch.get_rng_state().clone()
    restored, metadata = native.ParticleNetwork.load(target, path)
    assert metadata['weights_kind'] == 'raw'
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in slider.state_dict().items())
    receiver = Music3ARResponseAdapter(target, restored, model_identity={'fixture': 'owned-Music3-CPU'})
    assert torch.equal(adapter._response((1, 2, 3), scale=1.), receiver._response((1, 2, 3), scale=1.))


def test_late_product_load_failure_rolls_back_host_attachment_and_rng(tmp_path, monkeypatch):
    _, slider, _ = fixture()
    path = tmp_path / 'raw.safetensors'
    slider.save(path, {})
    target = Host()
    before, rng = inventory(target), torch.get_rng_state().clone()
    original = native.ParticleNetwork.load_state_dict
    def fail_after_load(self, values, **kwargs):
        original(self, values, **kwargs)
        raise RuntimeError('injected final load failure')
    monkeypatch.setattr(native.ParticleNetwork, 'load_state_dict', fail_after_load)
    with pytest.raises(RuntimeError, match='injected final load'):
        native.ParticleNetwork.load(target, path)
    assert inventory(target) == before
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('malformation', ['missing', 'extra', 'shape', 'dtype', 'nan', 'alpha'])
def test_malformed_product_checkpoint_preserves_unattached_host(tmp_path, malformation):
    _, slider, _ = fixture()
    path = tmp_path / 'adapter.safetensors'
    slider.save(path, {})
    with safe_open(path, framework='pt', device='cpu') as archive:
        metadata = archive.metadata()
    values = load_file(path)
    if malformation == 'missing':
        del values['particles']
    elif malformation == 'extra':
        values['extra'] = torch.ones(1)
    elif malformation == 'shape':
        values['particles'] = values['particles'][:-1]
    elif malformation == 'dtype':
        values['particles'] = values['particles'].double()
    elif malformation == 'nan':
        values['particles'][0, 0] = float('nan')
    else:
        key = next(key for key in values if key.endswith('.alpha'))
        values[key] = torch.tensor(7.)
    save_file(values, path, metadata=metadata)
    target = Host()
    before, state, rng = inventory(target), deepcopy(target.state_dict()), torch.get_rng_state().clone()
    with pytest.raises(ValueError, match='checkpoint'):
        native.ParticleNetwork.load(target, path)
    assert inventory(target) == before
    assert all(torch.equal(value, target.state_dict()[key]) for key, value in state.items())
    assert torch.equal(torch.get_rng_state(), rng)


def test_averaged_checkpoint_refused_before_any_host_inspection(tmp_path):
    import json
    path = tmp_path / 'averaged.safetensors'
    save_file({'weight': torch.ones(1)}, path, metadata={'conceptmod': json.dumps({'weights_kind': 'ema'})})
    with pytest.raises(ValueError, match='Averaged adapter'):
        native.ParticleNetwork.load(None, path)


def test_export_refuses_averaged_provenance_and_alternate_weights(tmp_path):
    _, slider, _ = fixture()
    with pytest.raises(ValueError, match='Averaged adapter'):
        slider.save(tmp_path / 'averaged.safetensors', {'weights_kind': 'ema'})
    state = deepcopy(slider.state_dict())
    state['particles'].add_(1.)
    with pytest.raises(ValueError, match='raw current'):
        slider.save(tmp_path / 'alternate.safetensors', {}, state=state)
    assert not list(tmp_path.iterdir())


def test_native_relocation_receipt_preserves_original_source_identity():
    import hashlib
    from importlib.resources import files
    import json
    record = json.loads(files('music3').joinpath('native-provenance.json').read_text())
    assert record['source']['commit'] == '285180ed9813f12e912bb6bce6bada00496fd19a'
    assert record['source']['sha256'] == record['source']['upstream_record']['packaged_sha256']
    assert record['product']['packaged_sha256'] == hashlib.sha256(files('music3').joinpath('native.py').read_bytes()).hexdigest()
