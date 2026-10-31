"""Exact-source admission and old-preview refusals for the final core pin."""
from copy import deepcopy
from importlib import metadata
import os
from pathlib import Path, PurePosixPath

import pytest
import torch

from music3_hndl import build_cpu_game, source
from particle_sliders import pg09


def test_final_pin_recursive_source_data_closure_and_import_origin():
    lock = source.verify_core()
    assert source.CORE_COMMIT == lock['commit'] == 'a794e3b4a8813b012f2c4faa3da82b78d4a90214'
    dist = metadata.distribution('particle-sliders-core')
    assert dist.version == '0.12.0' and metadata.version('particlegan') == '0.9.4'
    assert Path(pg09.__file__).resolve() == Path(dist.locate_file('particle_sliders/pg09.py')).resolve()
    assert {'particle_sliders/lifecycle.py', 'particle_sliders/inference.py',
            'particle_sliders/pg09.py', 'particle_sliders/fibo_projection.hndl',
            'particle_sliders/fibo_hndl_010_sources.json'} <= lock['sources'].keys()


@pytest.mark.parametrize('relative', [
    'particle_sliders/pg09.py', 'particle_sliders/lifecycle.py',
    'particle_sliders/fibo_projection.hndl', 'particle_sliders/fibo_hndl_010_sources.json',
])
def test_changed_installed_core_code_or_data_is_refused(tmp_path, monkeypatch, relative):
    actual = metadata.distribution('particle-sliders-core')
    original_distribution = source.metadata.distribution
    lock = source.verify_core()
    for name in lock['sources']:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(actual.locate_file(name)).read_bytes())
    target = tmp_path / relative
    target.write_bytes(target.read_bytes() + b'\n')
    class Dist:
        def __getattr__(self, name):
            return getattr(actual, name)
        def locate_file(self, name):
            return tmp_path / str(name)
    monkeypatch.setattr(source.metadata, 'distribution',
                        lambda name: Dist() if name == 'particle-sliders-core' else original_distribution(name))
    with pytest.raises(RuntimeError, match='installed source differs: ' + relative):
        source.verify_core()


def test_unreviewed_nested_core_data_is_refused(monkeypatch):
    actual = metadata.distribution('particle-sliders-core')
    original_distribution = source.metadata.distribution
    class Dist:
        files = [*actual.files, PurePosixPath('particle_sliders/nested/unreviewed.hndl')]
        def __getattr__(self, name):
            return getattr(actual, name)
    monkeypatch.setattr(source.metadata, 'distribution',
                        lambda name: Dist() if name == 'particle-sliders-core' else original_distribution(name))
    with pytest.raises(RuntimeError, match='source inventory differs'):
        source.verify_core()


def test_foreign_imported_core_module_is_refused(monkeypatch):
    import sys
    from types import ModuleType
    foreign = ModuleType('particle_sliders.pg09')
    foreign.__file__ = '/unreviewed/particle_sliders/pg09.py'
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    with pytest.raises(RuntimeError, match='foreign shared module already imported'):
        source.verify_core()


@pytest.mark.parametrize('kind', ['checkpoint', 'export'])
def test_actual_old_preview_packet_is_refused_without_mutation(kind):
    directory = os.environ.get('MUSIC3_OLD_PACKET_DIR')
    if not directory:
        pytest.skip('provide preserved actual 4abc preview packets; see final-pin receipt')
    packet = torch.load(Path(directory) / ('old4abc-' + kind + '.pt'), map_location='cpu', weights_only=True)
    with build_cpu_game() as run:
        before = deepcopy(run.game.state_dict())
        rng = torch.get_rng_state().clone()
        assert packet['config'] != before['config']
        load = run.game.load_state_dict if kind == 'checkpoint' else run.game.load_export_state_dict
        with pytest.raises((RuntimeError, ValueError)):
            load(packet)
        assert pg09._hash(run.game.state_dict()) == pg09._hash(before)
        assert torch.equal(torch.get_rng_state(), rng)
        assert not run.model.training and all(not p.requires_grad and p.grad is None for p in run.model.parameters())


def test_ema_option_is_not_silently_claimed_or_accepted():
    # This bounded product exposes raw low-level game exports, not session EMA.
    with pytest.raises(TypeError, match='ema_decay'):
        build_cpu_game(ema_decay=.9)
