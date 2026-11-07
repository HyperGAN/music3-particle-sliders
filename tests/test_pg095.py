"""Current installed release identity and allocation-free native safety gates."""
from importlib import metadata
import json
from pathlib import Path

import pytest

from music3 import lifecycle as product
from music3.response_source import verify_response_core
from particle_sliders import pg09
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from particle_sliders.response import open_response


def test_installed_product_and_runtime_select_one_official_release():
    lock = verify_response_core()
    assert metadata.version("particlegan") == pg09.PG_VERSION == "0.9.5"
    assert metadata.version("particle-sliders-core") == lock["core"]["version"] == "0.14.0"
    assert lock["core"]["commit"] == product.MILESTONE == "8a696eb33c1a9d34472afad3a3199bd4776a797f"
    assert lock["particlegan"]["release_commit"] == "1b8355b184b21bb7d81104ee590813c262f6df0a"
    assert metadata.metadata("music3-particle-sliders")["Requires-Python"] == ">=3.11"
    requirements = metadata.requires("music3-particle-sliders")
    assert any(product.MILESTONE in requirement for requirement in requirements)


@pytest.mark.parametrize("profile,domain", [
    ("release", "auto"), ("release-0.9.3", "auto"),
    ("alz-continuous-v1", "continuous"), ("release-0.9.5", "continuous"),
])
def test_unqualified_profile_pairs_fail_before_factory(monkeypatch, profile, domain):
    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified profile allocated the native factory")
    monkeypatch.setattr(product, "_build_native", forbidden)
    config = SessionConfig("music3", "pg095-profile-rejection", dummy=True, width=32,
                           pg_source_profile=profile, penalty_input_domain=domain)
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(config)


@pytest.mark.parametrize("width", [170, 4096])
def test_official_release_and_product_both_reject_wide_native_vectors(monkeypatch, width):
    with pytest.raises(ValueError, match="width >=170"):
        pg09.validate_pg09_domain((1, width), response_domain="continuous_vector",
                                 pg_source_profile="release-0.9.5", penalty_input_domain="auto")
    def forbidden(*args, **kwargs):
        raise AssertionError("native width gate allocated a model")
    monkeypatch.setattr(product, "_build_native", forbidden)
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(SessionConfig("music3", "pg095-width-rejection", dummy=True, width=width))


def test_historical_cli_and_response_locks_remain_separate():
    root = Path(__file__).resolve().parents[1]
    old_cli = json.loads((root / "core.lock.json").read_text())
    old_response = json.loads((root / "cohorts/legacy-response/response-core.lock.json").read_text())
    assert old_cli["version"] == "0.6.0" and old_cli["particlegan_version"] == "0.9.1"
    assert old_response["core"]["version"] == "0.8.2"
    assert old_response["particlegan"]["version"] == "0.9.1"


def test_current_core_contains_no_owned_music3_implementation():
    from importlib.util import find_spec
    from particle_sliders import pg09_audio
    from music3.response_adapter import Music3ARResponseAdapter
    assert find_spec('particle_sliders.music3_native') is None
    assert not hasattr(pg09_audio, 'Music3ARResponseAdapter')
    assert issubclass(Music3ARResponseAdapter, pg09_audio.ARLastHiddenResponseAdapter)
    assert Music3ARResponseAdapter.__module__ == 'music3.response_adapter'


@pytest.mark.parametrize('decay', [0., .9, True])
def test_product_refuses_weight_ema_before_native_admission(monkeypatch, decay):
    monkeypatch.setattr(product, '_validate_model', lambda *a: pytest.fail('native admission before RAW policy'))
    with pytest.raises(ValueError, match='Weight EMA is disabled'):
        product.validate(dict(dummy=True, width=32, ema_decay=decay))


def test_previous_pg095_response_lock_keeps_its_original_hash():
    import hashlib
    root = Path(__file__).resolve().parents[1] / 'cohorts/pg095-response-3dd-20261105'
    receipt = json.loads((root / 'source.json').read_text())
    contents = (root / 'response-core.lock.json').read_bytes()
    assert hashlib.sha256(contents).hexdigest() == receipt['files']['music3/response-core.lock.json']
    assert json.loads(contents)['core']['commit'] == '3dd672a7f9b8c169daef90fd63e4bb42b9ddc423'
    assert receipt['product_commit'] == 'a5791a5373fdd3043c6a692601c9a915119c88a0'

    preintegration = Path(__file__).resolve().parents[1] / 'cohorts/product-owned-native-core9e9dcea-20261107'
    retained = (preintegration / 'response-core.lock.json').read_bytes()
    record = json.loads((preintegration / 'source.json').read_text())
    assert hashlib.sha256(retained).hexdigest() == '2a535e742c9007b65ca04d45afe643b755105189b6dd8b5a647de31c0752ddef'
    assert record['files']['music3/response-core.lock.json'] == hashlib.sha256(retained).hexdigest()
    assert json.loads(retained)['core']['commit'] == '9e9dcea567b38a550bd50e2fea5c2fc3e669706c'
    assert record['product_commit'] == 'abb871b0a8fdfb986cff66b891398ed44652fad3'
