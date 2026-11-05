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
    assert metadata.version("particle-sliders-core") == lock["core"]["version"] == "0.13.0"
    assert lock["core"]["commit"] == product.MILESTONE == "3dd672a7f9b8c169daef90fd63e4bb42b9ddc423"
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
