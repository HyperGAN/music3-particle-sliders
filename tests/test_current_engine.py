"""Current installed release identity and allocation-free native safety gates."""
from importlib import metadata

import pytest

from music3 import lifecycle as product
from music3.response_source import verify_response_core
from particle_sliders import engine
from particle_sliders.lifecycle import SessionConfig, UnsupportedOperation
from particle_sliders.response import open_response


def test_installed_product_and_runtime_select_one_official_release():
    lock = verify_response_core()
    assert metadata.version("particlegan") == engine.PG_VERSION == "0.11.1"
    assert metadata.version("particle-sliders-core") == lock["core"]["version"] == "0.15.2"
    assert lock["core"]["commit"] == product.MILESTONE == "ec1161f5d74481be3540cb0dfa63d0e8f8b54e90"
    assert lock["particlegan"]["release_commit"] == engine.PG_SOURCE_COMMIT
    assert metadata.metadata("music3-particle-sliders")["Requires-Python"] == ">=3.11"
    requirements = metadata.requires("music3-particle-sliders")
    assert any(product.MILESTONE in requirement for requirement in requirements)


@pytest.mark.parametrize("profile,domain", [
    ("release", "auto"), ("release-0.11.0", "auto"),
    ("alz-continuous-v1", "continuous"), ("release-0.11.1", "continuous"),
])
def test_unqualified_profile_pairs_fail_before_factory(monkeypatch, profile, domain):
    def forbidden(*args, **kwargs):
        raise AssertionError("unqualified profile allocated the native factory")
    monkeypatch.setattr(product, "_build_native", forbidden)
    config = SessionConfig("music3", "current-profile-rejection", dummy=True, width=32,
                           pg_source_profile=profile, penalty_input_domain=domain)
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(config)


@pytest.mark.parametrize("width", [170, 4096])
def test_official_release_and_product_both_reject_wide_native_vectors(monkeypatch, width):
    with pytest.raises(ValueError, match="width >=170"):
        engine.validate_engine_domain((1, width), response_domain="continuous_vector",
                                 pg_source_profile="release-0.11.1", penalty_input_domain="auto")
    def forbidden(*args, **kwargs):
        raise AssertionError("native width gate allocated a model")
    monkeypatch.setattr(product, "_build_native", forbidden)
    with pytest.raises((ValueError, UnsupportedOperation)):
        open_response(SessionConfig("music3", "auto-width-rejection", dummy=True, width=width))

def test_installed_surface_loads_the_admitted_game_class():
    from music3.surface import load_game_class

    assert load_game_class() is engine.ConditionalParticleGAN
    with pytest.raises(ValueError, match="source-checkout"):
        load_game_class(shared_checkout="another-tree")
