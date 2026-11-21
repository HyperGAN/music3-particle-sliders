"""Access the exact installed shared game class."""


def load_game_class(shared_checkout=None, core_checkout=None):
    if shared_checkout is not None or core_checkout is not None:
        raise ValueError("source-checkout paths are unsupported; install the exact locked package")

    from music3.response_source import verify_response_core

    verify_response_core()
    from particle_sliders.engine import ConditionalParticleGAN

    return ConditionalParticleGAN
