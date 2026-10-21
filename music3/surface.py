"""Access the verified installed game class without a source-checkout fallback."""
from music3.shared import bootstrap


def load_game_class(shared_checkout=None, core_checkout=None):
    if shared_checkout is not None or core_checkout is not None:
        raise ValueError("source-checkout paths are unsupported; install the exact locked package")
    return bootstrap().core.ConditionalPG09
