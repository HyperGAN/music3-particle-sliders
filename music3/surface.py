"""Source-verified shared game access; this product owns no game mathematics."""
from music3.shared import bootstrap


def load_game_class(shared_checkout=None, core_checkout=None):
    return bootstrap(shared_checkout, core_checkout).core.ConditionalPG09
