"""MiniMax Music 3 particle sliders.

Importing this package does not download Hub weights. The shared game is
``particle_sliders.winning_formulation``; this package owns Music 3 hosts,
prompt cards, and the train/infer wiring around that stamp.
"""

from music3.defaults import BASE_MODEL_ID, HUB_WEIGHTS_ID

__all__ = ["BASE_MODEL_ID", "HUB_WEIGHTS_ID"]
