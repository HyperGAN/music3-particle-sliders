"""Bind this product to the shared stamp. Model surfaces may move; the game may not."""

from __future__ import annotations

from particle_sliders import winning_formulation

from music3.defaults import MUSIC_ADV_BATCH


def load_stamp():
    """The only game entry point. Do not construct a local critic or router."""
    return winning_formulation()


def declare(stamp, *, g_lr=None, d_lr=None, particle_lr=None, adv_batch=None) -> dict:
    """Copy the stamp and apply Music 3 model-surface overrides, then require()."""
    declared = stamp.as_dict()
    if g_lr is not None:
        declared["g_lr"] = float(g_lr)
    if d_lr is not None:
        declared["d_lr"] = float(d_lr)
    if particle_lr is not None:
        declared["particle_lr"] = float(particle_lr)
    declared["adv_batch"] = int(MUSIC_ADV_BATCH if adv_batch is None else adv_batch)
    stamp.require(declared)
    return declared
