"""Separate Music3 core0.12 / HyperGAN develop / HNDL0.10 candidate.

Import performs no model allocation. Use build_cpu_game for bounded software
qualification. Native pretrained execution and rendering are not admitted.
"""

def build_cpu_game(**options):
    from .binding import build_cpu_game as build
    return build(**options)
