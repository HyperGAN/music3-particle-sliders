#!/usr/bin/env python3
"""Music 3 routed-particle trainer.

Replaces ``conceptmod/textsliders/train_lora_music3_particle.py``. The LM host
and prompt card stay here. RoutedMLP, the gmix critic, and the particle
regularizer stay in particle-sliders-core and are reached through
``winning_formulation()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music3.train import main  # noqa: E402


if __name__ == "__main__":
    main(host="particle")
