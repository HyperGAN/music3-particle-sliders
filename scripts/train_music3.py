#!/usr/bin/env python3
"""Train a Music 3 slider from this repo.

    python scripts/train_music3.py --help
    python scripts/train_music3.py --dummy

The game is particle_sliders.winning_formulation. This script does not
checkout HyperGAN/particle-sliders and does not copy the gmix modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music3.train import main  # noqa: E402


if __name__ == "__main__":
    main()
