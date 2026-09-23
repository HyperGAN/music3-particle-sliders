#!/usr/bin/env python3
"""Condition-encoder host Music 3 slider (MiniMaxMusic3ConditionEncoder).

The historical MSE/curriculum encoder loop remains in HyperGAN/particle-sliders
``conceptmod/textsliders/train_encoder_music3.py``. Rank and alpha are the
stamp's adapter size. This entry does not copy that loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music3.train import main  # noqa: E402


if __name__ == "__main__":
    main(host="encoder")
