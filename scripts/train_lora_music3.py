#!/usr/bin/env python3
"""Transformer-host Music 3 slider (MiniMaxMusic3Attention / full blocks).

Historical dust/nmse defaults lived in HyperGAN/particle-sliders
``conceptmod/textsliders/train_lora_music3.py``. This entry trains
``winning_formulation()`` on the transformer host instead of copying that loss.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music3.train import main  # noqa: E402


if __name__ == "__main__":
    main(host="transformer")
