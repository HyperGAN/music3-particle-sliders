#!/usr/bin/env python3
"""Language-model host Music 3 slider (Qwen3Attention).

The historical ``--lm_target v9`` menu remains in HyperGAN/particle-sliders
``conceptmod/textsliders/train_lm_slider_music3.py``. This product entry uses
the winning stamp's teacher (``faithful_plus_neu``) and refuses other targets.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music3.train import main  # noqa: E402


if __name__ == "__main__":
    main(host="lm")
