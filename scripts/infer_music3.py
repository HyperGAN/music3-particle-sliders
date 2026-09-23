#!/usr/bin/env python3
"""Write a Music 3 listen card from this repo.

    python scripts/infer_music3.py --help
    python scripts/infer_music3.py --dummy
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music3.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
