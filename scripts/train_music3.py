#!/usr/bin/env python3
"""Route Music3 training to the reviewed shared PG09 worker; native runs require coordinator admission."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from music3.train import main  # noqa: E402


if __name__ == "__main__":
    main()
