from __future__ import annotations

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent


def install() -> None:
    path = str(MODELS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


install()
