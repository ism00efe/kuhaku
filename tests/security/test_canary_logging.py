"""The canary token itself never reaches a log; only a hash prefix does."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

SCRIPT = """
import logging, sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
from kuhaku.core.security import guard
print(guard.CANARY_TOKEN)
print(guard.CANARY_FINGERPRINT)
"""


def test_canary_token_is_not_logged_at_import():
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
        check=True,
        timeout=120,
    )
    token, fingerprint = proc.stdout.split()
    assert token not in proc.stderr
    assert fingerprint in proc.stderr
