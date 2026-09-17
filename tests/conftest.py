"""Shared pytest configuration.

Practice mode
-------------
Set ``MLBOOK_IMPL=practice`` to run the *same* tests against the stubs you typed
in ``practice/`` instead of the reference implementations in ``src/mlbook/``:

    MLBOOK_IMPL=practice pytest tests/test_transformer_multihead.py -q

Any ``mlbook.<area>.<module>`` import is redirected to ``practice/<area>/<module>.py``
when that file exists; everything else falls back to the reference package, so
you can practise one module at a time.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PRACTICE = ROOT / "practice"


class _PracticeRedirect(importlib.abc.MetaPathFinder):
    """Serve ``mlbook.X.Y`` from ``practice/X/Y.py`` when the practice file exists."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if not fullname.startswith("mlbook."):
            return None
        rel = Path(*fullname.split(".")[1:])
        candidate = PRACTICE / rel.with_suffix(".py")
        if candidate.is_file():
            return importlib.util.spec_from_file_location(fullname, candidate)
        return None


if os.environ.get("MLBOOK_IMPL", "").lower() == "practice":
    sys.meta_path.insert(0, _PracticeRedirect())


@pytest.fixture(autouse=True)
def _seed_everything():
    np.random.seed(0)
    torch.manual_seed(0)
