"""Conftest for tests.

Ensure beartype runtime checking is enabled before importing the package.

This module sets MESA_FRAMES_RUNTIME_TYPECHECKING=1 at import time so tests that
assert beartype failures at import or construct time behave deterministically.
"""

import os

os.environ.setdefault("MESA_FRAMES_RUNTIME_TYPECHECKING", "1")
