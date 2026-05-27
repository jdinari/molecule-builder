"""
conftest.py
===========
pytest configuration and shared fixtures for the molbuilder test suite.
"""

import sys
from pathlib import Path

# Ensure the package root is on sys.path so tests run from any working dir.
sys.path.insert(0, str(Path(__file__).parent.parent))
