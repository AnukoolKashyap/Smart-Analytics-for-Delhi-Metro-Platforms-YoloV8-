"""
Shared pytest setup. Adds src/ to sys.path so tests can do plain
`from zones import ZoneManager` etc., matching how the rest of the project
already imports between its own modules (no packaging/installation step
required to run either the app or the tests).
"""

import os
import sys

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC_DIR)
