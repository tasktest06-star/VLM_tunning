#!/usr/bin/env python3
"""Thin entry point. The logic is in vlmlab/preflight.py so it can be tested."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
from vlmlab.cli import main

if __name__ == "__main__":
    sys.exit(main(["preflight"] + sys.argv[1:]))
