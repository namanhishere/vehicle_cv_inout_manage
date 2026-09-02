#!/usr/bin/env python3
"""Wrapper so docs can reference `gate_cli.py passwd` etc."""

import sys

from gate.cli import main

if __name__ == "__main__":
    sys.exit(main())
