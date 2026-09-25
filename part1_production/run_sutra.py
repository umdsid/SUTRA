#!/usr/bin/env python3
"""Canonical public SUTRA production entry point."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from sutra.cli.contextual_flow import main
if __name__ == "__main__":
    main()
