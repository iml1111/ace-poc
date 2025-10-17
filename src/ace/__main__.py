"""
Entry point for running ACE framework as a module.

Usage:
    python -m ace offline --dataset labeling --epochs 2
    python -m ace online --dataset labeling
    python -m ace stats
"""

from .cli import main

if __name__ == "__main__":
    main()
