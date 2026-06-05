"""Run V*Bench/MMVP/BLINK on a single checkpoint by overriding CHKPT_PATHS.

Usage:
    python evaluation/run_one_ckpt.py --ckpt /path/to/checkpoint [--plvr]

If --plvr is set, PLVR_TARGET_ONLY is False and per-stage STEP_LIST is used.
"""
import argparse
import importlib
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint dir")
    ap.add_argument("--plvr", action="store_true", help="Treat as P-LVR (B1)")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import evaluation.evaluation_local as ev
    ev.CHKPT_PATHS = [args.ckpt]
    if args.plvr:
        ev.PLVR_TARGET_ONLY = False
    ev.main()


if __name__ == "__main__":
    main()
