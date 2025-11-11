#!/usr/bin/env python3
# update_thresholds.py - write symmetric/asymmetric thresholds into a saved meta.joblib
#
# Usage examples:
#   python update_thresholds.py --meta models/ser60_meta.joblib --symmetric 0.90
#   python update_thresholds.py --meta models/ser60_meta.joblib --low 0.90 --high 0.95
#
import argparse
from joblib import load, dump

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="Path to meta.joblib")
    ap.add_argument("--symmetric", type=float, default=None, help="Single threshold for both classes")
    ap.add_argument("--low", type=float, default=None, help="Threshold for flat/low_expr")
    ap.add_argument("--high", type=float, default=None, help="Threshold for expressive/high_expr")
    args = ap.parse_args()

    meta = load(args.meta)

    # Ensure thresholds dict exists
    thr = dict(meta.get("thresholds", {}))

    # Apply updates
    if args.symmetric is not None:
        thr["symmetric"] = float(args.symmetric)
        # If low/high not explicitly set, align them to symmetric
        thr.setdefault("low_expr", float(args.symmetric))
        thr.setdefault("high_expr", float(args.symmetric))

    if args.low is not None:
        thr["low_expr"] = float(args.low)
    if args.high is not None:
        thr["high_expr"] = float(args.high)

    meta["thresholds"] = thr
    dump(meta, args.meta)
    print(f"[OK] Updated {args.meta}")
    print("thresholds:", meta["thresholds"])

if __name__ == "__main__":
    main()
