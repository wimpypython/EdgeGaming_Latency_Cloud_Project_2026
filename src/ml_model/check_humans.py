"""
Check how much of the CSKnow feature store is human play vs. bot replay.

This file is CSKnow's evaluation feature store — some traces are their
trained bot playing, not humans. Training a human-movement predictor on
bot traces would quietly invalidate the result, so check the split before
building the dataset.

Usage:
    python check_humans.py
    python check_humans.py path/to/file.hdf5
"""

import sys
import h5py
import numpy as np

DEFAULT_PATH = r"C:\Users\Atharva\Downloads\sample_csknow\all_train_outputs\behaviorTreeTeamFeatureStore_28.hdf5"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"Opening: {path}\n")

    with h5py.File(path, "r") as f:
        # --- per-trace bot flags (in the 'extra' group) ---
        print("=== Trace-level flags (extra/) ===")
        n_traces = f["extra/trace index"].shape[0]
        print(f"Total traces: {n_traces}\n")

        bot_any = np.zeros(n_traces, dtype=bool)
        for team in ("CT", "T"):
            for i in range(5):
                key = f"extra/trace is bot player {team} {i}"
                if key in f:
                    flags = f[key][:]
                    bot_any |= flags
                    print(f"  {team} {i}: {flags.sum():>4} / {n_traces} traces are bot")

        print()
        print(f"Traces with ANY bot player : {bot_any.sum():>4} / {n_traces}")
        print(f"Traces fully human         : {(~bot_any).sum():>4} / {n_traces}")

        for key in ("extra/trace one non replay bot", "extra/trace one non replay team"):
            if key in f:
                v = f[key][:]
                print(f"{key.split('/')[-1]:<28}: {v.sum():>4} True")

        # --- row-level validity ---
        print("\n=== Row-level flags (data/) ===")
        n_rows = f["data/valid"].shape[0]
        valid = f["data/valid"][:]
        print(f"Total rows      : {n_rows:,}")
        print(f"Rows marked valid: {valid.sum():,} ({valid.mean() * 100:.1f}%)")

        if "data/test success" in f:
            ts = f["data/test success"][:]
            print(f"test success True: {ts.sum():,} ({ts.mean() * 100:.1f}%)")

        # --- test names, which usually reveal the trace type ---
        if "data/test name" in f:
            names = f["data/test name"][:]
            decoded = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
            uniq, counts = np.unique(decoded, return_counts=True)
            print(f"\nDistinct test names: {len(uniq)}")
            for u, c in sorted(zip(uniq, counts), key=lambda x: -x[1])[:15]:
                print(f"  {c:>8,}  {u}")

        # --- source demos ---
        if "extra/demo file" in f:
            demos = f["extra/demo file"][:]
            print(f"\nSource demo files: {len(demos)}")
            for d in demos[:5]:
                print("  ", d.decode() if isinstance(d, bytes) else d)
            if len(demos) > 5:
                print(f"   ... and {len(demos) - 5} more")


if __name__ == "__main__":
    main()
