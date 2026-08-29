"""
Check the tick spacing in the CSKnow feature store.

This determines your prediction horizon: the 48 history steps (t-1 .. t-48)
cover a different real-time window depending on whether ticks are consecutive
(128 Hz) or decimated.

Usage:
    python check_ticks.py
    python check_ticks.py path/to/file.hdf5
"""

import sys
import h5py
import numpy as np

DEFAULT_PATH = r"C:\Users\Atharva\Downloads\sample_csknow\all_train_outputs\behaviorTreeTeamFeatureStore_28.hdf5"

TICK_RATE = 128.0  # competitive CS:GO


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"Opening: {path}\n")

    with h5py.File(path, "r") as f:
        n = f["data/game tick number"].shape[0]
        sample = min(n, 20000)

        ticks = f["data/game tick number"][:sample]
        rounds = f["data/round id"][:sample]

        print(f"Total rows in file : {n:,}")
        print(f"Rows sampled       : {sample:,}")
        print(f"Distinct rounds    : {len(np.unique(rounds)):,}\n")

        # Only compare consecutive rows inside the same round —
        # round boundaries produce meaningless jumps.
        same_round = rounds[1:] == rounds[:-1]
        diffs = np.diff(ticks)[same_round]

        if diffs.size == 0:
            print("No within-round consecutive rows found in this sample.")
            return

        vals, counts = np.unique(diffs, return_counts=True)
        order = np.argsort(-counts)

        print("Most common tick gaps between consecutive rows:")
        print(f"{'gap':>8} {'count':>10} {'share':>8}  {'= seconds':>12}")
        for i in order[:8]:
            gap = int(vals[i])
            share = counts[i] / diffs.size * 100
            secs = gap / TICK_RATE
            print(f"{gap:>8} {counts[i]:>10,} {share:>7.1f}%  {secs:>11.4f}s")

        dominant = int(vals[order[0]])
        print()
        print(f"Dominant gap: {dominant} tick(s) = {dominant / TICK_RATE:.4f}s")

        window_ticks = dominant * 48
        window_secs = window_ticks / TICK_RATE
        print(f"48 history steps span: {window_ticks} ticks = {window_secs:.3f}s")
        print(f"Effective sample rate: {TICK_RATE / dominant:.1f} Hz")
        print()

        if window_secs < 1.0:
            print("Good fit: sub-second history window suits packet-loss")
            print("compensation. Frame the task as short-horizon prediction.")
        else:
            print("Note: history spans over a second. Frame the prediction")
            print("horizon accordingly in the report, and say so plainly.")


if __name__ == "__main__":
    main()
