"""
Answer four questions about the CSKnow feature store before any modelling.

WHY THIS EXISTS
    Everything downstream -- the dead-reckoning baseline, the reference
    "18.1 units" figure, the train/test split -- depends on facts about this
    file that were assumed rather than measured. This script measures them.

    Q1  Is history step t-1 exactly one row back, or some other stride?
    Q2  Is velocity in units per SECOND or units per STEP?
    Q3  What is a "round id"? Does it repeat across demos?
    Q4  Is there a per-row demo index we could split on?

    It reads the file and prints. It changes nothing.

Usage:
    python check_alignment.py
    python check_alignment.py --input /mnt/c/Users/Atharva/Downloads/...
"""

import argparse

import h5py
import numpy as np

DEFAULT_INPUT = (
    r"C:\Users\Atharva\Downloads\sample_csknow\all_train_outputs"
    r"\behaviorTreeTeamFeatureStore_28.hdf5"
)

STEP_SECONDS = 0.0625   # 8 game ticks at 128 Hz -- the value under test
PLAYERS = [("CT", 0), ("T", 0)]
CANDIDATE_LAGS = [1, 2, 3, 4, 8]


def rule(title):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def get_xyz(f, base):
    """Stack the three separate 1-D arrays for one quantity into (n, 3)."""
    return np.stack([f[f"{base} {ax}"][:] for ax in "xyz"], axis=1)


# ---------------------------------------------------------------------------
# Q1 -- history stride
# ---------------------------------------------------------------------------

def q1_history_stride(f, round_ids):
    rule("Q1  How many rows back is history step t-1?")
    print("If t-1 is one row back, comparing 'pos t-1' at row i against")
    print("'pos' at row i-1 should give a difference of essentially zero.\n")

    for team, slot in PLAYERS:
        base = f"data/player pos {team} {slot}"
        now = get_xyz(f, base)
        h1 = get_xyz(f, f"{base} t-1")

        valid = f[f"data/player history valid {team} {slot} t-1"][:].astype(bool)
        alive = f[f"data/alive {team} {slot}"][:].astype(bool)
        usable = valid & alive

        print(f"  {team}{slot}:  {usable.sum():,} rows alive with t-1 valid")

        best = None
        for lag in CANDIDATE_LAGS:
            # Compare row i against row i-lag, staying inside one round block.
            idx = np.arange(lag, len(now))
            same = round_ids[idx] == round_ids[idx - lag]
            sel = idx[same & usable[idx]]
            if len(sel) == 0:
                continue

            diff = np.linalg.norm(h1[sel] - now[sel - lag], axis=1)
            med = np.median(diff)
            exact = np.mean(diff < 1e-3) * 100

            print(f"      lag {lag}:  median |diff| = {med:9.3f} units"
                  f"   exact matches {exact:5.1f}%   (n={len(sel):,})")

            if best is None or med < best[1]:
                best = (lag, med)

        if best:
            print(f"      -> best fit: lag {best[0]}"
                  f" (median {best[1]:.3f})")
            if best[1] > 1.0:
                print("      -> WARNING: no candidate lag matches cleanly.")
                print("         History is not a simple row offset. Stop and")
                print("         work out what it is before building anything.")
        print()


# ---------------------------------------------------------------------------
# Q2 -- velocity units
# ---------------------------------------------------------------------------

def q2_velocity_units(f):
    rule("Q2  Is velocity units-per-second or units-per-step?")
    print("Inside a single row: distance travelled from history step t-2 to")
    print("t-1, divided by the speed recorded at t-2.")
    print("  ratio near 0.0625 -> velocity is units per SECOND")
    print("  ratio near 1.0    -> velocity is units per STEP\n")

    for team, slot in PLAYERS:
        p = f"data/player pos {team} {slot}"
        v = f"data/player velocity {team} {slot}"

        pos_t2 = get_xyz(f, f"{p} t-2")
        pos_t1 = get_xyz(f, f"{p} t-1")
        vel_t2 = get_xyz(f, f"{v} t-2")

        ok = (f[f"data/player history valid {team} {slot} t-1"][:].astype(bool)
              & f[f"data/player history valid {team} {slot} t-2"][:].astype(bool)
              & f[f"data/alive {team} {slot}"][:].astype(bool))

        dist = np.linalg.norm(pos_t1 - pos_t2, axis=1)
        speed = np.linalg.norm(vel_t2, axis=1)

        # Only use rows where the player is genuinely moving. A stationary
        # player gives 0/0 and tells us nothing. Direction changes blur the
        # ratio, so the median over fast rows is the trustworthy statistic.
        moving = ok & (speed > 50)
        if moving.sum() < 100:
            print(f"  {team}{slot}: too few moving rows to judge")
            continue

        ratio = dist[moving] / speed[moving]

        print(f"  {team}{slot}:  n={moving.sum():,} moving rows")
        print(f"      median ratio      : {np.median(ratio):.4f}")
        print(f"      25th / 75th pct   : {np.percentile(ratio, 25):.4f}"
              f" / {np.percentile(ratio, 75):.4f}")
        print(f"      median speed      : {np.median(speed[moving]):.1f}"
              f"  (game units per whatever the unit is)")

        med = np.median(ratio)
        if abs(med - STEP_SECONDS) < abs(med - 1.0):
            print(f"      -> consistent with units per SECOND"
                  f" (dt = {STEP_SECONDS})")
        else:
            print("      -> consistent with units per STEP (dt = 1)")
        print()


# ---------------------------------------------------------------------------
# Q3 -- round structure
# ---------------------------------------------------------------------------

def q3_round_structure(f, round_ids):
    rule("Q3  What is a round id?")

    n = len(round_ids)
    uniq = np.unique(round_ids)

    # A "block" is a run of identical consecutive round ids -- one round as
    # it actually appears in the file.
    changes = np.flatnonzero(round_ids[1:] != round_ids[:-1]) + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [n]])
    block_len = ends - starts
    block_id = round_ids[starts]

    print(f"  rows                    : {n:,}")
    print(f"  distinct round id values: {len(uniq)}")
    print(f"  contiguous blocks       : {len(block_len)}")
    print(f"  id range                : {uniq.min()} .. {uniq.max()}")

    print(f"\n  rows per block  median {np.median(block_len):,.0f}"
          f"   min {block_len.min():,}   max {block_len.max():,}")
    print(f"  seconds per block (at {STEP_SECONDS}s/row):"
          f"  median {np.median(block_len) * STEP_SECONDS:.1f}"
          f"   max {block_len.max() * STEP_SECONDS:.1f}")
    print("  (A competitive CS round cannot exceed ~155s.)")

    if len(block_len) > len(uniq):
        counts = np.bincount(np.searchsorted(uniq, block_id))
        print(f"\n  -> ROUND IDS REPEAT. {len(block_len)} blocks share only"
              f" {len(uniq)} ids.")
        print(f"     Most reused id appears in {counts.max()} separate blocks.")
        print("     Splitting train/test on round id would put the SAME id")
        print("     on both sides. Split on blocks or on demo instead.")
    else:
        print("\n  -> Each round id appears as exactly one block.")

    # Game tick resets mark where one demo ends and the next begins.
    if "data/game tick number" in f:
        tick = f["data/game tick number"][:]
        resets = int(np.sum(np.diff(tick.astype(np.int64)) < 0))
        print(f"\n  game tick number goes backwards {resets} times")
        print(f"  -> suggests roughly {resets + 1} separate demos in the file")
        print("     (handoff says 25 source demos)")


# ---------------------------------------------------------------------------
# Q4 -- demo index
# ---------------------------------------------------------------------------

def q4_demo_index(f, n_rows):
    rule("Q4  Is there a per-row demo index to split on?")

    hits = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            low = name.lower()
            if any(k in low for k in ("demo", "game id", "match", "source", "file")):
                hits.append((name, obj.shape, obj.dtype))

    f.visititems(visit)

    if not hits:
        print("  No dataset name mentions a demo/match/source.")
        return

    for name, shape, dtype in hits:
        per_row = " <- PER-ROW, usable for splitting" if shape == (n_rows,) else ""
        print(f"  {name:<45} {str(shape):<12} {dtype}{per_row}")

    print("\n  A per-row demo index is the split you want: rounds from one")
    print("  match share players, economy and site preferences, so a")
    print("  round-level split still leaks match identity into the test set.")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    args = ap.parse_args()

    print(f"Reading: {args.input}")

    with h5py.File(args.input, "r") as f:
        round_ids = f["data/round id"][:]
        q1_history_stride(f, round_ids)
        q2_velocity_units(f)
        q3_round_structure(f, round_ids)
        q4_demo_index(f, len(round_ids))

    print("\nDone. Nothing was modified.")


if __name__ == "__main__":
    main()
