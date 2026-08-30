"""
Parse the CSKnow feature store into training-ready arrays.

WHY THIS VERSION EXISTS (read this before changing anything)
    The first version of this parser read the file's own "t-1", "t-2", ...
    history fields. Measurement on 30 Aug 2026 (check_alignment.py) showed
    those fields step TWO rows at a time -- 125ms, not the 62.5ms every
    document assumed. That silently made the prediction gap 187.5ms while
    calling it 62.5ms, and it would have made dead reckoning extrapolate a
    third of the required distance.

    So this version ignores the stored history fields completely and builds
    history from the row timeline itself. Rows are 8 game ticks apart, which
    this script verifies rather than assumes. Every frame in X is therefore
    exactly 62.5ms from the next, and the gap from the last input frame to
    the target is exactly HORIZON rows.

WHAT ONE EXAMPLE LOOKS LIKE
    Row i is "now". X[i] is rows i-HISTORY_LEN+1 .. i inclusive, so the last
    frame of X IS the current state. y[i] is the position at row i+HORIZON.

        HORIZON = 1  ->  62.5ms ahead   (~4 dropped updates at 64Hz)
        HORIZON = 3  ->  187.5ms ahead  (~12 dropped updates)

    Note this is burst loss, not single-packet loss. A CS update interval is
    15.6ms at 64Hz; 62.5ms is several consecutive drops. Say that, not
    "one lost packet".

WHAT A BLOCK IS
    A block is one continuous stretch of gameplay: same game id, same round
    id, and consecutive rows exactly 8 ticks apart. Examples never cross a
    block boundary. Splitting train/test uses GAME id, not round id, because
    rounds from one match share players, economy and site preferences.

Usage:
    python parse_csknow.py
    python parse_csknow.py --history 16 --horizon 3
"""

import argparse
import os

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

DEFAULT_INPUT = (
    r"C:\Users\Atharva\Downloads\sample_csknow\all_train_outputs"
    r"\behaviorTreeTeamFeatureStore_28.hdf5"
)
DEFAULT_OUTPUT = "parsed_data.npz"

HISTORY_LEN = 16        # past rows fed to the model, INCLUDING the current row
HORIZON = 1             # rows ahead to predict

TEAMS = ("CT", "T")
SLOTS = (0, 1, 2, 3, 4)

TICKS_PER_ROW = 8       # verified, not assumed -- see build_blocks()
STEP_SECONDS = 0.0625   # 8 ticks at 128 Hz

# Fastest a CS player can plausibly move on the ground is ~250 u/s. Allow
# generous headroom for falls and boosts; anything past this means the
# example spans a teleport or a boundary we failed to catch.
MAX_PLAUSIBLE_SPEED = 600.0

# Nothing in CS moves this fast. A single example past this is a teleport,
# which means an example crossed a boundary. Abort on one, not on a
# percentage -- see the note next to the check.
TELEPORT_SPEED = 1500.0


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def build_blocks(round_ids, game_ids, ticks):
    """
    Label every row with a block index.

    A new block starts wherever the game changes, the round changes, or the
    tick gap is not exactly TICKS_PER_ROW. That last condition is what makes
    this self-healing: if the file has hidden discontinuities, they become
    block boundaries instead of silently corrupting a training example.
    """
    n = len(round_ids)
    tick_gap = np.diff(ticks.astype(np.int64))

    boundary = (
        (np.diff(game_ids) != 0)
        | (np.diff(round_ids) != 0)
        | (tick_gap != TICKS_PER_ROW)
    )
    block_ids = np.concatenate([[0], np.cumsum(boundary)])

    # Report how much of the file departs from the assumed spacing.
    inside = ~((np.diff(game_ids) != 0) | (np.diff(round_ids) != 0))
    if inside.sum():
        bad = (tick_gap[inside] != TICKS_PER_ROW).sum()
        pct = bad / inside.sum() * 100
        print(f"  tick spacing : {100 - pct:.2f}% of within-round gaps are"
              f" exactly {TICKS_PER_ROW} ticks")
        if pct > 0:
            odd = np.unique(tick_gap[inside][tick_gap[inside] != TICKS_PER_ROW])
            print(f"                 {bad:,} exceptions, gaps seen: "
                  f"{odd[:10].tolist()}")
            print("                 these became block boundaries, not bad rows")
        if pct > 20:
            raise SystemExit(
                "ABORT: over 20% of rows are not 8 ticks apart. The 16Hz "
                "assumption is wrong. Investigate before parsing."
            )
    return block_ids


# ---------------------------------------------------------------------------
# Per-player extraction
# ---------------------------------------------------------------------------

def load_player(f, team, slot):
    """Return feats (n_rows, 6) = [px py pz vx vy vz] and alive (n_rows,)."""
    p = f"data/player pos {team} {slot}"
    v = f"data/player velocity {team} {slot}"
    pos = np.stack([f[f"{p} {ax}"][:] for ax in "xyz"], axis=1)
    vel = np.stack([f[f"{v} {ax}"][:] for ax in "xyz"], axis=1)
    alive = f[f"data/alive {team} {slot}"][:].astype(bool)
    return np.concatenate([pos, vel], axis=1).astype(np.float32), alive


def windows(feats, alive, block_ids, history_len, horizon):
    """
    Build every valid (history, target) pair for one player.

    Returns X (m, history_len, 6), y (m, 3), and the row index of "now" for
    each example so the caller can attach round and game labels.
    """
    n = len(feats)
    rows = np.arange(n)

    # Index matrix: row i looks back to i-history_len+1, ending AT i.
    offsets = np.arange(-history_len + 1, 1)
    hist_idx = rows[:, None] + offsets[None, :]          # (n, history_len)
    tgt_idx = rows + horizon                             # (n,)

    in_range = (hist_idx[:, 0] >= 0) & (tgt_idx < n)

    # Clip so fancy indexing is safe; invalid rows get filtered out anyway.
    safe_hist = np.clip(hist_idx, 0, n - 1)
    safe_tgt = np.clip(tgt_idx, 0, n - 1)

    # Every history row, and the target row, must sit in the same block.
    same_block = (
        (block_ids[safe_hist] == block_ids[:, None]).all(axis=1)
        & (block_ids[safe_tgt] == block_ids)
    )

    # The player must be alive throughout the history and at the target.
    alive_all = alive[safe_hist].all(axis=1) & alive[safe_tgt]

    keep = in_range & same_block & alive_all
    if not keep.any():
        return None, None, None

    X = feats[safe_hist[keep]]                           # (m, history_len, 6)
    y = feats[safe_tgt[keep], :3]                        # (m, 3)
    return X, y, rows[keep]


# ---------------------------------------------------------------------------

def build(input_path, output_path, history_len, horizon):
    print(f"Reading : {input_path}")
    print(f"History : {history_len} rows "
          f"({history_len * STEP_SECONDS:.3f}s, ending at the current row)")
    print(f"Horizon : {horizon} row(s) "
          f"({horizon * STEP_SECONDS:.4f}s ahead)\n")

    all_X, all_y, all_round, all_game, all_player = [], [], [], [], []

    with h5py.File(input_path, "r") as f:
        round_ids = f["data/round id"][:]
        game_ids = f["data/game id"][:]
        ticks = f["data/game tick number"][:]
        n_rows = len(round_ids)

        print(f"  rows         : {n_rows:,}")
        block_ids = build_blocks(round_ids, game_ids, ticks)
        n_blocks = block_ids[-1] + 1
        print(f"  blocks       : {n_blocks:,}")
        print(f"  games        : {len(np.unique(game_ids))}\n")

        for team in TEAMS:
            for slot in SLOTS:
                feats, alive = load_player(f, team, slot)
                X, y, at = windows(feats, alive, block_ids,
                                   history_len, horizon)
                if X is None:
                    print(f"  {team} {slot}: no usable rows, skipping")
                    continue

                all_X.append(X)
                all_y.append(y)
                all_round.append(round_ids[at])
                all_game.append(game_ids[at])
                all_player.append(np.full(len(at), f"{team}{slot}"))

                print(f"  {team} {slot}: {len(at):>7,} examples "
                      f"({len(at) / n_rows * 100:4.1f}% of rows usable)")

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    rounds = np.concatenate(all_round)
    games = np.concatenate(all_game)
    players = np.concatenate(all_player)

    print(f"\nTotal examples : {len(X):,}")
    print(f"X shape        : {X.shape}   (examples, history, "
          f"[px py pz vx vy vz])")
    print(f"y shape        : {y.shape}   (examples, [px py pz])")

    # -------------------------------------------------------------------
    # Self-checks. These exist so a wrong assumption crashes the run
    # instead of quietly producing a plausible-looking training set.
    # -------------------------------------------------------------------
    last_pos = X[:, -1, :3]
    moved = np.linalg.norm(y - last_pos, axis=1)
    implied_speed = moved / (horizon * STEP_SECONDS)

    print(f"\nDistance moved over {horizon} row(s) "
          f"({horizon * STEP_SECONDS:.4f}s), game units:")
    print(f"  median : {np.median(moved):7.2f}")
    print(f"  mean   : {moved.mean():7.2f}")
    print(f"  90th   : {np.percentile(moved, 90):7.2f}")
    print(f"  max    : {moved.max():7.2f}")
    print(f"  implied median speed : {np.median(implied_speed):.1f} units/sec")

    # A single teleport is proof the block logic failed, even though it is a
    # tiny fraction of the data. Tested by deliberately removing the block
    # guard: max speed hit 5,840 u/s while only 0.4% of examples were
    # affected, so a percentage threshold alone would have let it through.
    worst = implied_speed.max()
    if worst > TELEPORT_SPEED:
        raise SystemExit(
            f"ABORT: an example implies {worst:.0f} u/s. That is a teleport, "
            f"not movement. Some example spans a block boundary."
        )

    over = implied_speed > MAX_PLAUSIBLE_SPEED
    if over.mean() > 0.01:
        raise SystemExit(
            f"ABORT: {over.mean()*100:.2f}% of examples imply speeds over "
            f"{MAX_PLAUSIBLE_SPEED} u/s. Examples are probably spanning a "
            f"boundary the block logic missed."
        )
    if over.any():
        print(f"  ({over.sum():,} examples over {MAX_PLAUSIBLE_SPEED} u/s "
              f"-- falls and boosts, under the 1% abort threshold)")

    # Independent check: step distance measured INSIDE X, which never touches
    # the target. Should be close to the target distance at horizon 1.
    if history_len >= 2:
        inner = np.linalg.norm(np.diff(X[:, :, :3], axis=1), axis=2)
        print(f"\n  cross-check, median distance between consecutive frames"
              f" inside X: {np.median(inner):.2f}")
        print("  (measured 6.1 units per 62.5ms row on 30 Aug; these should"
              " agree)")

    np.savez_compressed(
        output_path,
        X=X, y=y, rounds=rounds, games=games, players=players,
        history_len=history_len, horizon=horizon,
        step_seconds=STEP_SECONDS,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
    print("\nLoad it later with:")
    print("    d = np.load('parsed_data.npz')")
    print("    X, y, games = d['X'], d['y'], d['games']")
    print("\nSplit train/test on 'games', not on 'rounds'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--history", type=int, default=HISTORY_LEN)
    ap.add_argument("--horizon", type=int, default=HORIZON)
    args = ap.parse_args()

    if args.history < 1:
        raise SystemExit("history must be at least 1")
    if args.horizon < 1:
        raise SystemExit("horizon must be at least 1")

    build(args.input, args.output, args.history, args.horizon)


if __name__ == "__main__":
    main()
