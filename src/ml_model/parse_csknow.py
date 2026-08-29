"""
Parse the CSKnow feature store into training-ready arrays.

WHAT THIS DOES
    The HDF5 file stores every value as its own array, with names like
    "data/player pos CT 0 t-17 y". That is unusable for PyTorch. This script
    gathers those scattered arrays into two solid blocks of numbers:

        X : (n_examples, history_len, 6)   the recent past
        y : (n_examples, 3)                where the player actually went

    The 6 features per timestep are position (x, y, z) and velocity (x, y, z).

WHAT ONE EXAMPLE LOOKS LIKE
    Take one player at one moment. Look back HISTORY_LEN steps (each step is
    62.5ms). That stack of positions and velocities is the input. The player's
    position HORIZON steps later is the answer the model has to predict.

    HORIZON = 1 means "predict 62.5ms ahead" -- roughly one lost packet.
    HORIZON = 3 means "predict 187ms ahead"  -- a short burst of loss.

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

DEFAULT_INPUT = r"C:\Users\Atharva\Downloads\sample_csknow\all_train_outputs\behaviorTreeTeamFeatureStore_28.hdf5"
DEFAULT_OUTPUT = "parsed_data.npz"

HISTORY_LEN = 16   # how many past steps to feed the model (max 48 in this data)
HORIZON = 1        # how many steps ahead to predict

TEAMS = ("CT", "T")
SLOTS = (0, 1, 2, 3, 4)

STEP_SECONDS = 0.0625   # 8 game ticks at 128 Hz


# ---------------------------------------------------------------------------

def load_player(f, team, slot, history_len):
    """
    Pull one player's history and current state out of the file.

    Returns
        hist : (n_rows, history_len, 6)  oldest step first
        now  : (n_rows, 3)               current position
        ok   : (n_rows,) bool            rows where this player is alive
                                         and every history step is valid
    """
    p = f"data/player pos {team} {slot}"
    v = f"data/player velocity {team} {slot}"

    # Current position -- the "now" that history leads up to.
    now = np.stack([f[f"{p} {ax}"][:] for ax in "xyz"], axis=1)

    # History. In the file, t-1 is the most recent past step and t-48 the
    # oldest, so we walk backwards to get oldest-first ordering.
    steps = []
    for t in range(history_len, 0, -1):
        pos = np.stack([f[f"{p} t-{t} {ax}"][:] for ax in "xyz"], axis=1)
        vel = np.stack([f[f"{v} t-{t} {ax}"][:] for ax in "xyz"], axis=1)
        steps.append(np.concatenate([pos, vel], axis=1))   # (n_rows, 6)

    hist = np.stack(steps, axis=1)   # (n_rows, history_len, 6)

    # A row is only usable if the player is alive and the game actually
    # recorded every history step (early in a round it hasn't yet).
    ok = f[f"data/alive {team} {slot}"][:].astype(bool)
    for t in range(1, history_len + 1):
        ok &= f[f"data/player history valid {team} {slot} t-{t}"][:].astype(bool)

    return hist, now, ok


def build(input_path, output_path, history_len, horizon):
    print(f"Reading : {input_path}")
    print(f"History : {history_len} steps ({history_len * STEP_SECONDS:.3f}s)")
    print(f"Horizon : {horizon} step(s) ({horizon * STEP_SECONDS:.4f}s ahead)\n")

    all_X, all_y, all_round, all_player = [], [], [], []

    with h5py.File(input_path, "r") as f:
        round_ids = f["data/round id"][:]
        n_rows = len(round_ids)

        for team in TEAMS:
            for slot in SLOTS:
                hist, now, ok = load_player(f, team, slot, history_len)

                # The answer lives HORIZON rows further down the file. Shift
                # the current-position array back to line it up with history.
                target = np.roll(now, -horizon, axis=0)

                # Rolling wraps around the end of the file, and it also crosses
                # round boundaries -- both would pair a player's history with
                # a position from a completely different situation. Only keep
                # rows where the target is genuinely HORIZON steps later in the
                # same round.
                same_round = np.zeros(n_rows, dtype=bool)
                same_round[:-horizon] = round_ids[horizon:] == round_ids[:-horizon]

                # The player must also still be alive when we check the answer.
                alive_later = np.zeros(n_rows, dtype=bool)
                alive_later[:-horizon] = ok[horizon:]

                keep = ok & same_round & alive_later

                if keep.sum() == 0:
                    print(f"  {team} {slot}: no usable rows, skipping")
                    continue

                all_X.append(hist[keep])
                all_y.append(target[keep])
                all_round.append(round_ids[keep])
                all_player.append(np.full(keep.sum(), f"{team}{slot}"))

                print(f"  {team} {slot}: {keep.sum():>7,} examples "
                      f"({keep.mean() * 100:4.1f}% of rows usable)")

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    rounds = np.concatenate(all_round)
    players = np.concatenate(all_player)

    print(f"\nTotal examples : {len(X):,}")
    print(f"X shape        : {X.shape}   (examples, history, [px py pz vx vy vz])")
    print(f"y shape        : {y.shape}   (examples, [px py pz])")

    # How far does a player actually travel over the prediction horizon? This
    # is the scale your model's error should be judged against.
    last_pos = X[:, -1, :3]
    moved = np.linalg.norm(y - last_pos, axis=1)
    print(f"\nDistance moved over {horizon} step(s), in game units:")
    print(f"  median : {np.median(moved):7.1f}")
    print(f"  mean   : {moved.mean():7.1f}")
    print(f"  90th   : {np.percentile(moved, 90):7.1f}")
    print(f"  max    : {moved.max():7.1f}")
    print("\n  (A model whose error is much bigger than the median distance")
    print("   moved is not doing anything useful.)")

    np.savez_compressed(
        output_path,
        X=X, y=y, rounds=rounds, players=players,
        history_len=history_len, horizon=horizon,
        step_seconds=STEP_SECONDS,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\nSaved to {output_path} ({size_mb:.1f} MB)")
    print("\nLoad it later with:")
    print("    d = np.load('parsed_data.npz')")
    print("    X, y = d['X'], d['y']")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--history", type=int, default=HISTORY_LEN,
                    help="past steps to use as input (max 48)")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help="steps ahead to predict")
    args = ap.parse_args()

    if not 1 <= args.history <= 48:
        raise SystemExit("history must be between 1 and 48")
    if args.horizon < 1:
        raise SystemExit("horizon must be at least 1")

    build(args.input, args.output, args.history, args.horizon)


if __name__ == "__main__":
    main()
