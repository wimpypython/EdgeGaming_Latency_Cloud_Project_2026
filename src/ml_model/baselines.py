"""
The two baselines every model in this project must be measured against.

WHY BASELINES COME FIRST
    "Our model's error is 4.2 units" means nothing on its own. It only means
    something next to the error of a method that took no effort. There are
    two such methods, and they bracket the problem:

    NULL -- predict the player did not move at all.
        Its error IS the distance the player travelled. That is the floor.
        Anything that cannot beat this is worse than doing nothing.

    DEAD RECKONING -- predict the player kept going in a straight line.
        pos + velocity * dt. This is what real game netcode actually does
        when packets stop arriving, so it is the number that matters. A
        model that beats NULL but loses to DR has not improved on practice.

WHY dt IS THE DANGEROUS LINE
    Velocity is units per SECOND (measured 30 Aug, not assumed). The target
    sits HORIZON rows ahead, and a row is 62.5ms. So dt = horizon * 0.0625.

    Getting this wrong silently hands your model a fake win:
        no dt at all      -> DR extrapolates 16x too far, error explodes
        dt too small      -> DR undershoots, error inflates

    Neither would crash. Both would look like a great result. So dt is NOT
    hardcoded here -- it is read out of the npz the parser wrote, and cannot
    drift out of sync with the data it is applied to.

WHY THE SPLIT LIVES IN THIS FILE
    DR has no parameters, so it cannot overfit, and could in principle be
    scored on everything. But the final comparison table has to measure
    every method on the SAME held-out examples or it is not a comparison.
    game_split() is written once here and imported everywhere else.

    The split is by GAME, never randomly and never by round. Consecutive
    examples share 15 of their 16 history frames, so a random split puts
    near-duplicates on both sides and inflates the score. Rounds from one
    match share players, economy and site preferences, so a round-level
    split still leaks match identity.

Usage:
    python baselines.py
    python baselines.py --input parsed_data.npz --test-games 5
"""

import argparse

import numpy as np


# ---------------------------------------------------------------------------
# The split -- imported by every other script
# ---------------------------------------------------------------------------

def game_split(games, n_test=5, seed=0):
    """
    Split examples into train and test by whole game.

    Deterministic given the same seed, so every script that imports this
    gets the identical split without needing to write a file to disk.

    Returns (train_mask, test_mask, test_game_ids).
    """
    unique = np.unique(games)
    if n_test >= len(unique):
        raise SystemExit(
            f"n_test={n_test} but there are only {len(unique)} games"
        )

    rng = np.random.default_rng(seed)
    test_games = np.sort(rng.choice(unique, size=n_test, replace=False))

    test_mask = np.isin(games, test_games)
    train_mask = ~test_mask

    # A split that overlaps is worse than no split, because it looks fine.
    assert not np.intersect1d(games[train_mask], games[test_mask]).size, \
        "train and test share a game -- the split is broken"

    return train_mask, test_mask, test_games


# ---------------------------------------------------------------------------
# The baselines themselves
# ---------------------------------------------------------------------------

def predict_null(X):
    """Predict the player did not move: return the current position."""
    return X[:, -1, 0:3]


def predict_dead_reckoning(X, dt):
    """
    Predict the player kept going in a straight line.

    Uses ONE of the 16 history frames and throws the other 15 away. That is
    the point -- it is deliberately dumb. A sequence model gets all 16. If
    it cannot beat a method using 1/16th of the information, that is a real
    finding and it should be reported, not hidden.
    """
    pos = X[:, -1, 0:3]
    vel = X[:, -1, 3:6]
    return pos + vel * dt


def errors(pred, y):
    """Euclidean distance between prediction and truth, per example."""
    return np.linalg.norm(pred - y, axis=1)


def describe(name, err):
    print(f"  {name:<18} {np.median(err):8.3f} {err.mean():8.3f} "
          f"{np.percentile(err, 90):8.3f} {np.percentile(err, 99):8.3f} "
          f"{err.max():9.2f}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="parsed_data.npz")
    ap.add_argument("--test-games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=False)
    X, y, games = d["X"], d["y"], d["games"]
    horizon = int(d["horizon"])
    step_seconds = float(d["step_seconds"])

    # Derived, never hardcoded. Regenerate the npz at a different horizon
    # and this follows automatically.
    dt = horizon * step_seconds

    print(f"Loaded  : {args.input}")
    print(f"X       : {X.shape}")
    print(f"horizon : {horizon} row(s)")
    print(f"dt      : {horizon} x {step_seconds} = {dt:.4f} s\n")

    train_mask, test_mask, test_games = game_split(
        games, n_test=args.test_games, seed=args.seed)

    print(f"Split by game (seed {args.seed}):")
    print(f"  train : {train_mask.sum():>8,} examples  "
          f"{len(np.unique(games[train_mask])):>3} games")
    print(f"  test  : {test_mask.sum():>8,} examples  "
          f"{len(np.unique(games[test_mask])):>3} games  "
          f"({test_mask.mean()*100:.1f}%)")
    print(f"  held-out game ids: {test_games.tolist()}")

    Xt, yt = X[test_mask], y[test_mask]

    null_err = errors(predict_null(Xt), yt)
    dr_err = errors(predict_dead_reckoning(Xt, dt), yt)

    print(f"\nError on held-out test set, game units "
          f"({dt*1000:.1f}ms ahead):")
    print(f"  {'method':<18} {'median':>8} {'mean':>8} {'90th':>8} "
          f"{'99th':>8} {'max':>9}")
    describe("null (no move)", null_err)
    describe("dead reckoning", dr_err)

    improvement = (1 - np.median(dr_err) / np.median(null_err)) * 100
    print(f"\n  DR is {improvement:.1f}% better than doing nothing (median)")

    # -------------------------------------------------------------------
    # Where does dead reckoning actually fail? This is the whole argument
    # for using a sequence model, so measure it rather than assert it.
    #
    # Compare the heading at the current frame against the heading one
    # frame earlier. A player travelling straight has a small angle; one
    # turning has a large one. DR only knows the current velocity vector,
    # so it should degrade sharply as the turn angle grows.
    # -------------------------------------------------------------------
    v_now = Xt[:, -1, 3:5]      # horizontal velocity only; z is falls/steps
    v_prev = Xt[:, -2, 3:5]
    s_now = np.linalg.norm(v_now, axis=1)
    s_prev = np.linalg.norm(v_prev, axis=1)

    moving = (s_now > 50) & (s_prev > 50)
    cos = np.ones(len(Xt))
    cos[moving] = np.clip(
        (v_now[moving] * v_prev[moving]).sum(axis=1)
        / (s_now[moving] * s_prev[moving]), -1, 1)
    turn_deg = np.degrees(np.arccos(cos))

    print("\nDead reckoning error by how sharply the player is turning:")
    print(f"  {'':<18} {'median':>8} {'mean':>8} {'90th':>8} "
          f"{'99th':>8} {'max':>9}   n")
    bands = [("stationary", ~moving, None),
             ("straight <5deg", moving & (turn_deg < 5), None),
             ("gentle 5-20deg", moving & (turn_deg >= 5) & (turn_deg < 20), None),
             ("sharp >=20deg", moving & (turn_deg >= 20), None)]
    for label, mask, _ in bands:
        if mask.sum() < 50:
            print(f"  {label:<18} (too few examples: {mask.sum()})")
            continue
        e = dr_err[mask]
        print(f"  {label:<18} {np.median(e):8.3f} {e.mean():8.3f} "
              f"{np.percentile(e, 90):8.3f} {np.percentile(e, 99):8.3f} "
              f"{e.max():9.2f}   {mask.sum():,}")

    # -------------------------------------------------------------------
    # Sanity checks. These catch a wrong dt, which would otherwise look
    # like a great result rather than a bug.
    # -------------------------------------------------------------------
    print()
    if np.median(dr_err) >= np.median(null_err):
        print("  WARNING: dead reckoning is no better than predicting no")
        print("  movement at all. Over a short horizon that should not")
        print("  happen. Suspect dt or the velocity columns before")
        print("  believing this.")
    if np.median(dr_err) < 0.05:
        print("  WARNING: dead reckoning error is suspiciously close to")
        print("  zero. Check that velocity is not derived from the target.")

    print("\nThis is the bar. A model must beat the dead reckoning row,")
    print("not the null row, to have improved on current practice.")


if __name__ == "__main__":
    main()
