"""
Number #2: how does position error grow across a burst of lost packets?

WHAT A BURST ACTUALLY IS
    The server stops hearing from a client at frame t. It still has to
    render a position at t+1, and again at t+2, and t+3, until packets
    resume. That is not one prediction -- it is a whole blackout that
    has to be filled, and the error at the far end is what a player
    actually sees.

    At 64Hz an update arrives every 15.6ms, so one row of this dataset
    (62.5ms) is about four consecutive drops. A burst of 12 rows is
    750ms, or roughly 48 dropped updates -- a bad but entirely real
    network stall.

WHY THIS IS DIFFERENT FROM THE HORIZON RUNS
    Those measured single-point accuracy at three separate horizons,
    each in isolation. This treats the same numbers as one curve: error
    as a function of how long the blackout has lasted. Same data, but
    the question is "how fast does the picture degrade" rather than
    "how good is one guess".

    Neither method sees a new frame during the blackout. Dead reckoning
    keeps extrapolating one stale velocity vector, so its straight line
    walks further from a player who has turned or stopped. The model
    gets one shot too, but from a second of history rather than one
    instant.

WHAT THIS IS NOT
    Not autoregressive rollout. A real client would feed its own guess
    back in as the next frame and predict again. This model emits a
    position, not a full frame with velocity, so its output cannot be
    fed back without inventing the missing fields. Doing that properly
    means a different model and a retrain. Say so in future work rather
    than implying this is a rollout.

    Each checkpoint is evaluated at the horizon it was TRAINED for, so
    every point is a model doing the job it was fitted to do. That is
    the fair comparison; evaluating a horizon-4 model at 750ms would
    measure a mismatch, not a method.

Usage:
    python burst_loss.py
    python burst_loss.py --checkpoints gru_h4_s0.pt gru_h8_s0.pt gru_h12_s0.pt
"""

import argparse
import os

import numpy as np
import torch

from baselines import game_split
from gru_model import GRUPredictor

DEFAULT_CHECKPOINTS = [
    "gru_h4_s0.pt", "gru_h4_s1.pt",
    "gru_h12_s0.pt", "gru_h12_s1.pt",
]
FRAME_BUDGET_MS = 1000.0 / 60.0


def evaluate(ckpt_path, npz_for_horizon, batch=8192):
    """
    Score one checkpoint against dead reckoning on its own held-out games.

    Returns a dict of measured numbers, or None if the matching parsed
    file is missing.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    horizon = int(ck["horizon"])
    step = float(ck["step_seconds"])
    dt = horizon * step

    npz = npz_for_horizon.get(horizon)
    if npz is None or not os.path.exists(npz):
        print(f"  {ckpt_path}: needs horizon-{horizon} data, not found "
              f"(expected {npz})")
        return None

    d = np.load(npz, allow_pickle=False)
    X, y, games = d["X"], d["y"], d["games"]
    if int(d["horizon"]) != horizon:
        raise SystemExit(
            f"{npz} is horizon {int(d['horizon'])} but {ckpt_path} was "
            f"trained at horizon {horizon}. Mismatched files."
        )

    # The seed decides the split, and it must be the SAME seed the model
    # was trained with, or the "held-out" games were seen in training.
    seed = int(ck.get("seed", 0))
    if "test_games" in ck:
        test_mask = np.isin(games, ck["test_games"])
    else:
        _, test_mask, _ = game_split(games, seed=seed)

    Xte = torch.from_numpy(X[test_mask])
    yte = torch.from_numpy(y[test_mask])

    model = GRUPredictor(hidden=ck["hidden"], layers=ck["layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch):
            preds.append(model(Xte[i:i + batch]))
    gru_pred = torch.cat(preds)

    pos, vel = Xte[:, -1, 0:3], Xte[:, -1, 3:6]
    dr_err = torch.linalg.norm(pos + vel * dt - yte, dim=1).numpy()
    null_err = torch.linalg.norm(pos - yte, dim=1).numpy()
    gru_err = torch.linalg.norm(gru_pred - yte, dim=1).numpy()

    return dict(
        ckpt=os.path.basename(ckpt_path), horizon=horizon, ms=dt * 1000,
        n=len(gru_err),
        null=np.median(null_err),
        dr=np.median(dr_err), dr90=np.percentile(dr_err, 90),
        dr99=np.percentile(dr_err, 99),
        gru=np.median(gru_err), gru90=np.percentile(gru_err, 90),
        gru99=np.percentile(gru_err, 99),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--plot", default="burst_loss.png")
    args = ap.parse_args()

    torch.set_num_threads(1)

    # Which parsed file holds which horizon. Adjust if you named them
    # differently -- the script verifies the match before using one.
    npz_for_horizon = {
        1: "parsed_h1.npz",
        4: "parsed_h4.npz",
        8: "parsed_h8.npz",
        12: "parsed_data.npz",
    }

    print("Burst-loss evaluation")
    print("Each checkpoint scored at the horizon it was trained for, on")
    print("its own held-out games.\n")

    rows = []
    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"  {path}: not found, skipping")
            continue
        r = evaluate(path, npz_for_horizon)
        if r:
            rows.append(r)
            print(f"  {r['ckpt']:<16} horizon {r['horizon']:>2} "
                  f"({r['ms']:>5.1f} ms)  {r['n']:>7,} test examples")

    if not rows:
        raise SystemExit("\nNo checkpoints could be evaluated.")

    rows.sort(key=lambda r: (r["horizon"], r["ckpt"]))

    # -------------------------------------------------------------------
    print("\nPosition error after a burst of lost updates, game units")
    print("(1 unit ~ 1 inch; a player hitbox is ~32 units wide)\n")
    print(f"  {'burst':>6} {'ms':>6} {'drops':>6} | {'DR med':>7} "
          f"{'GRU med':>8} {'gain':>6} | {'DR 90th':>8} {'GRU 90th':>9} "
          f"{'gain':>6} | {'checkpoint':<15}")
    print("  " + "-" * 92)

    for r in rows:
        drops = round(r["ms"] / (1000 / 64))      # updates lost at 64Hz
        med_gain = (1 - r["gru"] / r["dr"]) * 100
        p90_gain = (1 - r["gru90"] / r["dr90"]) * 100
        print(f"  {r['horizon']:>6} {r['ms']:>6.1f} {drops:>6} | "
              f"{r['dr']:>7.2f} {r['gru']:>8.2f} {med_gain:>5.1f}% | "
              f"{r['dr90']:>8.2f} {r['gru90']:>9.2f} {p90_gain:>5.1f}% | "
              f"{r['ckpt']:<15}")

    # -------------------------------------------------------------------
    # How fast does each method degrade? Fit error ~ burst^k. A larger
    # exponent means the picture falls apart faster as the stall runs on.
    # -------------------------------------------------------------------
    def power_fit(key):
        h = np.array([r["horizon"] for r in rows], float)
        v = np.array([r[key] for r in rows], float)
        good = np.isfinite(v) & (v > 0)
        if len(np.unique(h[good])) < 2:
            return np.nan
        return np.polyfit(np.log(h[good]), np.log(v[good]), 1)[0]

    print(f"\n  DR degrades as burst^{power_fit('dr'):.2f}, "
          f"GRU as burst^{power_fit('gru'):.2f} (median)")
    print(f"  DR degrades as burst^{power_fit('dr90'):.2f}, "
          f"GRU as burst^{power_fit('gru90'):.2f} (90th percentile)")
    print("  A smaller exponent means the picture degrades more slowly.")

    # -------------------------------------------------------------------
    # When does each method's error exceed a player's own width? Past
    # that point the server is rendering a player somewhere they are not.
    # -------------------------------------------------------------------
    print("\n  Burst length at which the 90th-percentile error exceeds")
    print("  a player's width (32 units):")
    for name, key in (("dead reckoning", "dr90"), ("GRU", "gru90")):
        crossed = [r for r in rows if r[key] > 32]
        if crossed:
            first = min(crossed, key=lambda r: r["horizon"])
            print(f"    {name:<16} by {first['ms']:.0f} ms "
                  f"({first[key]:.0f} units)")
        else:
            print(f"    {name:<16} not within the range tested")

    # -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_h = {}
        for r in rows:
            by_h.setdefault(r["horizon"], []).append(r)
        hs = sorted(by_h)
        ms = [by_h[h][0]["ms"] for h in hs]

        def mean_of(key):
            return [np.mean([r[key] for r in by_h[h]]) for h in hs]

        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

        a.plot(ms, mean_of("null"), "o--", color="grey", label="no prediction")
        a.plot(ms, mean_of("dr"), "o-", label="dead reckoning")
        a.plot(ms, mean_of("gru"), "o-", label="GRU")
        a.axhline(32, color="k", lw=.8, ls=":")
        a.text(ms[0], 33, "player width", fontsize=8)
        a.set_xlabel("burst duration (ms)")
        a.set_ylabel("median error (game units)")
        a.set_title("Median error vs burst length")
        a.legend(fontsize=8)
        a.grid(alpha=.3)

        b.plot(ms, mean_of("dr90"), "o-", label="dead reckoning")
        b.plot(ms, mean_of("gru90"), "o-", label="GRU")
        b.axhline(32, color="k", lw=.8, ls=":")
        b.text(ms[0], 34, "player width", fontsize=8)
        b.set_xlabel("burst duration (ms)")
        b.set_ylabel("90th percentile error (game units)")
        b.set_title("Worst-case error vs burst length")
        b.legend(fontsize=8)
        b.grid(alpha=.3)

        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\nPlot saved to {args.plot}")
    except ImportError:
        print("\n(matplotlib not installed, skipping plot)")

    print("\nNote: each method makes ONE prediction across the blackout,")
    print("not a step-by-step rollout. A real client would feed its own")
    print("guess back in and predict again; that needs a model emitting a")
    print("full frame, not just a position. Future work, not this.")


if __name__ == "__main__":
    main()
