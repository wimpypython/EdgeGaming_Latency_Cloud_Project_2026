"""
How far ahead does dead reckoning stop working?

WHY THIS RUNS BEFORE ANY MODEL
    At horizon 1 (62.5ms) dead reckoning's median error is 0.70 game units.
    A Source-engine unit is about one inch and a player hitbox is ~32 units
    wide, so DR is already accurate to under two centimetres. There is
    nothing there for a model to improve.

    The interesting region, if it exists, is further out. DR's error on a
    STRAIGHT run should grow roughly linearly with the horizon -- it is just
    an error in assumed speed, accumulating. DR's error on a TURN should
    grow faster, because lateral deviation from a curved path compounds. So
    the gap between straight and turning movement should widen with horizon.

    This script measures whether it does.

EITHER OUTCOME IS A RESULT
    If DR's error becomes large somewhere in this range, that horizon is
    where the project lives and it tells you what to train for.

    If DR stays small all the way out, the honest finding is that linear
    extrapolation is sufficient at these timescales. Report that rather than
    training a model to lose to it.

THE MISTAKE THIS AVOIDS
    The obvious approach re-windows the data at each horizon. But horizon 12
    drops examples near the end of a block that horizon 1 keeps, so each
    point on the curve would be measured on a different population and the
    trend would partly reflect that, not the horizon.

    So the validity mask is built ONCE at the largest horizon and reused for
    every horizon. Identical examples throughout; the only thing changing
    along the x-axis is how far ahead we predict.

    Block logic and the split are imported rather than reimplemented, so
    this cannot silently disagree with the parser or the baselines.

Usage:
    python horizon_sweep.py
    python horizon_sweep.py --horizons 1 2 4 8 16
"""

import argparse

import h5py
import numpy as np

from baselines import game_split
from parse_csknow import (DEFAULT_INPUT, HISTORY_LEN, SLOTS, STEP_SECONDS,
                          TEAMS, build_blocks, load_player)

DEFAULT_HORIZONS = [1, 2, 3, 4, 6, 8, 12]


def collect(input_path, history_len, horizons):
    """
    Gather what dead reckoning needs, for every horizon at once.

    DR uses a single frame, so there is no need to build the full
    (n, 16, 6) tensor here -- only the current state, the previous velocity
    (for the turn angle), and one target per horizon.
    """
    max_h = max(horizons)
    pos_a, vel_a, prev_a, game_a = [], [], [], []
    tgt_a = {h: [] for h in horizons}

    with h5py.File(input_path, "r") as f:
        round_ids = f["data/round id"][:]
        game_ids = f["data/game id"][:]
        ticks = f["data/game tick number"][:]
        n = len(round_ids)

        print(f"  rows   : {n:,}")
        block_ids = build_blocks(round_ids, game_ids, ticks)
        print(f"  blocks : {block_ids[-1] + 1:,}")
        print(f"  games  : {len(np.unique(game_ids))}\n")

        rows = np.arange(n)
        offsets = np.arange(-history_len + 1, 1)
        hist_idx = rows[:, None] + offsets[None, :]
        safe_hist = np.clip(hist_idx, 0, n - 1)

        # Built at the LARGEST horizon, then reused for all of them.
        far_idx = np.clip(rows + max_h, 0, n - 1)
        in_range = (hist_idx[:, 0] >= 0) & (rows + max_h < n)
        same_block = (
            (block_ids[safe_hist] == block_ids[:, None]).all(axis=1)
            & (block_ids[far_idx] == block_ids)
        )

        for team in TEAMS:
            for slot in SLOTS:
                feats, alive = load_player(f, team, slot)
                alive_all = alive[safe_hist].all(axis=1) & alive[far_idx]
                keep = in_range & same_block & alive_all
                if not keep.any():
                    print(f"  {team} {slot}: no usable rows")
                    continue

                sel = rows[keep]
                pos_a.append(feats[sel, 0:3])
                vel_a.append(feats[sel, 3:6])
                prev_a.append(feats[sel - 1, 3:5])
                game_a.append(game_ids[sel])
                for h in horizons:
                    tgt_a[h].append(feats[sel + h, 0:3])

                print(f"  {team} {slot}: {keep.sum():>7,} examples")

    return dict(
        pos=np.concatenate(pos_a),
        vel=np.concatenate(vel_a),
        prev=np.concatenate(prev_a),
        games=np.concatenate(game_a),
        targets={h: np.concatenate(tgt_a[h]) for h in horizons},
    )


def turn_bands(vel, prev):
    """Split examples by heading change between the last two frames."""
    v = vel[:, :2]                    # horizontal only; z is steps and falls
    s_now = np.linalg.norm(v, axis=1)
    s_prev = np.linalg.norm(prev, axis=1)

    moving = (s_now > 50) & (s_prev > 50)
    cos = np.ones(len(v))
    cos[moving] = np.clip(
        (v[moving] * prev[moving]).sum(axis=1) / (s_now[moving] * s_prev[moving]),
        -1, 1)
    deg = np.degrees(np.arccos(cos))

    return {
        "straight": moving & (deg < 5),
        "gentle": moving & (deg >= 5) & (deg < 20),
        "sharp": moving & (deg >= 20),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--history", type=int, default=HISTORY_LEN)
    ap.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    ap.add_argument("--test-games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default="horizon_sweep.png")
    args = ap.parse_args()

    horizons = sorted(set(args.horizons))
    print(f"Reading  : {args.input}")
    print(f"History  : {args.history} rows")
    print(f"Horizons : {horizons}  "
          f"({min(horizons)*STEP_SECONDS*1000:.1f}ms to "
          f"{max(horizons)*STEP_SECONDS*1000:.1f}ms)\n")

    d = collect(args.input, args.history, horizons)

    _, test_mask, test_games = game_split(
        d["games"], n_test=args.test_games, seed=args.seed)
    print(f"\nHeld-out games : {test_games.tolist()}")
    print(f"Test examples  : {test_mask.sum():,} "
          f"(identical set at every horizon)")

    pos, vel = d["pos"][test_mask], d["vel"][test_mask]
    bands = turn_bands(vel, d["prev"][test_mask])
    print(f"  straight {bands['straight'].sum():,}   "
          f"gentle {bands['gentle'].sum():,}   "
          f"sharp {bands['sharp'].sum():,}")

    print("\nDead reckoning error, held-out games, game units")
    print("(1 unit ~ 1 inch; a player hitbox is ~32 units wide)\n")
    print(f"  {'horizon':>7} {'ms':>7} {'null':>8} {'DR med':>8} {'DR 90th':>8} "
          f"{'DR 99th':>8} | {'straight':>8} {'gentle':>7} {'sharp':>7} "
          f"{'ratio':>7}")
    print("  " + "-" * 88)

    out = []
    for h in horizons:
        y = d["targets"][h][test_mask]
        dt = h * STEP_SECONDS

        null_err = np.linalg.norm(pos - y, axis=1)
        dr_err = np.linalg.norm(pos + vel * dt - y, axis=1)

        med = {k: (np.median(dr_err[m]) if m.sum() > 50 else np.nan)
               for k, m in bands.items()}
        ratio = med["sharp"] / med["straight"] if med["straight"] else np.nan

        print(f"  {h:>7} {dt*1000:>7.1f} {np.median(null_err):>8.2f} "
              f"{np.median(dr_err):>8.3f} {np.percentile(dr_err, 90):>8.2f} "
              f"{np.percentile(dr_err, 99):>8.2f} | {med['straight']:>8.3f} "
              f"{med['gentle']:>7.3f} {med['sharp']:>7.3f} {ratio:>6.2f}x")

        out.append(dict(h=h, ms=dt*1000, null=np.median(null_err),
                        dr=np.median(dr_err),
                        dr90=np.percentile(dr_err, 90),
                        dr99=np.percentile(dr_err, 99),
                        ratio=ratio, **med))

    # -----------------------------------------------------------------
    # Fit error ~ horizon^k. k near 1 is linear accumulation; k above 1
    # means the error compounds, which is where history should help.
    # -----------------------------------------------------------------
    def power_fit(key):
        """Slope of log(error) vs log(horizon), skipping unusable points."""
        v = np.array([r[key] for r in out], float)
        good = np.isfinite(v) & (v > 0)
        if good.sum() < 2:
            return np.nan
        return np.polyfit(np.log(hs[good]), np.log(v[good]), 1)[0]

    hs = np.array([r["h"] for r in out], float)
    k = power_fit("dr")
    ks = power_fit("sharp")
    kt = power_fit("straight")

    print(f"\n  DR error grows as horizon^{k:.2f}"
          f"   (straight h^{kt:.2f}, turning h^{ks:.2f})")
    if k > 1.15:
        print("  -> super-linear: error compounds. This is the regime where")
        print("     a sequence model has something to contribute.")
    elif k < 0.9:
        print("  -> saturating: error flattens. Movement is constrained.")
    else:
        print("  -> roughly linear: players travel close to straight lines,")
        print("     so a sequence model has limited room to improve on DR.")
        print("     Saying so plainly is a legitimate finding.")

    print("\n  ratio = turning error / straight error. If it grows with")
    print("  horizon, turns are where history pays off. If it stays flat,")
    print("  they are not, and the project needs a different angle.")

    # -----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ms = [r["ms"] for r in out]
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))

        a.plot(ms, [r["null"] for r in out], "o--", label="null (no move)")
        a.plot(ms, [r["dr"] for r in out], "o-", label="dead reckoning")
        a.plot(ms, [r["dr90"] for r in out], "s:", label="DR 90th pct")
        a.axhline(32, color="grey", lw=.8)
        a.text(ms[0], 33, "player width (32 u)", fontsize=8, color="grey")
        a.set_xlabel("prediction horizon (ms)")
        a.set_ylabel("position error (game units)")
        a.set_title(f"Error vs horizon   (DR ~ h^{k:.2f})")
        a.legend(fontsize=8)
        a.grid(alpha=.3)

        for name, style in (("straight", "o-"), ("gentle", "o-"),
                            ("sharp", "o-")):
            b.plot(ms, [r[name] for r in out], style, label=name)
        b.set_xlabel("prediction horizon (ms)")
        b.set_ylabel("DR median error (game units)")
        b.set_title("Dead reckoning by turn sharpness")
        b.legend(fontsize=8)
        b.grid(alpha=.3)

        fig.tight_layout()
        fig.savefig(args.plot, dpi=140)
        print(f"\nPlot saved to {args.plot}")
    except ImportError:
        print("\n(matplotlib not installed, skipping plot)")


if __name__ == "__main__":
    main()
