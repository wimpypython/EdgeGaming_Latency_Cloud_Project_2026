"""
Is the GRU doing anything a scalar couldn't?

THE OBSERVATION THAT PROMPTED THIS
    Across five checkpoints and three burst lengths, the ratio of GRU
    error to dead-reckoning error is almost perfectly constant:

        median : 0.906 +/- 0.020
        90th   : 0.768 +/- 0.026

    A constant ratio is exactly why both methods fit the same power law.
    c * h^k scaled by a constant is still h^k. So the identical exponents
    are not a coincidence -- they are what a constant ratio means.

THE DEFLATIONARY HYPOTHESIS
    Dead reckoning takes the instantaneous velocity at the last frame and
    assumes it holds for the whole blackout. But players turn, stop and
    double back, so straight-line displacement per unit time FALLS as the
    window grows -- measured at 98, 93.7, 83.7, 77.1 u/s for horizons
    1, 4, 8, 12. DR therefore overshoots, and overshoots more as the
    burst runs longer.

    A model could remove most of that with two cheap tricks: average the
    velocity over the window instead of trusting one noisy instant, and
    shrink the extrapolation by a learned factor. Both are scalar
    corrections to the same straight line, and scaling a straight line
    preserves the power law exactly.

    If that is all the GRU is doing, a single fitted number should
    reproduce most of its advantage.

WHAT EACH OUTCOME MEANS
    If shrunk DR captures most of the 23%: the model is close to trivial
    here. That is an uncomfortable result and an important one. Report it
    -- it is the first thing a viva examiner should be able to ask about
    and get a straight answer.

    If shrunk DR captures only a fraction: the GRU is doing something
    genuinely non-linear, and the obvious deflationary explanation has
    been ruled out rather than ignored.

    Either way the write-up is stronger for having asked.

HOW alpha IS FITTED
    On the TRAINING games only, never the held-out ones -- fitting a
    parameter on the test set is the same mistake as a random split, just
    smaller. One scalar, chosen to minimise mean squared error, which has
    a closed form.

Usage:
    python shrunk_dr.py --checkpoint gru_h8_s0.pt --input parsed_h8.npz
"""

import argparse
import os

import numpy as np
import torch

from baselines import game_split
from gru_model import GRUPredictor


def fit_alpha(pos, vel, y, dt):
    """
    Least-squares scalar for  pred = pos + vel * dt * alpha.

    Writing d = y - pos (where the player actually went) and
    v = vel * dt (where DR says they went), the error is |d - alpha*v|.
    Minimising the sum of squares gives alpha = <d,v> / <v,v>.
    """
    d = (y - pos).reshape(-1)
    v = (vel * dt).reshape(-1)
    return float(np.dot(d, v) / np.dot(v, v))


def stats(err):
    return (np.median(err), np.percentile(err, 90), np.percentile(err, 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    d = np.load(args.input, allow_pickle=False)
    X, y, games = d["X"], d["y"], d["games"]

    horizon = int(ck["horizon"])
    if int(d["horizon"]) != horizon:
        raise SystemExit(
            f"{args.input} is horizon {int(d['horizon'])} but the checkpoint "
            f"was trained at {horizon}. Mismatched files.")
    dt = horizon * float(ck["step_seconds"])

    # Same held-out games the model was scored on.
    test_mask = np.isin(games, ck["test_games"])
    train_mask = ~test_mask

    print(f"Checkpoint : {os.path.basename(args.checkpoint)}")
    print(f"Horizon    : {horizon} rows ({dt*1000:.0f} ms)")
    print(f"Train      : {train_mask.sum():,} examples")
    print(f"Test       : {test_mask.sum():,} examples "
          f"(games {ck['test_games'].tolist()})\n")

    # ---- fit alpha on TRAINING games only -----------------------------
    ptr, vtr, ytr = X[train_mask][:, -1, 0:3], X[train_mask][:, -1, 3:6], y[train_mask]
    alpha = fit_alpha(ptr, vtr, ytr, dt)
    print(f"Fitted shrinkage on training games: alpha = {alpha:.4f}")
    print(f"  (alpha = 1 would be plain dead reckoning; below 1 means DR")
    print(f"   systematically overshoots at this horizon)\n")

    # ---- also fit a version using the window-averaged velocity ---------
    # If averaging the noisy instantaneous velocity is most of what the
    # GRU gains, this second control will show it.
    vbar_tr = X[train_mask][:, :, 3:6].mean(axis=1)
    alpha_bar = fit_alpha(ptr, vbar_tr, ytr, dt)
    print(f"Same, using velocity averaged over all 16 frames: "
          f"alpha = {alpha_bar:.4f}\n")

    # ---- evaluate everything on the held-out games ---------------------
    Xte, yte = X[test_mask], y[test_mask]
    pos, vel = Xte[:, -1, 0:3], Xte[:, -1, 3:6]
    vbar = Xte[:, :, 3:6].mean(axis=1)

    preds = {
        "null (no move)": pos,
        "dead reckoning": pos + vel * dt,
        "shrunk DR": pos + vel * dt * alpha,
        "shrunk DR, avg vel": pos + vbar * dt * alpha_bar,
    }

    model = GRUPredictor(hidden=ck["hidden"], layers=ck["layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 8192):
            out.append(model(torch.from_numpy(Xte[i:i + 8192])))
    preds["GRU"] = torch.cat(out).numpy()

    errs = {k: np.linalg.norm(v - yte, axis=1) for k, v in preds.items()}

    print(f"  {'method':<20} {'median':>8} {'90th':>8} {'99th':>8}")
    print("  " + "-" * 46)
    for k, e in errs.items():
        m, p90, p99 = stats(e)
        print(f"  {k:<20} {m:>8.3f} {p90:>8.2f} {p99:>8.2f}")

    # ---- how much of the GRU's advantage did one scalar capture? -------
    dr_m, dr_90, _ = stats(errs["dead reckoning"])
    gru_m, gru_90, _ = stats(errs["GRU"])
    sh_m, sh_90, _ = stats(errs["shrunk DR"])
    av_m, av_90, _ = stats(errs["shrunk DR, avg vel"])

    print(f"\n  Improvement over plain dead reckoning:")
    print(f"  {'':<20} {'median':>8} {'90th':>8}")
    for label, m, p in (("shrunk DR", sh_m, sh_90),
                        ("shrunk DR, avg vel", av_m, av_90),
                        ("GRU", gru_m, gru_90)):
        print(f"  {label:<20} {(1-m/dr_m)*100:>7.1f}% {(1-p/dr_90)*100:>7.1f}%")

    gap = dr_90 - gru_90
    best_simple = min(sh_90, av_90)

    if gap <= 0:
        print(f"\n  The GRU does not beat dead reckoning at the 90th")
        print(f"  percentile here, so there is no advantage to attribute.")
        print(f"  Check the checkpoint matches the data before reading")
        print(f"  anything into the scalar controls.")
    else:
        def share(simple):
            return (dr_90 - simple) / gap * 100

        print(f"\n  Share of the GRU's 90th-percentile advantage that a scalar")
        print(f"  correction already accounts for:")
        print(f"    one scalar on instantaneous velocity : {share(sh_90):.0f}%")
        print(f"    one scalar on window-averaged velocity: {share(av_90):.0f}%")

        frac = share(best_simple)
        print()
        if frac >= 70:
            print("  Most of the advantage is reproducible with a single")
            print("  fitted number. The GRU is doing little a corrected")
            print("  straight line could not. Say so in the report -- it is")
            print("  the first thing an examiner should be able to ask about")
            print("  and get a straight answer to.")
        elif frac >= 30:
            print("  A scalar accounts for a substantial part of the")
            print("  advantage, but not all of it. Report both numbers: the")
            print("  trivial correction, and the residual the model adds on")
            print("  top of it.")
        else:
            print("  A scalar accounts for little of the advantage. The GRU")
            print("  is doing something genuinely non-linear, and the obvious")
            print("  deflationary explanation has been tested and ruled out")
            print("  rather than ignored.")

    print("\n  alpha was fitted on training games only. Fitting it on the")
    print("  held-out games would be the same mistake as a random split.")


if __name__ == "__main__":
    main()
