"""
Number #3: how long does a prediction take, on one CPU core?

WHY THIS MATTERS MORE THAN IT SOUNDS
    The argument for putting a model at the edge is not that it is more
    accurate -- dead reckoning is only ~8% behind on the median. The
    argument is that the accuracy is affordable: the prediction fits
    inside a frame, so you can have it for free.

    That is a claim about time, and it needs a measurement.

WHY DEAD RECKONING IS TIMED TOO
    A number with nothing to compare it against says nothing. DR is one
    multiply-add, so it will be microseconds. The point of measuring it
    is to state the cost precisely: "the model costs Nx more than linear
    extrapolation, and both fit the budget."

WHY PERCENTILES, NOT AVERAGES
    A game renders 60 frames a second. What matters is not the average
    frame but the worst one -- a single 20ms hitch is visible, a hundred
    fast frames are not. So this reports the 99th percentile and the max
    alongside the median.

WHY ONE THREAD
    torch.set_num_threads(1). A game server runs many things at once; a
    model that only fits the budget by consuming every core has not
    really fit it. It also makes the number comparable to the CSKnow
    paper's figure, which is quoted for a single core.

WHAT THIS MEASUREMENT IS NOT
    This is a laptop CPU running Windows with whatever else is open.
    It is an honest measurement of this machine, not a deployment
    benchmark on server hardware. Say that in the report rather than
    letting the number stand unqualified.

Usage:
    python latency.py --checkpoint gru_h12_s0.pt
    python latency.py --checkpoint gru_h12_s0.pt --reps 500
"""

import argparse
import platform
import time

import numpy as np
import torch

from gru_model import GRUPredictor

BATCHES = (1, 10, 64)
FRAME_BUDGET_MS = 1000.0 / 60.0     # 16.67 ms at 60 fps


def time_calls(fn, x, reps, warmup):
    """Return per-call times in ms. Each call timed separately, not averaged."""
    for _ in range(warmup):
        fn(x)

    times = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        fn(x)
        times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def describe(label, t, batch):
    print(f"  {label:<18} {np.median(t):8.3f} {np.percentile(t, 90):8.3f} "
          f"{np.percentile(t, 99):8.3f} {t.max():8.3f} "
          f"{np.median(t) / batch:10.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="parsed_data.npz")
    ap.add_argument("--checkpoint", default="gru_h12_s0.pt")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    torch.set_num_threads(1)

    print(f"Machine : {platform.processor() or platform.machine()}")
    print(f"          {platform.system()} {platform.release()}, "
          f"torch {torch.__version__}, 1 thread")
    print("          (a laptop CPU, not server hardware -- see the note at")
    print("           the bottom before quoting these numbers)\n")

    d = np.load(args.input, allow_pickle=False)
    X = torch.from_numpy(d["X"][:max(BATCHES)])
    horizon = int(d["horizon"])
    step = float(d["step_seconds"])
    dt = horizon * step

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GRUPredictor(hidden=ck["hidden"], layers=ck["layers"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint : {args.checkpoint}")
    print(f"  GRU hidden={ck['hidden']} layers={ck['layers']} "
          f"({n_params:,} parameters)")
    print(f"  trained at horizon {ck['horizon']} "
          f"({ck['horizon'] * ck['step_seconds'] * 1000:.0f} ms), "
          f"val median {ck['val_median']:.2f} vs DR {ck['dr_median']:.2f}")
    print(f"\nTiming {args.reps} calls each, {args.warmup} warmup, "
          f"single thread.\n")

    def gru_call(x):
        with torch.no_grad():
            return model(x)

    def dr_call(x):
        # Deliberately the same arithmetic baselines.py uses, so the
        # comparison is like-for-like rather than a hand-tuned version.
        return x[:, -1, 0:3] + x[:, -1, 3:6] * dt

    results = {}
    for b in BATCHES:
        xb = X[:b]
        print(f"batch {b}"
              f"{'  (one player)' if b == 1 else ''}"
              f"{'  (one game step, all players)' if b == 10 else ''}")
        print(f"  {'method':<18} {'median':>8} {'90th':>8} {'99th':>8} "
              f"{'max':>8} {'per player':>10}   ms")

        t_dr = time_calls(dr_call, xb, args.reps, args.warmup)
        t_gru = time_calls(gru_call, xb, args.reps, args.warmup)
        describe("dead reckoning", t_dr, b)
        describe("GRU", t_gru, b)

        ratio = np.median(t_gru) / np.median(t_dr)
        pct = np.median(t_gru) / FRAME_BUDGET_MS * 100
        print(f"  GRU costs {ratio:.0f}x dead reckoning, and uses "
              f"{pct:.1f}% of the {FRAME_BUDGET_MS:.2f} ms frame budget\n")

        results[b] = (t_dr, t_gru)

    # -------------------------------------------------------------------
    # The headline. Batch 10 is one game step for a full team, which is
    # how the CSKnow paper reports its own figure.
    # -------------------------------------------------------------------
    t_dr, t_gru = results[10]
    worst = t_gru.max()

    print("=" * 64)
    print("Headline (10 players, one game step, single CPU thread)")
    print(f"  dead reckoning : {np.median(t_dr):.4f} ms median")
    print(f"  GRU            : {np.median(t_gru):.4f} ms median, "
          f"{np.percentile(t_gru, 99):.4f} ms at the 99th")
    print(f"  frame budget   : {FRAME_BUDGET_MS:.2f} ms at 60 fps")
    print(f"  headroom       : {FRAME_BUDGET_MS / np.median(t_gru):.0f}x "
          f"on the median, {FRAME_BUDGET_MS / worst:.0f}x on the worst call")

    if worst > FRAME_BUDGET_MS:
        print("\n  WARNING: at least one call exceeded the frame budget.")
        print("  A game would drop that frame. Report the tail, not just")
        print("  the median.")
    else:
        print(f"\n  Every one of {args.reps} calls fit the budget.")

    print("\nNotes for the report:")
    print("  - Measured on a laptop CPU under a desktop OS, single thread.")
    print("    Honest for this machine; not a server benchmark.")
    print("  - Per-player cost falls sharply with batch size, because most")
    print("    of the time is fixed per-call overhead. If integration calls")
    print("    the model once per player instead of once per team, the cost")
    print("    is roughly 10x higher. Batch the team.")
    print("  - CSKnow report under 0.5 ms per game step on one core, on")
    print("    their hardware. Different model and different machine, so")
    print("    treat it as context rather than a like-for-like comparison.")


if __name__ == "__main__":
    main()
