"""
Why did the GRU lose to dead reckoning on live traffic?

THE QUESTION
    On the CSKnow held-out games the GRU beats dead reckoning by 8-10%.
    On the synthetic client it LOST by 40%. Exactly one of two things is
    true, and they have opposite fixes:

      A. The traffic generator is too smooth. Real players stop and cut;
         the generator mostly runs in straight lines. A model that has
         learned to expect deceleration will overshoot on motion that
         never decelerates, and dead reckoning -- which assumes nothing
         changes -- is near-optimal on a straight line.

      B. The live hook feeds the model something wrong. Then the dataset
         numbers describe a different computation than the live ones and
         neither can be trusted.

WHAT THIS MEASURES
    The model is given histories whose answer is known by construction:

      straight   constant velocity, no turning        -- what the client does
      turning    constant speed on a circular arc     -- what a cut looks like
      stopping   decelerating to rest                 -- what a stop looks like

    For each, the model's predicted travel distance is compared with
    dead reckoning's. The RATIO is the diagnostic:

      ratio ~ 1.0 on straight   -> the model agrees with DR; case B
                                   (something else is wrong)
      ratio < 1.0 on straight   -> the model expects slowing that the
                                   generator never does; case A
      direction wrong           -> the hook is broken outright

    No ground truth is needed. The question is not whether the model is
    right, it is whether it is being asked the question it was trained on.

Usage:
    python diag_gru_bias.py --dir ..\\ml_model
"""

import argparse
import math

import gru_conceal


def straight(speed):
    """Constant velocity along +x. What the client mostly produces."""
    rows = []
    for i in range(gru_conceal.HISTORY_LEN):
        t = (i - (gru_conceal.HISTORY_LEN - 1)) * gru_conceal.SOURCE_STEP
        rows.append([speed * t, 0.0, 0.0, speed, 0.0, 0.0])
    return rows, (speed, 0.0, 0.0)


def turning(speed, turn_rate_deg):
    """Constant speed on an arc. turn_rate_deg is degrees per second."""
    rows = []
    w = math.radians(turn_rate_deg)
    x = y = 0.0
    pts = []
    for i in range(gru_conceal.HISTORY_LEN):
        th = w * i * gru_conceal.SOURCE_STEP
        pts.append((x, y, speed * math.cos(th), speed * math.sin(th)))
        x += speed * math.cos(th) * gru_conceal.SOURCE_STEP
        y += speed * math.sin(th) * gru_conceal.SOURCE_STEP
    ox, oy = pts[-1][0], pts[-1][1]
    for px, py, vx, vy in pts:
        rows.append([px - ox, py - oy, 0.0, vx, vy, 0.0])
    last = pts[-1]
    return rows, (last[2], last[3], 0.0)


def stopping(speed, end_frac=0.25):
    """Decelerating over the history window, to end_frac of the speed.

    Not all the way to zero: dead reckoning's predicted distance would
    then be zero and the ratio undefined.
    """
    rows = []
    n = gru_conceal.HISTORY_LEN
    x = 0.0
    pts = []
    for i in range(n):
        v = speed * (1.0 - (1.0 - end_frac) * i / (n - 1))
        pts.append((x, v))
        x += v * gru_conceal.SOURCE_STEP
    ox = pts[-1][0]
    for px, v in pts:
        rows.append([px - ox, 0.0, 0.0, v, 0.0, 0.0])
    return rows, (pts[-1][1], 0.0, 0.0)


def norm(v):
    return math.sqrt(sum(c * c for c in v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../ml_model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=225.0,
                    help="the client's measured final speed")
    args = ap.parse_args()

    lad = gru_conceal.Ladder().load(args.dir, seed=args.seed)
    print(lad.describe())
    if not lad.models:
        return
    print()

    cases = [
        ("straight", straight(args.speed)),
        ("turn 60 deg/s", turning(args.speed, 60.0)),
        ("turn 180 deg/s", turning(args.speed, 180.0)),
        ("stopping", stopping(args.speed)),
    ]

    print(f"  Travel distance predicted over each horizon, at "
          f"{args.speed:.0f} u/s")
    print(f"  {'case':<16}{'rung':>6}{'DR':>9}{'GRU':>9}{'ratio':>8}"
          f"{'angle':>8}")
    print("  " + "-" * 56)

    for name, (hist, vel) in cases:
        origin = (hist[-1][0], hist[-1][1], hist[-1][2])
        for rows in lad.rows:
            h = gru_conceal.rung_seconds(rows)
            pred = lad.predict(rows, hist)
            gv = (pred[0] - origin[0], pred[1] - origin[1],
                  pred[2] - origin[2])
            dv = (vel[0] * h, vel[1] * h, vel[2] * h)
            dgru, ddr = norm(gv), norm(dv)

            # Angle between the model's displacement and DR's. Near zero
            # means they point the same way and only the distance
            # differs; large means the hook is feeding garbage.
            if dgru > 1e-6 and ddr > 1e-6:
                cos = sum(a * b for a, b in zip(gv, dv)) / (dgru * ddr)
                ang = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            else:
                ang = float("nan")

            print(f"  {name:<16}{rows:>6}{ddr:>9.1f}{dgru:>9.1f}"
                  f"{dgru/ddr if ddr else float('nan'):>8.2f}"
                  f"{ang:>8.1f}")
        print()

    # ------------------------------------------------------------------
    # Framing invariance. gru_model.to_delta() subtracts the current
    # position from every frame, so the SAME motion at a different place
    # on the map must give the same displacement. The live server feeds
    # absolute Dust2 coordinates near +/-2000; these synthetic cases sit
    # near the origin. If those disagreed, every live prediction would be
    # wrong in a way the dataset evaluation could never reveal, because
    # training and evaluation both used the same framing.
    hist, vel = straight(args.speed)
    shifted = [[r[0] + 1500.0, r[1] - 900.0, r[2] + 64.0,
                r[3], r[4], r[5]] for r in hist]
    worst = 0.0
    for rows in lad.rows:
        a = lad.predict(rows, hist)
        b = lad.predict(rows, shifted)
        d = max(abs((b[0] - 1500.0) - a[0]),
                abs((b[1] + 900.0) - a[1]),
                abs((b[2] - 64.0) - a[2]))
        worst = max(worst, d)
    print(f"  framing invariance: same motion moved 1500 units across the")
    print(f"  map changes the prediction by {worst:.4f} units "
          f"({'OK' if worst < 0.05 else 'FAILS -- the hook is wrong'})")
    print()

    print("  HOW TO READ THIS")
    print("  straight, ratio near 1.00, angle near 0  -> the model agrees")
    print("    with dead reckoning on straight motion. The live loss is")
    print("    then NOT a learned slowing bias, and the hook is suspect.")
    print("  straight, ratio well below 1.00          -> the model expects")
    print("    players to slow down. Correct for real play, wrong for the")
    print("    generator, and sufficient to explain the live result.")
    print("  angle large on straight                  -> the hook is")
    print("    broken: the prediction does not even point where the")
    print("    player is going.")
    print("  stopping, ratio above 1.00               -> the model is")
    print("    slower to believe a stop than DR is, which is the opposite")
    print("    bias and would show up as error on real stops.")


if __name__ == "__main__":
    main()
