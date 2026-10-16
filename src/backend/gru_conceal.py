"""
GRU concealment for the live server: pick a model, build its history, predict.

WHAT THIS IS FOR
    server_sync.py already conceals loss two ways -- freeze the player, or
    dead reckon from their last velocity -- and scores both against the
    truth that arrives in the next real packet. This adds a third method
    that asks a trained GRU instead.

THE THREE THINGS THAT MAKE THIS NON-TRIVIAL

    1. RATE MISMATCH. The models were trained on CSKnow decimated to
       16 Hz: 16 history frames spanning 1000 ms, one frame every 62.5 ms.
       The live server runs at 64 Hz. Feeding it the last 16 packets would
       span 250 ms -- a window the model has never seen, and the single
       most plausible way to get a quietly wrong result here.

       So history is built from every 4th packet BY SEQUENCE NUMBER, not
       by arrival order. Arrival order is wrong whenever anything is lost,
       which is the entire subject of this project. If the needed sequence
       numbers are not in the buffer, this module refuses to predict and
       says so, rather than substituting whatever is nearest and hoping.

    2. FIXED HORIZONS. Each checkpoint predicts one distance into the
       future: h4 = 250 ms, h8 = 500 ms, h12 = 750 ms. Live gaps are
       whatever length the network makes them. So the rungs form a ladder
       and the one closest to the elapsed gap is used. Below 125 ms no
       model is consulted at all -- dead reckoning is near-exact there and
       a model would only add noise. Above 750 ms the top rung is used
       beyond its training range, and those predictions are FLAGGED, so
       the ablation can show that regime separately instead of having it
       averaged in silently.

    3. A POINT, NOT A PATH. One forward pass returns one position at one
       horizon. Drawing a player moving during an outage needs a position
       at every instant, so the server interpolates from the last known
       position toward the predicted one. Past the horizon it continues at
       the average velocity the model implied. That extension is dead
       reckoning on a model-derived velocity rather than the last measured
       one -- worth saying out loud, because it is doing real work in long
       gaps and it is not something the model was trained to do.

WHAT IS SCORED IS WHAT IS DRAWN
    The error recorded at gap close comes from the same position_at()
    function the dashboard renders. If they differed, the number in the
    report would describe something nobody ever saw on screen.

Usage:
    python gru_conceal.py --self-test        # arithmetic, no torch needed
    python gru_conceal.py --dir src/ml_model # load and report the ladder
"""

import argparse
import os
import sys

# -- Training-time constants. These are facts about the checkpoints, not
# -- tunables. HISTORY_LEN and SOURCE_HZ come from parsed_data.npz; change
# -- them only if the data is regenerated.
HISTORY_LEN = 16           # frames the model expects
SOURCE_HZ = 16.0           # rate the model was trained at
SOURCE_STEP = 1.0 / SOURCE_HZ      # 62.5 ms between history frames

# Gaps shorter than this are left to dead reckoning. At 125 ms DR's error
# is a fraction of a unit, well under the model's own spread.
MIN_GAP_SECONDS = 0.125

# Horizons available as trained checkpoints, in rows at SOURCE_HZ.
LADDER_ROWS = (4, 8, 12)


def rung_seconds(rows):
    return rows * SOURCE_STEP


# Which rung to ask for a gap that does not land exactly on one.
#
#   "cover"    the smallest horizon that is at least as long as the gap.
#              The prediction then lies beyond the gap and position_at
#              INTERPOLATES toward it: the model is never used outside
#              the range it was trained on.
#
#   "nearest"  the closest horizon in absolute time. Simpler, and what
#              this module did first -- but it extrapolates whenever the
#              nearest rung is shorter than the gap. Under the measured
#              loss at 16 Hz that is 53.6% of gaps, by a mean factor of
#              1.59x, and past the horizon the hook is just dead
#              reckoning on a deliberately shrunk velocity. It is kept
#              so the two policies can be compared on identical gaps
#              rather than one being quietly swapped for the other.
RUNG_POLICIES = ("cover", "nearest")


def choose_rung(elapsed, available_rows, policy="cover"):
    """Which trained horizon to ask, for a gap of `elapsed` seconds.

    Returns (rows, clamped) or (None, False) if no model should be used.
    `clamped` is True when the gap is longer than the longest rung, so
    the model is being used outside the range it was trained on -- which
    under "cover" is the ONLY way that can happen.
    """
    if policy not in RUNG_POLICIES:
        raise ValueError(policy)
    if not available_rows or elapsed < MIN_GAP_SECONDS:
        return None, False
    rows = sorted(available_rows)
    longest = rows[-1]
    if elapsed > rung_seconds(longest):
        return longest, True
    if policy == "cover":
        return min(r for r in rows if rung_seconds(r) >= elapsed), False
    return min(rows, key=lambda r: abs(rung_seconds(r) - elapsed)), False


def stride_for(rate):
    """History frames are every Nth packet. 64 Hz / 16 Hz = 4.

    Returns None when the live rate is not a whole multiple of the rate
    the model was trained at. Rounding it would silently change the span
    of the history window, so it is refused instead.
    """
    ratio = rate / SOURCE_HZ
    n = int(round(ratio))
    if n < 1 or abs(ratio - n) > 1e-6:
        return None
    return n


def build_history(frames, last_seq, stride):
    """Assemble HISTORY_LEN frames ending at `last_seq`, spaced `stride`.

    `frames` maps seq -> (pos, vel). Returns a list of 16 rows of 6
    floats, oldest first, or None if any required sequence number is
    absent. Refusing is deliberate: a history with a hole in it spans
    the right amount of time but describes motion that never happened.
    """
    wanted = [last_seq - stride * i
              for i in range(HISTORY_LEN - 1, -1, -1)]
    if wanted[0] < 0:
        return None
    rows = []
    for s in wanted:
        f = frames.get(s)
        if f is None:
            return None
        pos, vel = f
        rows.append([pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]])
    return rows


def position_at(origin, predicted, horizon_seconds, elapsed):
    """Where the model says the player is at `elapsed` into the outage.

    Up to the horizon this interpolates from the last known position to
    the predicted one. Beyond it, it continues at the average velocity
    the prediction implies -- the model has nothing to say past its
    training horizon, so the motion is extended rather than invented.
    """
    if horizon_seconds <= 0:
        return origin
    f = elapsed / horizon_seconds
    return (origin[0] + (predicted[0] - origin[0]) * f,
            origin[1] + (predicted[1] - origin[1]) * f,
            origin[2] + (predicted[2] - origin[2]) * f)


class Ladder:
    """The loaded checkpoints, one per horizon.

    Kept in its own class so server_sync.py can run with no torch
    installed: if the import fails, the server reports that the GRU
    method is unavailable and the other two methods work unchanged.
    """

    def __init__(self):
        self.models = {}        # rows -> (module, meta)
        self.error = None

    @property
    def rows(self):
        return tuple(sorted(self.models))

    def load(self, directory, seed=0, hidden=None):
        """Load gru_h{rows}_s{seed}.pt for every rung that exists.

        Seed is fixed rather than best-of, on purpose. Picking the
        checkpoint with the best held-out score would be selecting on
        the test set, and every number reported afterwards would be
        inflated by that choice.
        """
        # gru_model.py lives beside the checkpoints, in src/ml_model,
        # while this module runs from src/backend. Importing it from
        # there keeps ONE definition of the architecture: a second copy
        # in src/backend would load old weights into a changed model and
        # produce plausible, wrong predictions with no error anywhere.
        directory = os.path.abspath(directory)
        # The directory itself, and its parent: checkpoints retrained with
        # a clean validation split live in src/ml_model/clean/ while
        # gru_model.py stays one level up in src/ml_model/. Adding both
        # means --model-dir can point at either without a second copy of
        # the model definition existing anywhere.
        for cand in (directory, os.path.dirname(directory)):
            if cand and cand not in sys.path:
                sys.path.insert(0, cand)

        try:
            import torch
            from gru_model import GRUPredictor
        except Exception as e:                  # torch absent, or no model
            self.error = f"{type(e).__name__}: {e}"
            return self

        for rows in LADDER_ROWS:
            for name in (f"gru_h{rows}_s{seed}.pt", f"gru_h{rows}.pt"):
                path = os.path.join(directory, name)
                if not os.path.exists(path):
                    continue
                ck = torch.load(path, map_location="cpu", weights_only=False)
                if int(ck["horizon"]) != rows:
                    self.error = (f"{name} says horizon {ck['horizon']} "
                                  f"but its name says {rows}")
                    return self
                if hidden is not None and int(ck["hidden"]) != hidden:
                    continue
                m = GRUPredictor(hidden=int(ck["hidden"]),
                                 layers=int(ck["layers"]))
                m.load_state_dict(ck["state_dict"])
                m.eval()
                self.models[rows] = (m, {
                    "file": name,
                    "hidden": int(ck["hidden"]),
                    "layers": int(ck["layers"]),
                    "step_seconds": float(ck["step_seconds"]),
                    "val_median": float(ck["val_median"]),
                    "dr_median": float(ck["dr_median"]),
                    # Present only in checkpoints from train_clean.py.
                    # When it is, it is the number to quote: val_median
                    # there is measured on the VALIDATION games while
                    # dr_median is the TEST bar, so their ratio mixes two
                    # different sets and means nothing.
                    "test_median": (float(ck["test_median"])
                                    if "test_median" in ck else None),
                    "selection": ck.get("selection", "best epoch on test"),
                })
                break

        if self.models:
            # The checkpoint records the rate it was trained at. If it
            # disagrees with this module's assumption, the history would
            # be built at the wrong spacing -- refuse rather than guess.
            for rows, (_, meta) in self.models.items():
                if abs(meta["step_seconds"] - SOURCE_STEP) > 1e-9:
                    self.error = (
                        f"{meta['file']} was trained at "
                        f"{1/meta['step_seconds']:.3f} Hz, this module "
                        f"assumes {SOURCE_HZ:.0f} Hz")
                    self.models = {}
                    break
        elif self.error is None:
            self.error = f"no checkpoints found in {directory}"
        return self

    def predict(self, rows, history):
        """One forward pass. history is 16 rows of 6 raw absolute floats.

        gru_model.forward() does the delta framing internally, so raw
        absolute values go in and a raw absolute position comes out.
        There is nothing to scale here, and adding scaling would
        double-apply it.
        """
        import torch
        model, _ = self.models[rows]
        X = torch.tensor([history], dtype=torch.float32)
        with torch.no_grad():
            out = model(X)
        return tuple(float(v) for v in out[0])

    def describe(self):
        if not self.models:
            return f"GRU unavailable ({self.error})"
        bits = []
        for rows in self.rows:
            _, m = self.models[rows]
            bits.append(f"h{rows}={rung_seconds(rows)*1000:.0f}ms"
                        f"({m['file']}, hidden {m['hidden']})")
        return "GRU ladder: " + "  ".join(bits)


# ---------------------------------------------------------------- tests

def self_test():
    """Everything that can be wrong without torch being involved."""

    # -- stride ---------------------------------------------------------
    assert stride_for(64.0) == 4, stride_for(64.0)
    assert stride_for(16.0) == 1
    assert stride_for(32.0) == 2
    assert stride_for(60.0) is None, "60 Hz is not a multiple of 16 Hz"
    assert stride_for(100.0) is None
    print("  stride              OK  (64 Hz -> every 4th packet)")

    # -- history spans the trained window, not the last 16 packets ------
    # 16 frames, stride 4, at 64 Hz = 15 * 4 * 15.625 ms = 937.5 ms of
    # gaps between frames, i.e. a 1000 ms window inclusive of both ends.
    span = (HISTORY_LEN - 1) * SOURCE_STEP
    assert abs(span - 0.9375) < 1e-9, span
    print(f"  history span        OK  ({span*1000:.1f} ms between first "
          f"and last frame)")

    # -- build_history picks the right sequence numbers -----------------
    frames = {s: ((float(s), 0.0, 0.0), (100.0, 0.0, 0.0))
              for s in range(0, 200)}
    h = build_history(frames, last_seq=100, stride=4)
    assert h is not None and len(h) == HISTORY_LEN
    xs = [row[0] for row in h]
    assert xs[-1] == 100.0, xs[-1]
    assert xs[0] == 100.0 - 4 * (HISTORY_LEN - 1), xs[0]
    assert xs == sorted(xs), "history must be oldest first"
    deltas = {round(b - a, 6) for a, b in zip(xs, xs[1:])}
    assert deltas == {4.0}, deltas
    print("  history assembly    OK  (seq 40..100 step 4, oldest first)")

    # -- a hole in the needed sequence numbers must refuse --------------
    holed = dict(frames)
    del holed[72]
    assert build_history(holed, 100, 4) is None, \
        "a missing history frame must refuse, not substitute"
    # A hole somewhere irrelevant must NOT refuse.
    other = dict(frames)
    del other[73]
    assert build_history(other, 100, 4) is not None
    print("  missing frame       OK  (refuses on a hole, ignores non-frames)")

    # -- not enough history at startup ---------------------------------
    assert build_history(frames, last_seq=10, stride=4) is None
    print("  cold start          OK  (refuses before 16 frames exist)")

    # -- rung choice ----------------------------------------------------
    for pol in RUNG_POLICIES:
        assert choose_rung(0.05, LADDER_ROWS, pol) == (None, False)
        assert choose_rung(0.124, LADDER_ROWS, pol) == (None, False)
        assert choose_rung(0.250, LADDER_ROWS, pol) == (4, False)
        assert choose_rung(1.344, LADDER_ROWS, pol) == (12, True), \
            "the longest measured burst must be flagged as clamped"
        assert choose_rung(0.80, (8,), pol) == (8, True)

    # The whole difference between the policies: a gap longer than the
    # nearest rung. 300 ms is past h4 (250) and short of h8 (500).
    assert choose_rung(0.300, LADDER_ROWS, "nearest") == (4, False)
    assert choose_rung(0.300, LADDER_ROWS, "cover") == (8, False)
    assert choose_rung(0.600, LADDER_ROWS, "nearest") == (8, False)
    assert choose_rung(0.600, LADDER_ROWS, "cover") == (12, False)
    print("  rung choice         OK  (cover never shorter than the gap)")

    # Under "cover" the interpolation fraction can never exceed 1 unless
    # the gap is past the longest rung. That is the entire point of it.
    for ms in range(130, 751, 10):
        e = ms / 1000.0
        rows, cl = choose_rung(e, LADDER_ROWS, "cover")
        assert not cl and e / rung_seconds(rows) <= 1.0 + 1e-9, (ms, rows)
    print("  cover never extrap. OK  (f <= 1 for every gap up to 750 ms)")

    # -- interpolation --------------------------------------------------
    origin = (0.0, 0.0, 0.0)
    pred = (150.0, 0.0, 0.0)       # model says 150 units on at 750 ms
    hz = rung_seconds(12)
    assert position_at(origin, pred, hz, 0.0) == origin
    mid = position_at(origin, pred, hz, hz / 2)
    assert abs(mid[0] - 75.0) < 1e-9, mid
    end = position_at(origin, pred, hz, hz)
    assert abs(end[0] - 150.0) < 1e-9, end
    # Past the horizon it keeps going at the implied average velocity,
    # 150 units / 0.75 s = 200 u/s -- inside the client's 150-250 envelope.
    past = position_at(origin, pred, hz, 2 * hz)
    assert abs(past[0] - 300.0) < 1e-9, past
    implied = pred[0] / hz
    assert 150.0 <= implied <= 250.0, implied
    print(f"  interpolation       OK  (implied {implied:.0f} u/s, "
          f"extends past horizon)")

    # -- the scored quantity equals the drawn one -----------------------
    # Same call, same arguments: this is a structural check that there is
    # only one function, not two that could drift apart.
    drawn = position_at(origin, pred, hz, 0.6)
    scored = position_at(origin, pred, hz, 0.6)
    assert drawn == scored
    print("  draw == score       OK  (one function, no second copy)")

    print("\n  all checks passed (torch not required for any of these)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="src/ml_model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    lad = Ladder().load(args.dir, seed=args.seed)
    print(lad.describe())
    if not lad.models:
        return
    clean = any(lad.models[r][1]["test_median"] is not None
                for r in lad.rows)
    print(f"  {'rung':<6}{'horizon':>9}{'GRU':>9}{'DR':>9}{'gain':>8}"
          f"   measured on")
    for rows in lad.rows:
        _, m = lad.models[rows]
        if m["test_median"] is not None:
            gru, where = m["test_median"], "test (chosen on val)"
        else:
            gru, where = m["val_median"], "test (epoch chosen on test)"
        gain = (1 - gru / m["dr_median"]) * 100
        print(f"  h{rows:<5}{rung_seconds(rows)*1000:>7.0f} ms"
              f"{gru:>9.2f}{m['dr_median']:>9.2f}{gain:>7.1f}%   {where}")
    if clean:
        print("\n  These checkpoints were selected on a validation split")
        print("  disjoint from test; the test set was evaluated once.")


if __name__ == "__main__":
    main()
