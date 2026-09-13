"""
A GRU that predicts where a player will be, given 16 frames of history.

THE JOB
    in  : (batch, 16, 6)   16 frames of [px py pz vx vy vz]
    out : (batch, 3)       predicted position HORIZON rows later

    Same job dead reckoning does, with 16 frames instead of 1.

WHY DELTA FRAMING (the one design decision that matters)
    X holds absolute Dust2 coordinates, roughly +/-2000. Fed in raw, the
    model would spend its capacity memorising that mid is near x=500 and
    B site near y=2500 -- learning the MAP, not movement. It would then
    fail completely on any other map.

    So the current position is subtracted from every frame and from the
    target. Every example then starts at (0,0,0) and the model sees only
    "where did this player go, relative to where they are now".

    This also makes the comparison with dead reckoning like-for-like,
    because DR is itself a delta method: it predicts an offset.

WHY THE FRAMING LIVES INSIDE forward()
    If the caller had to remember to delta-frame the input and un-frame
    the output, that is two places to get it wrong, in every script that
    touches the model. Instead forward() takes RAW absolute positions and
    returns RAW absolute positions. There is nothing to remember.

    Adding the origin back is a constant shift, so it does not change the
    loss gradient -- training on absolute outputs is mathematically the
    same as training on deltas, but with no chance of a framing mismatch
    between train time and eval time.

WHY FIXED SCALES, NOT FITTED STATISTICS
    Networks train badly on large inputs, so deltas are divided down.
    The divisors are CONSTANTS, not means and standard deviations fitted
    on the training set. Fitted statistics are another way for test data
    to influence training, and this task does not need them.

Usage:
    python gru_model.py                      # shape + sanity check only
    python gru_model.py --hidden 128
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

# Typical magnitude of a position delta over the window, in game units.
# Deltas over 16 frames run to a few hundred; this brings them near 1.
POS_SCALE = 100.0

# Roughly max CS ground speed in units/second, so velocities land near 1.
VEL_SCALE = 250.0


# ---------------------------------------------------------------------------
# Framing -- module level so it can be tested on its own
# ---------------------------------------------------------------------------

def to_delta(X):
    """
    Absolute -> relative.

    X : (batch, seq, 6) raw
    returns (Xd, origin) where origin is (batch, 3), the current position.
    """
    origin = X[:, -1, 0:3]
    Xd = torch.empty_like(X)
    Xd[:, :, 0:3] = (X[:, :, 0:3] - origin[:, None, :]) / POS_SCALE
    Xd[:, :, 3:6] = X[:, :, 3:6] / VEL_SCALE
    return Xd, origin


def from_delta(out, origin):
    """Relative -> absolute. Exact inverse of the position half of to_delta."""
    return out * POS_SCALE + origin


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class GRUPredictor(nn.Module):
    """
    A GRU walks the 16 frames in order, carrying a hidden state that
    summarises everything seen so far. At each frame it decides how much
    of the new frame to absorb and how much of the running summary to
    keep. After the last frame that hidden state is a learned compression
    of the whole trajectory, and one linear layer turns it into a
    position offset.

    Dead reckoning sees one velocity vector. This sees whether the player
    has been accelerating, weaving, or holding steady for a second. At
    500ms that context is what separates "will keep running" from "is
    about to stop".
    """

    def __init__(self, n_features=6, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 3)

    def forward(self, X):
        """X: (batch, seq, 6) absolute -> (batch, 3) absolute."""
        Xd, origin = to_delta(X)
        _, h = self.gru(Xd)          # h: (layers, batch, hidden)
        offset = self.head(h[-1])    # last layer's final hidden state
        return from_delta(offset, origin)


# ---------------------------------------------------------------------------
# Checks -- run before training, so a shape bug costs seconds not hours
# ---------------------------------------------------------------------------

def check_framing_roundtrip(X):
    """to_delta then from_delta must return the original positions."""
    Xd, origin = to_delta(X)
    # The last frame's delta is zero by construction, so recovering it is
    # the same as recovering the origin.
    recovered = from_delta(Xd[:, -1, 0:3], origin)
    err = (recovered - X[:, -1, 0:3]).abs().max().item()
    print(f"  framing round-trip max error : {err:.3e}")
    assert err < 1e-2, "delta framing is not reversible -- stop here"

    print(f"  delta position range         : "
          f"[{Xd[:, :, 0:3].min():.2f}, {Xd[:, :, 0:3].max():.2f}]")
    print(f"  scaled velocity range        : "
          f"[{Xd[:, :, 3:6].min():.2f}, {Xd[:, :, 3:6].max():.2f}]")
    print("  (both should sit roughly within +/-5; far larger means the")
    print("   scales are wrong and training will struggle)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="parsed_data.npz")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)          # the project's latency claim is 1 core

    d = np.load(args.input, allow_pickle=False)
    X_np, y_np = d["X"], d["y"]
    horizon = int(d["horizon"])
    step = float(d["step_seconds"])

    print(f"Loaded  : {args.input}")
    print(f"X       : {X_np.shape}")
    print(f"horizon : {horizon} rows ({horizon * step * 1000:.1f} ms)")
    if horizon != 8:
        print("  NOTE: the sweep says horizon 8 (500ms) is where dead")
        print("  reckoning actually fails. Regenerate with:")
        print("      python parse_csknow.py --horizon 8")

    # A small batch is enough to check shapes.
    idx = np.random.default_rng(0).choice(len(X_np), args.batch, replace=False)
    X = torch.from_numpy(X_np[idx])
    y = torch.from_numpy(y_np[idx])

    print("\nFraming")
    check_framing_roundtrip(X)

    model = GRUPredictor(hidden=args.hidden, layers=args.layers)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel")
    print(f"  hidden units                 : {args.hidden}")
    print(f"  layers                       : {args.layers}")
    print(f"  parameters                   : {n_params:,}")

    model.eval()
    with torch.no_grad():
        out = model(X)

    print(f"\nShapes")
    print(f"  in                           : {tuple(X.shape)}")
    print(f"  out                          : {tuple(out.shape)}")
    assert out.shape == (len(X), 3), \
        f"expected ({len(X)}, 3), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "model produced NaN or inf"
    print("  OK: (batch, 16, 6) -> (batch, 3)")

    # -------------------------------------------------------------------
    # An UNTRAINED model outputs small random offsets, so once the origin
    # is added back its error should land near the null baseline. Wildly
    # different means the framing or scaling is inverted -- which is
    # exactly the bug worth catching before spending an hour training.
    # -------------------------------------------------------------------
    with torch.no_grad():
        untrained = torch.linalg.norm(out - y, dim=1)
        null = torch.linalg.norm(X[:, -1, 0:3] - y, dim=1)

    print(f"\nSanity (untrained model, {args.batch} random examples)")
    print(f"  null baseline median error   : {null.median():.2f} units")
    print(f"  untrained model median error : {untrained.median():.2f} units")
    ratio = (untrained.median() / null.median()).item()
    if ratio > 10:
        print("  WARNING: untrained error is far above the null baseline.")
        print("  Suspect the framing or the scales before training.")
    else:
        print("  OK: same order of magnitude, as expected before training.")

    # -------------------------------------------------------------------
    # Inference latency on ONE core. batch=1 is a single player; batch=10
    # is every player in a round, which is how the CSKnow paper reports
    # its own figure.
    # -------------------------------------------------------------------
    print(f"\nInference latency, single CPU thread")
    for b in (1, 10, 64):
        xb = X[:b]
        with torch.no_grad():
            for _ in range(20):
                model(xb)                      # warm up
            t0 = time.perf_counter()
            for _ in range(200):
                model(xb)
            ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"  batch {b:>3} : {ms:7.3f} ms per call"
              f"   ({ms / b:6.4f} ms per player)")
    print("  (frame budget at 60fps is 16.7 ms)")

    print("\nNothing was trained. This only checks that the plumbing is")
    print("right. Training is the next script.")


if __name__ == "__main__":
    main()
