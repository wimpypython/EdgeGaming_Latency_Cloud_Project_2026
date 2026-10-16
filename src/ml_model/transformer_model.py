"""
A transformer encoder that predicts where a player will be, given 16
frames of history. The third method numbers #2 and #3 are specified as.

WHAT THIS IS, AND WHAT IT IS NOT
    The project plan calls for a "CSKnow transformer baseline". The
    CSKnow paper's transformer does behaviour modelling over longer
    horizons, and its code and weights are not in hand. So this is NOT a
    reproduction of their model and must not be described as one.

    It is a transformer encoder solving EXACTLY the task our GRU solves:

        in  : (batch, 16, 6)   16 frames of [px py pz vx vy vz]
        out : (batch, 3)       predicted position HORIZON rows later

    Same inputs, same outputs, same framing, same training loop, same
    held-out games. That makes it an architecture comparison, which is
    the useful thing, rather than a claim about someone else's paper.

WHY THE FRAMING IS IMPORTED, NOT REIMPLEMENTED
    to_delta / from_delta come from gru_model.py. If this file had its
    own copy, the two architectures could end up framed differently --
    and a framing difference would show up as an architecture result.
    One definition, imported, so that cannot happen.

WHY TWO SIZES ARE REPORTED
    The GRU has 14,019 parameters. A transformer at d_model=64 with 2
    layers has roughly 66k. Reporting only the large one invites "it
    only won because it is bigger"; reporting only a matched one invites
    "you undersized it". Both are run:

        --d-model 32 --layers 1 --ff-mult 4    ~13.5k, matched
        --d-model 64 --layers 2 --ff-mult 4    ~100k, conventional

WHY A TRANSFORMER IS NOT EXPECTED TO WIN
    Self-attention earns its cost on long-range dependencies. The window
    here is 16 frames -- one second. There is no long range in it. The
    GRU's own size ablation already suggests capacity is not the limit:
    doubling hidden units bought 1.8%. If the transformer wins by a wide
    margin, suspect a bug before believing it.

POSITIONAL ENCODING
    Self-attention is permutation-invariant: without position
    information the model cannot tell the oldest frame from the newest,
    which for a trajectory is fatal. A LEARNED embedding over 16
    positions is used -- 16 is small and fixed, so there is nothing for
    sinusoidal encoding to generalise to.

Usage:
    python transformer_model.py                       # shape + sanity only
    python transformer_model.py --d-model 64 --layers 2
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

# ONE definition of the framing, shared with the GRU.
from gru_model import to_delta, from_delta, POS_SCALE, VEL_SCALE


class TransformerPredictor(nn.Module):
    """
    Project each frame to d_model, add a learned position embedding, run
    a transformer encoder over the 16 frames, and read the prediction
    off the LAST position's output.

    Reading off the last position rather than mean-pooling is deliberate:
    it is what the GRU does -- its final hidden state is the state after
    the last frame -- so the two differ in how they mix the sequence, not
    in what they read out of it.
    """

    def __init__(self, n_features=6, d_model=32, layers=1, nhead=4,
                 ff_mult=4, dropout=0.0, seq_len=16):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model {d_model} not divisible by nhead "
                             f"{nhead}")
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # stabler without a warmup schedule
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(d_model, 3)

    def forward(self, X):
        """X: (batch, seq, 6) absolute -> (batch, 3) absolute.

        Identical contract to GRUPredictor.forward: raw absolute in, raw
        absolute out, framing handled internally. Callers cannot tell the
        two apart, which is what lets one training loop serve both.
        """
        Xd, origin = to_delta(X)
        h = self.proj(Xd) + self.pos[:, :Xd.shape[1], :]
        h = self.enc(h)
        offset = self.head(h[:, -1, :])
        return from_delta(offset, origin)


# ---------------------------------------------------------------------------
# Checks -- run before training, so a shape bug costs seconds not hours
# ---------------------------------------------------------------------------

def check_positions_matter(model, X):
    """Reversing the history must change the prediction.

    If it does not, the positional embedding is not doing its job and the
    model is seeing a bag of frames rather than a trajectory -- which
    would still train, still produce a plausible number, and be wrong
    about the thing this task is made of.
    """
    model.eval()
    with torch.no_grad():
        a = model(X)
        b = model(torch.flip(X, dims=[1]))
    gap = (a - b).abs().max().item()
    print(f"  forward vs reversed history  : {gap:.4f} units apart")
    assert gap > 1e-3, (
        "reversing the history changed nothing -- the position embedding "
        "is not being used, so the model cannot tell past from present")
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="parsed_data.npz")
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff-mult", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(1)           # the project's latency claim is 1 core

    d = np.load(args.input, allow_pickle=False)
    X_np, y_np = d["X"], d["y"]
    horizon = int(d["horizon"])
    step = float(d["step_seconds"])

    print(f"Loaded  : {args.input}")
    print(f"X       : {X_np.shape}")
    print(f"horizon : {horizon} rows ({horizon * step * 1000:.1f} ms)")

    idx = np.random.default_rng(0).choice(len(X_np), args.batch,
                                          replace=False)
    X = torch.from_numpy(X_np[idx])
    y = torch.from_numpy(y_np[idx])

    model = TransformerPredictor(d_model=args.d_model, layers=args.layers,
                                 nhead=args.nhead, ff_mult=args.ff_mult)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nModel")
    print(f"  d_model                      : {args.d_model}")
    print(f"  layers                       : {args.layers}")
    print(f"  heads                        : {args.nhead}")
    print(f"  parameters                   : {n_params:,}")
    print(f"  (the GRU has 14,019 -- ratio {n_params/14019:.2f}x)")

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

    print(f"\nFraming and ordering")
    # Same framing as the GRU, so the same invariance must hold: the same
    # motion somewhere else on the map must give the same displacement.
    shifted = X.clone()
    shifted[:, :, 0] += 1500.0
    shifted[:, :, 1] -= 900.0
    with torch.no_grad():
        out2 = model(shifted)
    drift = max((out2[:, 0] - 1500.0 - out[:, 0]).abs().max().item(),
                (out2[:, 1] + 900.0 - out[:, 1]).abs().max().item())
    print(f"  map-position invariance      : {drift:.4f} units")
    assert drift < 0.05, "delta framing is not holding -- stop here"

    check_positions_matter(model, X)

    with torch.no_grad():
        untrained = torch.linalg.norm(out - y, dim=1)
        null = torch.linalg.norm(X[:, -1, 0:3] - y, dim=1)
    print(f"\nSanity (untrained, {args.batch} random examples)")
    print(f"  null baseline median error   : {null.median():.2f} units")
    print(f"  untrained model median error : {untrained.median():.2f} units")
    ratio = (untrained.median() / null.median()).item()
    print("  OK: same order of magnitude" if ratio < 10 else
          "  WARNING: far above null -- suspect framing or scales")

    print(f"\nInference latency, single CPU thread")
    for b in (1, 10, 64):
        xb = X[:b]
        with torch.no_grad():
            for _ in range(20):
                model(xb)
            t0 = time.perf_counter()
            for _ in range(200):
                model(xb)
            ms = (time.perf_counter() - t0) / 200 * 1000
        print(f"  batch {b:>3} : {ms:7.3f} ms per call"
              f"   ({ms / b:6.4f} ms per player)")
    print("  (frame budget at 60fps is 16.7 ms; the GRU is 1.07 ms at")
    print("   batch 10, so this is the number that decides whether a")
    print("   transformer is deployable here at all)")

    print("\nNothing was trained. Train with:")
    print(f"  python train_clean.py --input {args.input} "
          f"--arch transformer --d-model {args.d_model} "
          f"--layers {args.layers}")


if __name__ == "__main__":
    main()
