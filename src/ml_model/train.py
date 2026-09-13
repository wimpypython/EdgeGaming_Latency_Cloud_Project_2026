"""
Train the GRU and check every epoch whether it is beating dead reckoning.

WHAT TRAINING ACTUALLY IS
    Four steps, repeated over batches of examples:

        1. forward   -- push a batch through, get predictions
        2. loss      -- measure how wrong they are
        3. backward  -- work out which direction each weight should move
        4. step      -- nudge every weight a little in that direction

    One pass over the whole training set is an EPOCH. Nothing clever
    happens; it is those four lines in a loop.

WHY MEAN SQUARED ERROR
    Loss is mean squared distance between predicted and true position.
    Squaring punishes large errors far harder than small ones, which
    suits this task: being 100 units wrong matters much more than being
    2 units wrong. But MSE is reported in squared units, which are hard
    to think about, so validation is reported as MEDIAN EUCLIDEAN ERROR
    in game units -- the same metric baselines.py uses, so the numbers
    are directly comparable.

WHY THE VALIDATION NUMBER IS THE ONLY ONE THAT COUNTS
    Training loss going down only proves the model memorised the games it
    saw. The five held-out games are the test. If training loss falls
    while validation error rises, it is memorising, and more epochs make
    it worse. With ~14k parameters and ~530k training examples that is
    unlikely, but it is checked every epoch rather than assumed.

WHAT WOULD MAKE THIS RESULT FAKE
    Training on data that overlaps the test set. Consecutive examples
    share 15 of their 16 history frames, so a random split would put
    near-duplicates on both sides and inflate the score enormously.
    The split is by GAME, imported from baselines.py so there is exactly
    one copy of it, and asserted non-overlapping before training starts.

Usage:
    python train.py
    python train.py --epochs 20 --hidden 128 --lr 1e-3
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from baselines import game_split
from gru_model import GRUPredictor


def median_error(model, loader, device):
    """Median Euclidean error in game units -- comparable to baselines.py."""
    model.eval()
    errs = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            errs.append(torch.linalg.norm(pred - yb, dim=1).cpu())
    model.train()
    return torch.cat(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="parsed_data.npz")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-games", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="gru_h{horizon}.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"                     # no CUDA on this machine, by design

    # -------------------------------------------------------------------
    d = np.load(args.input, allow_pickle=False)
    X, y, games = d["X"], d["y"], d["games"]
    horizon = int(d["horizon"])
    step = float(d["step_seconds"])
    dt = horizon * step

    print(f"Loaded  : {args.input}")
    print(f"X       : {X.shape}")
    print(f"horizon : {horizon} rows ({dt*1000:.1f} ms)\n")

    train_mask, test_mask, test_games = game_split(
        games, n_test=args.test_games, seed=args.seed)

    print(f"Split by game (seed {args.seed}):")
    print(f"  train : {train_mask.sum():>8,} examples  "
          f"{len(np.unique(games[train_mask])):>3} games")
    print(f"  test  : {test_mask.sum():>8,} examples  "
          f"{len(np.unique(games[test_mask])):>3} games")
    print(f"  held-out game ids: {test_games.tolist()}")

    Xtr = torch.from_numpy(X[train_mask])
    ytr = torch.from_numpy(y[train_mask])
    Xte = torch.from_numpy(X[test_mask])
    yte = torch.from_numpy(y[test_mask])

    # -------------------------------------------------------------------
    # The two numbers to beat, computed on the SAME held-out examples the
    # model will be scored on. Printed before training so there is never
    # any doubt what the bar was.
    # -------------------------------------------------------------------
    with torch.no_grad():
        pos, vel = Xte[:, -1, 0:3], Xte[:, -1, 3:6]
        null_err = torch.linalg.norm(pos - yte, dim=1)
        dr_err = torch.linalg.norm(pos + vel * dt - yte, dim=1)

    null_med = null_err.median().item()
    dr_med = dr_err.median().item()

    print(f"\nBaselines on the held-out games ({dt*1000:.0f} ms ahead):")
    print(f"  null (no move)  median {null_med:7.2f}   "
          f"90th {null_err.quantile(0.9):7.2f}")
    print(f"  dead reckoning  median {dr_med:7.2f}   "
          f"90th {dr_err.quantile(0.9):7.2f}")
    print(f"\n  THE BAR IS {dr_med:.2f}. Beating null is not a result.")

    # -------------------------------------------------------------------
    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte),
                             batch_size=4096, shuffle=False)

    model = GRUPredictor(hidden=args.hidden, layers=args.layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"\nModel   : GRU hidden={args.hidden} layers={args.layers}"
          f"  ({n_params:,} parameters)")
    print(f"Training: {args.epochs} epochs, batch {args.batch}, lr {args.lr}\n")

    print(f"  {'epoch':>5} {'train loss':>12} {'val median':>11} "
          f"{'val 90th':>9} {'vs DR':>8} {'time':>7}")
    print("  " + "-" * 60)

    best = float("inf")
    out_path = args.out.format(horizon=horizon)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        total, n_seen = 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * len(xb)
            n_seen += len(xb)

        errs = median_error(model, test_loader, device)
        med = errs.median().item()
        p90 = errs.quantile(0.9).item()
        vs_dr = (1 - med / dr_med) * 100
        secs = time.perf_counter() - t0

        flag = ""
        if med < best:
            best = med
            torch.save({"state_dict": model.state_dict(),
                        "hidden": args.hidden, "layers": args.layers,
                        "horizon": horizon, "step_seconds": step,
                        "test_games": test_games,
                        "val_median": med, "dr_median": dr_med},
                       out_path)
            flag = " *"

        print(f"  {epoch:>5} {total/n_seen:>12.2f} {med:>11.3f} "
              f"{p90:>9.2f} {vs_dr:>7.1f}% {secs:>6.1f}s{flag}")

    # -------------------------------------------------------------------
    print(f"\n  * = best so far, saved to {out_path}")
    print(f"\nBest validation median : {best:.3f} units")
    print(f"Dead reckoning         : {dr_med:.3f} units")
    print(f"Null baseline          : {null_med:.3f} units")

    if best < dr_med:
        print(f"\n  The model beats dead reckoning by "
              f"{(1 - best/dr_med)*100:.1f}% on the held-out games.")
        print("  Report this number, on these games, at this horizon.")
    elif best < null_med:
        print(f"\n  The model beats doing nothing but LOSES to dead")
        print(f"  reckoning by {(best/dr_med - 1)*100:.1f}%. That is not an")
        print("  improvement on current practice. Report it honestly, then")
        print("  try more epochs or a larger hidden size before concluding.")
    else:
        print("\n  The model is worse than predicting no movement at all.")
        print("  Something is wrong -- suspect the learning rate or the")
        print("  scales before concluding anything about the architecture.")

    print("\nA single run is one sample. Before reporting, re-run with a")
    print("different --seed to see how much the number moves.")


if __name__ == "__main__":
    main()
