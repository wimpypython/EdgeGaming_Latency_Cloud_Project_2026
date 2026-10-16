"""
Retrain with a clean validation split, so the reported number is not
selected on the test set.

WHAT WAS WRONG WITH train.py
    It evaluates on the held-out games every epoch and saves whichever
    epoch scores best:

        errs = median_error(model, test_loader, device)
        if med < best:
            torch.save(...)

    That is model selection on the test set. It is one choice among ten
    epochs rather than a hyperparameter sweep, so the bias is small --
    but it is a bias, it is in the favourable direction, and a reported
    result should not contain one that is this easy to remove.

WHAT THIS DOES INSTEAD
    Three disjoint sets of GAMES:

        test   5 games   touched EXACTLY ONCE, at the very end
        val    4 games   scored every epoch, chooses the epoch
        train 16 games   gradient updates

    The test set is not loaded into a DataLoader until training has
    finished and the epoch has been chosen. There is no code path by
    which a test score can influence a saved weight.

    Both numbers are printed: the validation median that drove the
    choice, and the test median that is the result. The gap between them
    is the size of the bias the old script was absorbing, which is worth
    knowing rather than guessing.

WHY THE SPLIT IS BY GAME, TWICE
    Consecutive examples share 15 of their 16 history frames, so any
    split finer than a whole game puts near-duplicates on both sides.
    That applies to the train/val split exactly as it applies to
    train/test, so val is whole games too -- otherwise the epoch would
    be chosen on data the model had effectively trained on, and the
    choice would be no better than the old one.

Usage:
    python train_clean.py --input parsed_data.npz --epochs 12
    python train_clean.py --input parsed_data.npz --seed 1
"""

import argparse
import os
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
            errs.append(torch.linalg.norm(model(xb) - yb, dim=1).cpu())
    model.train()
    return torch.cat(errs)


def three_way_split(games, n_test, n_val, seed):
    """test / val / train, disjoint at the level of whole games.

    The test split is taken with the project's one copy of game_split, at
    the same seed train.py used, so the test games are IDENTICAL to the
    ones every previous result was measured on. Only the remaining games
    are then divided, which means the new number is comparable with the
    old one rather than being measured on a different sample.
    """
    train_mask, test_mask, test_games = game_split(games, n_test=n_test,
                                                   seed=seed)

    pool = np.unique(games[train_mask])
    if n_val >= len(pool):
        raise SystemExit(f"n_val={n_val} but only {len(pool)} games remain")

    rng = np.random.default_rng(seed + 9973)      # independent of the first
    val_games = np.sort(rng.choice(pool, size=n_val, replace=False))

    val_mask = np.isin(games, val_games)
    fit_mask = train_mask & ~val_mask

    # Three assertions, because a split that overlaps looks fine.
    assert not np.intersect1d(games[fit_mask], games[val_mask]).size
    assert not np.intersect1d(games[fit_mask], games[test_mask]).size
    assert not np.intersect1d(games[val_mask], games[test_mask]).size
    assert fit_mask.sum() + val_mask.sum() + test_mask.sum() == len(games)

    return fit_mask, val_mask, test_mask, val_games, test_games


def baseline_errors(X, y, dt):
    pos, vel = X[:, -1, 0:3], X[:, -1, 3:6]
    null = torch.linalg.norm(pos - y, dim=1)
    dr = torch.linalg.norm(pos + vel * dt - y, dim=1)
    return null, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="parsed_data.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--arch", choices=["gru", "transformer"], default="gru",
                    help="both go through this same training loop, on the "
                         "same split, with the same loss and optimiser -- "
                         "so a difference in the result is a difference in "
                         "the architecture and not in how it was trained")
    ap.add_argument("--hidden", type=int, default=64,
                    help="GRU hidden units")
    ap.add_argument("--d-model", type=int, default=32,
                    help="transformer width. 32/1 layer is parameter-"
                         "matched to the GRU; 64/2 is the conventional size")
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff-mult", type=int, default=4,
                    help="feed-forward width as a multiple of d_model. "
                         "4 at d_model 32 gives ~13.5k parameters, which "
                         "is the match for the GRU's 14,019")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--test-games", type=int, default=5,
                    help="must match train.py so the test set is the same")
    ap.add_argument("--val-games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="clean",
                    help="checkpoints go here, not beside the old ones, so "
                         "nothing is overwritten and the two can be compared")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"

    d = np.load(args.input, allow_pickle=False)
    X, y, games = d["X"], d["y"], d["games"]
    horizon = int(d["horizon"])
    step = float(d["step_seconds"])
    dt = horizon * step

    print(f"Loaded  : {args.input}")
    print(f"X       : {X.shape}")
    print(f"horizon : {horizon} rows ({dt*1000:.1f} ms)\n")

    fit_mask, val_mask, test_mask, val_games, test_games = three_way_split(
        games, args.test_games, args.val_games, args.seed)

    print(f"Three-way split by game (seed {args.seed}):")
    for label, m in (("train", fit_mask), ("val", val_mask),
                     ("test", test_mask)):
        print(f"  {label:<6}{m.sum():>8,} examples  "
              f"{len(np.unique(games[m])):>3} games")
    print(f"  val  games: {val_games.tolist()}")
    print(f"  test games: {test_games.tolist()}   <- touched once, at the end")

    Xtr = torch.from_numpy(X[fit_mask]); ytr = torch.from_numpy(y[fit_mask])
    Xva = torch.from_numpy(X[val_mask]); yva = torch.from_numpy(y[val_mask])

    # The bar on VALIDATION. The test bar is computed later, with the
    # test set, and not before.
    with torch.no_grad():
        va_null, va_dr = baseline_errors(Xva, yva, dt)
    va_dr_med = va_dr.median().item()
    print(f"\nValidation baselines ({dt*1000:.0f} ms ahead):")
    print(f"  null            median {va_null.median():7.2f}")
    print(f"  dead reckoning  median {va_dr_med:7.2f}   <- the bar, on val")

    train_loader = DataLoader(TensorDataset(Xtr, ytr),
                              batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=4096)

    if args.arch == "gru":
        model = GRUPredictor(hidden=args.hidden, layers=args.layers)
        desc = f"GRU hidden={args.hidden} layers={args.layers}"
        tag = f"gru_h{horizon}"
    else:
        # Imported here rather than at module scope so a missing
        # transformer_model.py cannot break the GRU path.
        from transformer_model import TransformerPredictor
        model = TransformerPredictor(d_model=args.d_model,
                                     layers=args.layers, nhead=args.nhead,
                                     ff_mult=args.ff_mult)
        desc = (f"Transformer d_model={args.d_model} layers={args.layers} "
                f"heads={args.nhead} ff={args.d_model*args.ff_mult}")
        tag = f"tfm{args.d_model}_h{horizon}"
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"\nModel   : {desc}  ({n_params:,} parameters)")
    print(f"Training: {args.epochs} epochs, batch {args.batch}, "
          f"lr {args.lr}\n")
    print(f"  {'epoch':>5} {'train loss':>12} {'val median':>11} "
          f"{'vs DR(val)':>11} {'time':>7}")
    print("  " + "-" * 52)

    best_val = float("inf")
    best_state = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        total, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)

        med = median_error(model, val_loader, device).median().item()
        flag = ""
        if med < best_val:
            best_val, best_epoch = med, epoch
            # Held in memory, not written: nothing on disk is chosen by
            # anything the test set has seen.
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            flag = " *"

        print(f"  {epoch:>5} {total/seen:>12.2f} {med:>11.3f} "
              f"{(1 - med/va_dr_med)*100:>10.1f}% "
              f"{time.perf_counter()-t0:>6.1f}s{flag}")

    # -------------------------------------------------------------------
    # Training is over and the epoch is chosen. NOW the test set.
    # -------------------------------------------------------------------
    print(f"\n  best epoch {best_epoch} by validation median "
          f"({best_val:.3f} units)")
    model.load_state_dict(best_state)

    Xte = torch.from_numpy(X[test_mask]); yte = torch.from_numpy(y[test_mask])
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=4096)
    with torch.no_grad():
        te_null, te_dr = baseline_errors(Xte, yte, dt)
    te_errs = median_error(model, test_loader, device)

    te_dr_med = te_dr.median().item()
    te_med = te_errs.median().item()
    gain = (1 - te_med / te_dr_med) * 100

    print("\n" + "=" * 60)
    print(f"  TEST SET, touched once, {len(yte):,} examples "
          f"({dt*1000:.0f} ms ahead)")
    print(f"  {'method':<18}{'median':>9}{'90th':>9}")
    print(f"  {'null (no move)':<18}{te_null.median():>9.2f}"
          f"{te_null.quantile(0.9):>9.2f}")
    print(f"  {'dead reckoning':<18}{te_dr_med:>9.2f}"
          f"{te_dr.quantile(0.9):>9.2f}")
    label = "GRU" if args.arch == "gru" else f"Transformer {args.d_model}"
    print(f"  {label:<18}{te_med:>9.2f}{te_errs.quantile(0.9):>9.2f}")
    print(f"\n  {label} beats dead reckoning by {gain:.1f}% at the median.")
    print(f"  Chosen on validation, measured on test. No epoch was")
    print(f"  selected using this number.")
    print("=" * 60)

    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"{tag}_s{args.seed}.pt")
    torch.save({"state_dict": model.state_dict(),
                "arch": args.arch,
                "hidden": args.hidden, "d_model": args.d_model,
                "nhead": args.nhead, "ff_mult": args.ff_mult,
                "layers": args.layers,
                "n_params": n_params,
                "horizon": horizon, "step_seconds": step,
                "test_games": test_games, "val_games": val_games,
                "best_epoch": best_epoch,
                "val_median": best_val,
                # dr_median is the TEST bar, so gru_conceal.py's ladder
                # report shows the honest gain for this checkpoint.
                "dr_median": te_dr_med,
                "test_median": te_med,
                "selection": "validation games, test touched once"},
               out)
    print(f"\nsaved {out}")

    if args.arch == "transformer":
        print(f"\nCompare against the GRU at this horizon, on these same")
        print(f"test games. {n_params:,} parameters against the GRU's")
        print(f"14,019 -- quote both, so the comparison cannot be read as")
        print(f"one architecture simply having been given more capacity.")
    else:
        print(f"\nCompare with the old checkpoint's reported figure. The")
        print(f"difference is the selection bias the old script carried --")
        print(f"expected to be small, and worth stating either way.")


if __name__ == "__main__":
    main()
