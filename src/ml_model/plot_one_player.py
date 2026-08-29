"""
Plot one player's movement path from the CSKnow data.

The point of this script is to make the data feel real. It pulls the X and Y
position of a single player across a single round and draws the path they
walked. You should end up looking at a person moving around Dust2.

Usage:
    python plot_one_player.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

PATH = r"C:\Users\Atharva\Downloads\sample_csknow\all_train_outputs\behaviorTreeTeamFeatureStore_28.hdf5"

# Which player to look at. Team is "CT" or "T", slot is 0-4.
TEAM = "CT"
SLOT = 0


def main():
    with h5py.File(PATH, "r") as f:
        # Every row belongs to some round. Pick the first round in the file.
        round_ids = f["data/round id"][:]
        target_round = round_ids[0]
        rows = np.where(round_ids == target_round)[0]

        print(f"Round {target_round} has {len(rows)} rows")
        print(f"At 16 rows per second, that's about {len(rows) / 16:.1f} seconds\n")

        start, stop = rows[0], rows[-1] + 1

        # Pull this player's position for every row in the round.
        x = f[f"data/player pos {TEAM} {SLOT} x"][start:stop]
        y = f[f"data/player pos {TEAM} {SLOT} y"][start:stop]

        # A player who is dead or not yet spawned has no meaningful position,
        # so drop rows where they aren't alive.
        alive = f[f"data/alive {TEAM} {SLOT}"][start:stop]
        x, y = x[alive], y[alive]

        print(f"Player {TEAM} {SLOT} was alive for {alive.sum()} of {len(alive)} rows")

        if len(x) < 2:
            print("\nThis player was barely alive this round. Try a different")
            print("SLOT (0-4) or TEAM ('CT' / 'T') at the top of this file.")
            return

        # How far did they actually travel, step to step?
        steps = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        print(f"\nDistance moved per step (62.5ms):")
        print(f"  median : {np.median(steps):7.1f} game units")
        print(f"  max    : {steps.max():7.1f} game units")
        print(f"  total  : {steps.sum():7.1f} game units over the round")

    # Draw it.
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, linewidth=1.5, alpha=0.8)
    plt.scatter(x[0], y[0], s=120, marker="o", label="start", zorder=5)
    plt.scatter(x[-1], y[-1], s=120, marker="X", label="end", zorder=5)

    plt.title(f"Player {TEAM} {SLOT}, round {target_round}\n"
              f"{len(x)} positions at 16 Hz")
    plt.xlabel("map X")
    plt.ylabel("map Y")
    plt.legend()
    plt.axis("equal")   # so the map isn't stretched
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
