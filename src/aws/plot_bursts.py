#!/usr/bin/env python3
"""
Plot the burst-length distribution from a verify_netem.py capture.

WHY
    verify_netem.py prints the run-length histogram as ASCII hashes, which
    is fine for checking at the terminal but not something you can put in
    front of Dr Priya. This reads the same numbers out of the saved output
    and draws them.

    It PARSES the file rather than having the numbers typed in, so a new
    capture needs no code edits, and there is no chance of the plot and the
    measurement disagreeing because someone transcribed a digit wrong.

WHAT IT READS
    The histogram block verify_netem.py prints, whose lines look like:

           2 pkt (   31 ms) x1     #
          19 pkt (  297 ms) x2     ##

    meaning "a 2-packet outage happened once", "a 19-packet outage
    happened twice". Everything else in the file is ignored.

USAGE
    python plot_bursts.py burst500_long.txt
    python plot_bursts.py burst500_long.txt --rate 64 --out my_plot.png

    --rate must match the rate the capture ran at (default 64 Hz, the
    CS:GO update rate). It converts packets to milliseconds; get it wrong
    and every number on the x-axis is wrong by that ratio.
"""

import argparse
import re
import sys

import matplotlib
matplotlib.use("Agg")           # no display on a server; write a file
import matplotlib.pyplot as plt
import numpy as np

# "   19 pkt (  297 ms) x2    ##"  ->  length 19, count 2
HIST_LINE = re.compile(r"^\s*(\d+)\s+pkt\s*\(\s*\d+\s*ms\)\s*x(\d+)")

# dataviz reference palette, light mode
SERIES_1 = "#2a78d6"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# The band the ML results actually care about: below ~250 ms dead
# reckoning is already within 2 cm, above ~750 ms everything is bad.
BAND_LO, BAND_HI = 250.0, 750.0


def parse(path):
    """Return one entry per loss EVENT, in packets. A histogram line with
    x3 contributes three events, because the statistics below are over
    events, not over distinct lengths."""
    runs = []
    with open(path) as fh:
        for line in fh:
            m = HIST_LINE.match(line)
            if m:
                length, count = int(m.group(1)), int(m.group(2))
                runs.extend([length] * count)
    return np.array(runs, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", help="saved verify_netem.py output")
    ap.add_argument("--rate", type=float, default=64.0,
                    help="packets per second the capture ran at (default 64)")
    ap.add_argument("--out", default="burst_distribution.png")
    ap.add_argument("--bin", type=float, default=125.0,
                    help="bin width in ms (default 125)")
    args = ap.parse_args()

    runs = parse(args.capture)
    if runs.size == 0:
        print(f"no histogram lines found in {args.capture}")
        print("Either the run had zero loss, or this is not a")
        print("verify_netem.py output file.")
        sys.exit(1)

    ms_per_pkt = 1000.0 / args.rate
    ms = runs * ms_per_pkt

    # Print the same numbers the plot shows, so the two can be checked
    # against each other and against the original capture.
    print(f"events {len(ms)}   packets lost {int(runs.sum())}")
    print(f"mean {ms.mean():.0f} ms   median {np.median(ms):.0f} ms   "
          f"max {ms.max():.0f} ms")
    in_band = int(((ms >= BAND_LO) & (ms <= BAND_HI)).sum())
    print(f"in {BAND_LO:.0f}-{BAND_HI:.0f} ms band: {in_band}/{len(ms)}")

    edges = np.arange(0, ms.max() + args.bin, args.bin)
    counts, _ = np.histogram(ms, bins=edges)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # The band goes down first so the bars sit on top of it. Its label sits
    # ABOVE the axes, not inside them -- inside, it collides with whichever
    # bar label happens to be tallest in that range.
    ax.axvspan(BAND_LO, BAND_HI, color=SERIES_1, alpha=0.08, zorder=0)
    ax.annotate("250-750 ms: the horizons the model was measured on",
                xy=((BAND_LO + BAND_HI) / 2, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color=INK_2)

    # How much of the distribution falls past everything we tested. This is
    # the honest limitation, so it goes on the chart rather than only in the
    # terminal output.
    over = int((ms > BAND_HI).sum())
    if over:
        ax.annotate(f"{over} of {len(ms)} events run longer than 750 ms,\n"
                    f"beyond any horizon tested",
                    xy=(0.985, 0.97), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9, color=INK_2)

    # 2px surface gap between adjacent bars, per the mark spec.
    ax.bar(edges[:-1], counts, width=args.bin, align="edge",
           color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=2)

    # Direct labels, only on bars that have something in them.
    for x, c in zip(edges[:-1], counts):
        if c:
            ax.text(x + args.bin / 2, c + 0.08, str(c), ha="center",
                    va="bottom", fontsize=9, color=INK_2, zorder=3)

    ax.axvline(ms.mean(), color=INK_2, linestyle="--", linewidth=1.5,
               zorder=4)
    ax.text(ms.mean() + 12, counts.max() * 0.62,
            f"mean {ms.mean():.0f} ms", fontsize=9, color=INK_2, zorder=4)

    ax.set_title(f"Outage lengths under Gilbert-Elliott burst loss "
                 f"({len(ms)} events, 300 s at 64 Hz)",
                 fontsize=13, color=INK, pad=26, loc="left")
    ax.set_xlabel("outage length (ms)", fontsize=10, color=INK_2)
    ax.set_ylabel("number of events", fontsize=10, color=INK_2)

    ax.grid(axis="y", color=GRID, linewidth=1, zorder=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylim(0, counts.max() * 1.18)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
