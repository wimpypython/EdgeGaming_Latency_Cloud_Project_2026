"""
Replay real recorded player trajectories over the wire, instead of
generating synthetic motion.

WHY THIS EXISTS
    client.py invents movement: 150-250 u/s, gradual turns, never still.
    Measured against the dataset it was meant to resemble, that is wrong
    in a way that decides results:

        real play (CSKnow)      median speed 100 u/s, 25.3% of frames
                                below 10 u/s -- a quarter of the time the
                                player is standing still
        client.py              median speed ~200 u/s, essentially never
                                stationary

    Dead reckoning is near-optimal on motion that never stops or turns,
    so the synthetic traffic cannot distinguish concealment methods. A
    model trained on real players predicts the slowing that real players
    do, and on synthetic traffic every bit of that is error. The live run
    on 2026-10-13 measured exactly that: GRU 9.14 units against dead
    reckoning's 6.53, a 40% loss, on traffic DR finds trivial.

    This client removes the generator from the question. The positions on
    the wire are recorded human movement, replayed.

WHAT IS AND IS NOT SYNTHESISED
    Nothing is synthesised. The dataset is 16 Hz and this sends at 16 Hz,
    one packet per recorded frame, positions and velocities exactly as
    recorded. Sending at 64 Hz would mean inventing three positions
    between every pair of real frames, so it is not offered.

    The cost is stated rather than hidden: this measures concealment
    accuracy on real motion, at the dataset's own rate. The 64 Hz tick
    realism stays with client.py, where the motion is synthetic. Neither
    run claims to be the other.

HELD-OUT GAMES ONLY
    Replaying games the model trained on would be meaningless -- it has
    seen that movement. The split is imported from baselines.py, the
    same function train.py used, so there is exactly one definition of
    it. If that import fails this script REFUSES to run rather than
    reimplementing the split: a split that disagreed by one game would
    replay training data and every number afterwards would be wrong,
    with nothing to indicate it.

Usage:
    python replay_client.py --npz ../ml_model/parsed_data.npz \\
                            --model-dir ../ml_model --players 1
"""

import argparse
import os
import random
import socket
import sys
import time

import numpy as np

from packet import Packet, encode, END_SEQ, FIRING

HOST = "127.0.0.1"
PORT = 27015

# Gilbert-Elliott loss, simulated at the sender.
#
# WHY AT THE SENDER AND NOT WITH netem
#     netem shapes egress on the EC2 instance and gives one gap every few
#     seconds. Comparing concealment methods needs hundreds of gaps to
#     mean anything: a quarter of real frames are a player standing
#     still, where freezing is almost perfect, so a handful of gaps is
#     decided by which moments they happened to land on. A 6-gap run on
#     2026-10-13 put freeze at 16.9 units against DR's 58.3, while the
#     same data computed over 112,613 windows puts freeze at 61.2 and DR
#     at 33.6 -- the opposite ordering, from sampling alone.
#
# WHY THESE NUMBERS
#     They are not chosen to produce a result. They reproduce the loss
#     the netem run MEASURED: 5.69% overall, 610 ms mean burst. At 16 Hz
#     a 610 ms burst is 9.76 packets, so r = 1/9.76 = 0.1025, and
#     p = r * 0.0569/0.9431 = 0.0062 gives the 5.69%.
#
#     This is a loss MODEL, not netem. Runs using it say so. The netem
#     measurement remains the evidence about the real network.
LOSS_P = 0.0062        # good -> bad, per packet
LOSS_R = 0.1025        # bad -> good, per packet


def load_split(model_dir, npz_games, n_test, seed):
    """Import the project's one copy of the game split.

    Returns (test_mask, test_games). Raises on failure -- see the module
    docstring for why this is not worked around.
    """
    model_dir = os.path.abspath(model_dir)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from baselines import game_split          # deliberately not guarded
    _, test_mask, test_games = game_split(npz_games, n_test=n_test,
                                          seed=seed)
    return test_mask, test_games


def trajectories(X, games, rounds, players, mask):
    """Rebuild continuous per-player trajectories from the windows.

    Consecutive rows of X within one (game, round, player) are the same
    window advanced by exactly one frame, so a trajectory is the first
    row's 16 frames followed by the last frame of each row after it.
    That is verified here rather than assumed, because if the rows were
    not contiguous this would silently stitch together motion that never
    happened.
    """
    idx = np.where(mask)[0]
    pcodes = {s: i for i, s in enumerate(sorted(set(players.tolist())))}
    pi = np.array([pcodes[s] for s in players], dtype=np.int64)
    key = (games.astype(np.int64) << 40) + (rounds.astype(np.int64) << 20) + pi

    out = []
    for k in np.unique(key[idx]):
        rows = np.where((key == k) & mask)[0]
        if len(rows) < 2:
            continue
        if not np.all(np.diff(rows) == 1):
            continue                     # not contiguous: skip, not stitch
        # Verify the one-frame shift on this trajectory before using it.
        if not np.allclose(X[rows[0], 1:], X[rows[0] + 1, :-1], atol=1e-3):
            continue
        frames = [X[rows[0], i] for i in range(X.shape[1])]
        for r in rows[1:]:
            frames.append(X[r, -1])
        out.append(np.asarray(frames, dtype=np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="../ml_model/parsed_data.npz")
    ap.add_argument("--model-dir", default="../ml_model",
                    help="where baselines.py lives, for the game split")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--players", type=int, default=1,
                    help="how many trajectories to replay at once, each "
                         "with its own sequence space")
    ap.add_argument("--test-games", type=int, default=5,
                    help="must match what train.py used")
    ap.add_argument("--seed", type=int, default=0,
                    help="must match what train.py used, or the split "
                         "differs and training data gets replayed")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 means replay the whole trajectory")
    ap.add_argument("--pick", type=int, default=0,
                    help="index into the held-out trajectory list, so a "
                         "run can be repeated exactly")
    ap.add_argument("--loss", action="store_true",
                    help="simulate Gilbert-Elliott burst loss at the "
                         "sender, with the parameters the netem run "
                         "measured. Off by default")
    ap.add_argument("--loss-p", type=float, default=LOSS_P)
    ap.add_argument("--loss-r", type=float, default=LOSS_R)
    ap.add_argument("--loss-seed", type=int, default=0,
                    help="same seed gives the same loss pattern, so two "
                         "concealment methods can be compared on "
                         "identical gaps")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    X = d["X"]
    step = float(d["step_seconds"])
    rate = 1.0 / step

    test_mask, test_games = load_split(args.model_dir, d["games"],
                                       args.test_games, args.seed)
    trajs = trajectories(X, d["games"], d["rounds"], d["players"],
                         test_mask)

    print(f"dataset   : {args.npz}")
    print(f"rate      : {rate:.3f} Hz ({step*1000:.3f} ms per frame) "
          f"-- the dataset's own rate, nothing resampled")
    print(f"held-out  : games {sorted(test_games.tolist())} "
          f"({int(test_mask.sum()):,} windows)")
    print(f"usable    : {len(trajs)} continuous trajectories")

    if not trajs:
        print("no usable trajectories -- check --seed and --test-games "
              "match train.py")
        return

    # Each slot works through its own queue of DISTINCT trajectories.
    #
    # A long run needs more motion than one 20-second trajectory holds,
    # and replaying the same one twice would resample the same movement
    # rather than add evidence. Looping also teleports the player back to
    # the start, which dead reckoning would score as a colossal error at
    # every seam.
    #
    # So each new trajectory is sent under a NEW player id with a fresh
    # sequence space. The server treats it as a new player arriving,
    # which it is, and no seam is ever scored as a gap.
    order = [(args.pick + i) % len(trajs) for i in range(len(trajs))]
    queue = list(order)

    slots = []          # [pid, traj, frame_index, seq]
    next_pid = 0
    for _ in range(min(args.players, len(queue))):
        slots.append([next_pid, trajs[queue.pop(0)], 0, 0])
        next_pid += 1

    for pid, tr, _, _ in slots:
        sp = np.linalg.norm(tr[:, 3:6], axis=1)
        print(f"  slot {pid}: {len(tr):>4} frames = "
              f"{len(tr)*step:6.1f}s   median {np.median(sp):6.1f} u/s   "
              f"still {float((sp < 10).mean())*100:4.1f}% of frames")
    print(f"  then rotating through {len(queue)} further held-out "
          f"trajectories as each finishes")

    total_ticks = int(args.seconds / step) if args.seconds else None
    if total_ticks is None:
        total_ticks = sum(len(t) for t in trajs) // max(1, len(slots))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"\nreplaying to {args.host}:{args.port} for "
          f"{total_ticks*step:.1f}s -- run the server with --rate "
          f"{rate:.0f}\n")

    sent = 0
    dropped = 0
    retired = 0
    rngs = {}
    bad = {}
    if args.loss:
        exp_burst = 1.0 / args.loss_r * step
        exp_loss = args.loss_p / (args.loss_p + args.loss_r) * 100
        print(f"simulated burst loss at the sender: p={args.loss_p:.4f} "
              f"r={args.loss_r:.4f}")
        print(f"  expected {exp_loss:.2f}% loss, "
              f"{exp_burst*1000:.0f} ms mean burst "
              f"(netem measured 5.69% / 610 ms)")
    t0 = time.perf_counter()

    try:
        for n in range(total_ticks):
            for slot in slots:
                pid, tr, fi, seq = slot
                if fi >= len(tr):
                    # Retire this trajectory and start the next one under
                    # a new id, so the position discontinuity is never
                    # part of any player's trajectory.
                    #
                    # NOT by sending END_SEQ: the server treats that as
                    # "the run is over" and stops receiving, whichever
                    # player it came from. Sending one per retirement
                    # ended the whole run at the first short trajectory.
                    # A retired id simply stops being sent to.
                    retired += 1
                    if not queue:
                        slot[1] = None
                        continue
                    slot[0] = next_pid
                    next_pid += 1
                    slot[1] = trajs[queue.pop(0)]
                    slot[2] = 0
                    slot[3] = 0
                    pid, tr, fi, seq = slot
                if tr is None:
                    continue

                f = tr[fi]
                seq += 1
                slot[2] = fi + 1
                slot[3] = seq

                if args.loss:
                    if pid not in rngs:
                        rngs[pid] = random.Random(args.loss_seed * 1000 + pid)
                        bad[pid] = False
                    if bad[pid]:
                        if rngs[pid].random() < args.loss_r:
                            bad[pid] = False
                    elif rngs[pid].random() < args.loss_p:
                        bad[pid] = True
                    if bad[pid]:
                        dropped += 1
                        continue

                pkt = Packet(seq=seq, player_id=pid,
                             pos=(float(f[0]), float(f[1]), float(f[2])),
                             vel=(float(f[3]), float(f[4]), float(f[5])),
                             view=(0.0, 0.0), flags=0)
                sock.sendto(encode(pkt), (args.host, args.port))
                sent += 1

            if all(s[1] is None for s in slots):
                break

            target = t0 + (n + 1) * step
            slack = target - time.perf_counter()
            if slack > 0:
                time.sleep(slack)

            if not args.quiet and n % int(rate) == 0:
                print(f"\r  t={n*step:6.1f}s  sent {sent:,}  "
                      f"dropped {dropped:,}  trajectories used "
                      f"{next_pid}", end="", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted")

    elapsed = time.perf_counter() - t0
    for slot in slots:
        if slot[1] is not None:
            sock.sendto(encode(Packet(seq=END_SEQ, player_id=slot[0])),
                        (args.host, args.port))

    if args.loss:
        tot = sent + dropped
        print(f"\n  dropped {dropped:,} of {tot:,} "
              f"({dropped/tot*100:.2f}%) at the sender")
    print(f"\n\nsent {sent:,} packets in {elapsed:.2f}s "
          f"({sent/elapsed:.1f} pkt/s across {next_pid} trajectories)")
    print("every position on the wire was a recorded frame; none were "
          "generated or interpolated")
    if args.loss:
        print("loss was simulated at the sender with the netem-measured "
              "parameters; this run is not a netem run")


if __name__ == "__main__":
    main()
