#!/usr/bin/env python3
"""
Game client -- simulates players and sends their state as EGP1 packets.

WHY THIS EXISTS
    The project needs traffic that behaves like a game. Everything
    downstream consumes it: the server reconstructs state from it, the XDP
    classifier sorts it, the dashboard draws it, and the prediction model
    is judged on it.

    It would be easier to send random numbers. It would also be useless.
    If simulated players teleport or jitter, dead reckoning looks terrible
    and any model looks brilliant by comparison, and the whole result is an
    artefact of bad test data. So the motion here is constrained to match
    what was measured in the real CSKnow data.

MOVEMENT MODEL
    Each player walks a smooth path: a heading that turns gradually, a
    speed that drifts within plausible bounds, and a bounce off the map
    edges. Specifically:

      speed      150-250 units/second. CS:GO ground movement tops out
                 around 250 (knife out) and ~215 with a rifle. The real
                 data's median step was 6.13 units per 62.5 ms frame,
                 which is ~98 u/s, so most of the time players are well
                 under the cap -- as they are here.

      turn rate  up to ~90 degrees/second. Fast enough to be interesting
                 for a predictor, slow enough to be physically possible.

    Velocity is sent in units per SECOND, never per tick. The ambiguity
    between those two cost this project a week once already.

INTENT FLAGS
    Roughly every 2 seconds a player fires, and occasionally jumps or
    crouches. These set the flags byte that the XDP classifier reads. They
    are what makes classification meaningful -- a position update can be
    interpolated through, a missed shot cannot.

TESTING IT
    Against the already-validated receiver, which reports loss by
    sequence gap:

        python3 ../aws/verify_netem.py --listen
        python3 client.py --seconds 10

    Expect 100% delivery. Any loss with netem off means a bug here.

    Note: that receiver tracks ONE sequence space, so test it with the
    default single player. Multi-player is for the real server.
"""

import argparse
import math
import random
import socket
import time

from packet import Packet, encode, END_SEQ, FIRING, JUMPING, CROUCHING

PORT = 27015
TICK_HZ = 64.0

# Map bounds in game units. de_dust2 is roughly 4000 units across; this is
# a plain square because the geometry does not matter, only the motion.
MAP_MIN, MAP_MAX = -2000.0, 2000.0

# Speed envelope, units per second. See the module docstring for why.
SPEED_MIN, SPEED_MAX = 150.0, 250.0

MAX_TURN_DEG_PER_SEC = 90.0
MAX_ACCEL = 300.0           # units/second^2, how fast speed may change


class Player:
    """One simulated player walking a smooth path."""

    def __init__(self, player_id, rng):
        self.id = player_id
        self.rng = rng
        self.x = rng.uniform(MAP_MIN + 200, MAP_MAX - 200)
        self.y = rng.uniform(MAP_MIN + 200, MAP_MAX - 200)
        self.z = 64.0                       # eye height, constant: flat map
        self.heading = rng.uniform(0, 2 * math.pi)
        self.speed = rng.uniform(SPEED_MIN, SPEED_MAX)
        self.pitch = 0.0
        self.next_fire = rng.uniform(0.5, 3.0)
        self.firing_left = 0.0      # seconds remaining in the current burst

        # Velocity is stored, not recomputed at send time, so that what is
        # transmitted is exactly what moved the player. If these two ever
        # disagree, dead reckoning is being fed a lie and every error
        # measurement downstream is wrong.
        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed

    def step(self, dt):
        """Advance one tick. dt is seconds."""
        r = self.rng

        # Turn gradually. Scaling by dt keeps the motion identical whether
        # the tick rate is 64 Hz or 16 Hz -- otherwise changing --rate
        # would silently change how sharply players turn, and the horizon
        # comparisons would not be comparable across rates.
        max_turn = math.radians(MAX_TURN_DEG_PER_SEC) * dt
        self.heading += r.uniform(-max_turn, max_turn)

        self.speed += r.uniform(-MAX_ACCEL, MAX_ACCEL) * dt
        self.speed = max(SPEED_MIN, min(SPEED_MAX, self.speed))

        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off the walls. Reflecting the heading rather than
        # clamping the position keeps velocity consistent with the motion;
        # clamping would make the player appear to stop dead while still
        # reporting a velocity, which is precisely the inconsistency the
        # comment above warns about.
        if self.x < MAP_MIN or self.x > MAP_MAX:
            self.x = max(MAP_MIN, min(MAP_MAX, self.x))
            self.heading = math.pi - self.heading
        if self.y < MAP_MIN or self.y > MAP_MAX:
            self.y = max(MAP_MIN, min(MAP_MAX, self.y))
            self.heading = -self.heading

        self.vx = math.cos(self.heading) * self.speed
        self.vy = math.sin(self.heading) * self.speed

        # View direction follows movement, with a little drift, which is
        # roughly what a player does.
        self.pitch = max(-89.0, min(89.0, self.pitch + r.uniform(-2, 2)))

    def flags(self, dt):
        """Intent bits for this tick.

        Firing is a BURST, not an instant. A player holds the trigger for
        a fraction of a second, so FIRING stays set across consecutive
        ticks. Modelling it as a single tick would be wrong twice over:
        it is not how the game behaves, and it would leave the XDP
        classifier with well under 1% of packets to classify, which makes
        the classifier impossible to demonstrate or measure.

        A 0.15-0.4 s burst every 0.5-3 s puts intent traffic at roughly
        10-20% of packets, which matches the rough shape of a real match.
        """
        f = 0

        if self.firing_left > 0:
            self.firing_left -= dt
            f |= FIRING
        else:
            self.next_fire -= dt
            if self.next_fire <= 0:
                self.firing_left = self.rng.uniform(0.15, 0.40)
                self.next_fire = self.rng.uniform(0.5, 3.0)
                f |= FIRING
        if self.rng.random() < 0.004:
            f |= JUMPING
        if self.rng.random() < 0.01:
            f |= CROUCHING
        return f

    def to_packet(self, seq, flags):
        return Packet(
            seq=seq,
            player_id=self.id,
            pos=(self.x, self.y, self.z),
            vel=(self.vx, self.vy, 0.0),
            view=(self.pitch, math.degrees(self.heading) % 360.0),
            flags=flags,
        )


def run(host, port, n_players, rate_hz, seconds, seed, quiet):
    rng = random.Random(seed)
    players = [Player(i, rng) for i in range(n_players)]
    dt = 1.0 / rate_hz
    ticks = int(rate_hz * seconds)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"{n_players} player(s) -> {host}:{port} at {rate_hz:g} Hz "
          f"for {seconds:g}s")
    print(f"{ticks:,} ticks, {ticks * n_players:,} packets total\n")

    seq = 0
    sent = 0
    intent = 0
    start = time.perf_counter()

    for _ in range(ticks):
        for p in players:
            p.step(dt)
            f = p.flags(dt)
            if f:
                intent += 1
            sock.sendto(encode(p.to_packet(seq, f)), (host, port))
            seq += 1
            sent += 1

        # Busy-wait the last millisecond. sleep() alone is not accurate
        # enough at 15.6 ms intervals -- the OS routinely overshoots by
        # several milliseconds, which would make the send rate lower than
        # requested and quietly bias any timing measurement.
        target = start + (seq / n_players) * dt
        while True:
            left = target - time.perf_counter()
            if left <= 0:
                break
            if left > 0.002:
                time.sleep(left - 0.001)

    # Trailer so the receiver knows to stop and print. Sent five times in
    # case it is itself dropped -- it is the one packet whose loss would
    # leave the receiver hanging.
    for _ in range(5):
        sock.sendto(encode(Packet(seq=END_SEQ)), (host, port))
        time.sleep(0.01)

    elapsed = time.perf_counter() - start
    print(f"sent {sent:,} packets in {elapsed:.2f}s "
          f"(actual {sent/elapsed:.1f} pkt/s, "
          f"{sent/elapsed/n_players:.1f} Hz per player)")
    print(f"{intent:,} carried an intent flag "
          f"({intent/sent*100:.1f}%)")

    if not quiet:
        p = players[0]
        sp = math.hypot(p.vx, p.vy)
        print(f"\nplayer 0 final: pos ({p.x:.1f}, {p.y:.1f})  "
              f"speed {sp:.1f} u/s")
        if not (SPEED_MIN - 1 <= sp <= SPEED_MAX + 1):
            print(f"  WARNING: speed outside the {SPEED_MIN}-{SPEED_MAX} "
                  f"envelope -- movement model is wrong")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--players", type=int, default=1,
                    help="default 1, so verify_netem.py's single sequence "
                         "space stays valid")
    ap.add_argument("--rate", type=float, default=TICK_HZ)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="same seed gives the same run, so a bug is "
                         "reproducible")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run(args.host, args.port, args.players, args.rate, args.seconds,
        args.seed, args.quiet)


if __name__ == "__main__":
    main()
