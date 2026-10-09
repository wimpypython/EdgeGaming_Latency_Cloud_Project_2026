#!/usr/bin/env python3
"""
Authoritative game server -- receives EGP1, tracks players, conceals loss.

WHAT IT IS
    The receiving half of the system. It holds what the server believes
    is true about every player, and when packets stop arriving it has to
    keep believing something. That "something" is the whole project.

WHAT IT DOES
    1. Receives and validates EGP1 packets (via packet.py -- one format
       definition, not a second copy).
    2. Tracks per-player state: position, velocity, view, flags, seq.
    3. Detects loss by sequence gap.
    4. Conceals the gap with one of three methods.
    5. MEASURES ITS OWN ERROR.

THE MEASUREMENT -- the reason this file matters
    When a gap ends, the next real packet carries the TRUE position. So
    the server can score what it was guessing:

        predicted = last_pos + last_vel * gap_seconds     (dead reckoning)
        frozen    = last_pos                              (do nothing)
        error     = distance(prediction, true position)

    That is the same quantity as the offline number #2, measured on live
    traffic over a real network instead of on a stored dataset. Both
    methods are scored on the SAME gaps, so the comparison is honest by
    construction -- there is no way for one method to get easier cases.

    When the GRU is hooked in it becomes a third column and nothing else
    about this file changes.

THE OFF-BY-ONE THAT MATTERS
    If the last seq was 400 and the next is 412, the missing packets are
    401..411 -- that is ELEVEN, not twelve:

        missing = new_seq - last_seq - 1

    Getting this wrong inflates every outage length and every error
    measurement derived from it. It is the same class of silent mistake
    as the Week 1 parser bug: no crash, no warning, just wrong numbers.
    The self-test at the bottom checks it.

WHY THE LOCAL CLOCK, NOT send_ns
    Gap duration is measured with this machine's monotonic clock, not by
    subtracting the senders' timestamps. The sender and receiver clocks
    are not synchronised -- verify_netem.py says as much about its own
    one-way delay figure -- so a difference between their timestamps
    carries unknown clock skew. Monotonic time on one machine has none.

RUN IT
    python3 server_sync.py
    python3 server_sync.py --conceal freeze
    python3 server_sync.py --self-test       # no network, checks the maths

THEN, from the sending machine:
    python client.py --host <this machine> --players 3 --seconds 30
"""

import argparse
import collections
import json
import math
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from packet import decode, is_egp1, END_SEQ, BadPacket, INTENT_MASK

PORT = 27015
DEFAULT_RATE = 64.0

# A player is considered "in an outage" once this many tick intervals
# have passed with nothing from them. 2.5 rather than 1 because packets
# arrive slightly irregularly even on a clean link -- jitter alone would
# otherwise flip players in and out of concealment constantly.
OUTAGE_AFTER_TICKS = 2.5


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 +
                     (a[2] - b[2]) ** 2)


def dead_reckon(pos, vel, seconds):
    """pos + vel * dt. Velocity is units per SECOND -- see packet.py.

    This is what real netcode does, and it is the baseline the whole
    project is measured against."""
    return (pos[0] + vel[0] * seconds,
            pos[1] + vel[1] * seconds,
            pos[2] + vel[2] * seconds)


class PlayerState:
    """What the server believes about one player."""

    def __init__(self, player_id, now):
        self.id = player_id
        self.pos = (0.0, 0.0, 0.0)
        self.vel = (0.0, 0.0, 0.0)
        self.view = (0.0, 0.0)
        self.flags = 0
        self.seq = None               # None until the first packet
        self.last_real = now          # monotonic time of last real packet

        self.received = 0
        self.missing = 0              # packets never seen, by sequence
        self.outages = []             # (packets, seconds) per gap
        self.errors_dr = []           # concealment error, units
        self.errors_freeze = []
        self.intent_packets = 0

    def apply(self, pkt, now):
        """Take a real packet. Returns (missing_count, gap_seconds) if
        this packet closed a gap, else None."""
        gap = None

        if self.seq is not None and pkt.seq > self.seq:
            # THE off-by-one. 400 -> 412 means 401..411 are gone: 11.
            missing = pkt.seq - self.seq - 1
            if missing > 0:
                elapsed = now - self.last_real
                self.missing += missing
                self.outages.append((missing, elapsed))

                # Score both concealment methods against ground truth,
                # which is the position in THIS packet.
                predicted = dead_reckon(self.pos, self.vel, elapsed)
                self.errors_dr.append(dist(predicted, pkt.pos))
                self.errors_freeze.append(dist(self.pos, pkt.pos))

                gap = (missing, elapsed)

        # A packet older than what we have is a reorder, not an update.
        # UDP does not guarantee order. Applying it would move the player
        # backwards in time.
        if self.seq is not None and pkt.seq <= self.seq:
            return None

        self.pos = pkt.pos
        self.vel = pkt.vel
        self.view = pkt.view
        self.flags = pkt.flags
        self.seq = pkt.seq
        self.last_real = now
        self.received += 1
        if pkt.flags & INTENT_MASK:
            self.intent_packets += 1
        return gap

    def believed_position(self, now, method):
        """Where the server currently thinks this player is.

        During an outage this is a guess. That guess is the product."""
        elapsed = now - self.last_real
        if method == "freeze":
            return self.pos
        if method == "dr":
            return dead_reckon(self.pos, self.vel, elapsed)
        raise ValueError(method)

    def is_concealing(self, now, interval):
        return (now - self.last_real) > interval * OUTAGE_AFTER_TICKS


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = int(round((len(s) - 1) * p))
    return s[k]


def report(players, method, rate):
    interval_ms = 1000.0 / rate
    print()
    print("=" * 66)

    tot_recv = sum(p.received for p in players.values())
    tot_miss = sum(p.missing for p in players.values())
    expected = tot_recv + tot_miss

    print(f"  players seen   : {len(players)}")
    print(f"  received       : {tot_recv:,}")
    print(f"  missing by seq : {tot_miss:,}", end="")
    if expected:
        print(f"  ({tot_miss / expected * 100:.2f}%)")
    else:
        print()

    outages = [o for p in players.values() for o in p.outages]
    if not outages:
        print("\n  No gaps. Nothing was concealed, so there is no error to")
        print("  report. Apply netem loss to make this interesting.")
        print("=" * 66)
        return

    lens = [n for n, _ in outages]
    secs = [s for _, s in outages]
    print(f"\n  outage events  : {len(outages)}")
    print(f"  mean gap       : {sum(lens)/len(lens):.1f} packets "
          f"= {sum(secs)/len(secs)*1000:.0f} ms")
    print(f"  longest gap    : {max(lens)} packets "
          f"= {max(secs)*1000:.0f} ms")

    dr = [e for p in players.values() for e in p.errors_dr]
    fz = [e for p in players.values() for e in p.errors_freeze]

    print(f"\n  CONCEALMENT ERROR over {len(dr)} gaps, game units")
    print(f"  {'method':<16}{'median':>10}{'90th':>10}{'max':>10}")
    print(f"  {'freeze':<16}{percentile(fz,0.5):>10.2f}"
          f"{percentile(fz,0.9):>10.2f}{max(fz):>10.2f}")
    print(f"  {'dead reckoning':<16}{percentile(dr,0.5):>10.2f}"
          f"{percentile(dr,0.9):>10.2f}{max(dr):>10.2f}")

    med_f, med_d = percentile(fz, 0.5), percentile(dr, 0.5)
    if med_f > 0:
        gain = (med_f - med_d) / med_f * 100
        print(f"\n  Dead reckoning beats freeze by {gain:.1f}% at the median.")
        if gain < 0:
            print("  NEGATIVE -- dead reckoning is doing worse than doing")
            print("  nothing. That means velocity is being misread, or the")
            print("  gap duration is wrong. Check units before believing it.")

    print("\n  Same gaps scored both ways, so neither method got easier")
    print("  cases. The GRU slots in here as a third row.")
    print("=" * 66)


class World:
    """Everything the receive loop owns and the HTTP thread reads.

    The two run concurrently, so every access goes through `lock`.
    Without it the dashboard can read a player mid-update and render a
    position whose x is new and whose y is old -- a rare, intermittent
    jump that looks like a physics bug and is miserable to track down.
    """

    def __init__(self, rate, method):
        self.lock = threading.Lock()
        self.players = {}
        self.rate = rate
        self.interval = 1.0 / rate
        self.method = method
        self.not_ours = 0
        self.started = None

        # Packets arriving before this monotonic time are discarded.
        # The dashboard sets it to stage an outage on demand.
        #
        # Discarded, not special-cased: loss is detected by sequence
        # gap, so an injected drop is indistinguishable from a real one
        # and flows through exactly the same gap arithmetic, the same
        # concealment and the same error measurement. There is no
        # separate code path that could behave differently in the demo
        # than it does under netem.
        self.drop_until = 0.0

    def snapshot(self, now):
        """JSON-safe view of the world. Called from the HTTP thread."""
        with self.lock:
            players = []
            for pid in sorted(self.players):
                p = self.players[pid]
                bx, by, bz = p.believed_position(now, self.method)
                players.append({
                    "id": pid,
                    "believed": [round(bx, 2), round(by, 2), round(bz, 2)],
                    "last_real": [round(v, 2) for v in p.pos],
                    "vel": [round(v, 2) for v in p.vel],
                    "view": [round(v, 2) for v in p.view],
                    "flags": p.flags,
                    "intent": bool(p.flags & INTENT_MASK),
                    "seq": p.seq,
                    "concealing": p.is_concealing(now, self.interval),
                    "concealed_for_ms": round(
                        max(0.0, now - p.last_real) * 1000.0, 1),
                    "received": p.received,
                    "missing": p.missing,
                })

            dr = [e for p in self.players.values() for e in p.errors_dr]
            fz = [e for p in self.players.values() for e in p.errors_freeze]
            recv = sum(p.received for p in self.players.values())
            miss = sum(p.missing for p in self.players.values())
            events = sum(len(p.outages) for p in self.players.values())

            return {
                "method": self.method,
                "rate": self.rate,
                "dropping": now < self.drop_until,
                "drop_remaining_ms": round(
                    max(0.0, self.drop_until - now) * 1000.0),
                "players": players,
                "stats": {
                    "received": recv,
                    "missing": miss,
                    "loss_pct": round(miss / (recv + miss) * 100, 2)
                                if recv + miss else 0.0,
                    "outage_events": events,
                    "error_median_dr": round(percentile(dr, 0.5), 2)
                                       if dr else None,
                    "error_median_freeze": round(percentile(fz, 0.5), 2)
                                           if fz else None,
                    "error_p90_dr": round(percentile(dr, 0.9), 2)
                                    if dr else None,
                    "error_p90_freeze": round(percentile(fz, 0.9), 2)
                                        if fz else None,
                },
            }


def make_handler(world):
    class Handler(BaseHTTPRequestHandler):

        def _send(self, payload, code=200):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            # The dashboard may be opened as a local file, which makes
            # its origin "null". Without this the browser blocks every
            # fetch and the page sits empty with only a console error.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)

            if u.path in ("/state", "/"):
                self._send(world.snapshot(time.monotonic()))

            elif u.path == "/conceal":
                m = (q.get("method") or [""])[0]
                if m not in ("dr", "freeze"):
                    self._send({"error": "method must be dr or freeze"}, 400)
                    return
                with world.lock:
                    world.method = m
                self._send({"method": m})

            elif u.path == "/inject":
                try:
                    ms = float((q.get("ms") or ["500"])[0])
                except ValueError:
                    self._send({"error": "ms must be a number"}, 400)
                    return
                # Capped: an unbounded value from a query string would
                # silently wedge the server with no way back.
                ms = max(0.0, min(5000.0, ms))
                with world.lock:
                    world.drop_until = time.monotonic() + ms / 1000.0
                self._send({"dropping_for_ms": ms})

            else:
                self._send({"error": "not found"}, 404)

        def log_message(self, *args):
            pass        # polling a few times a second would flood stdout

    return Handler


def start_http(world, port):
    srv = ThreadingHTTPServer(("", port), make_handler(world))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv


def serve(port, rate, method, seconds, quiet, http_port):
    world = World(rate, method)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    sock.settimeout(1.0)

    print(f"listening on :{port}, concealing with '{method}'")
    if http_port:
        start_http(world, http_port)
        print(f"state endpoint on http://0.0.0.0:{http_port}/state")
        print(f"  toggle   : /conceal?method=dr|freeze")
        print(f"  outage   : /inject?ms=500")
    print("waiting for packets (Ctrl-C to stop)")
    print()

    last_draw = 0.0
    try:
        while True:
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                if (world.started and seconds
                        and time.monotonic() - world.started >= seconds):
                    break
                continue

            # Read the clock AFTER recvfrom returns, never before.
            #
            # recvfrom blocks for however long the gap lasts. A timestamp
            # taken before it is stale by exactly the quantity we are
            # trying to measure, so every outage would come out as one
            # loop iteration (~16 ms) no matter how long it really was --
            # and dead reckoning would be scored on a 16 ms extrapolation
            # instead of a 400 ms one. Both the outage lengths and the
            # concealment comparison would be wrong, with nothing to
            # indicate it.
            now = time.monotonic()

            if not is_egp1(data):
                world.not_ours += 1
                continue
            try:
                pkt = decode(data)
            except BadPacket:
                world.not_ours += 1
                continue

            if pkt.seq == END_SEQ:
                break

            # Injected outage: drop it on the floor. The sequence number
            # is never recorded, so the next accepted packet registers as
            # a gap through the normal path.
            if now < world.drop_until:
                continue

            with world.lock:
                if world.started is None:
                    world.started = now
                if pkt.player_id not in world.players:
                    world.players[pkt.player_id] = PlayerState(
                        pkt.player_id, now)
                world.players[pkt.player_id].apply(pkt, now)
                method_now = world.method

            if not quiet and now - last_draw > 0.5:
                last_draw = now
                with world.lock:
                    bits = []
                    for pid in sorted(world.players):
                        p = world.players[pid]
                        x, y, _ = p.believed_position(now, method_now)
                        mark = "~" if p.is_concealing(
                            now, world.interval) else " "
                        bits.append(f"p{pid}{mark}({x:7.0f},{y:7.0f})")
                print("\r  " + "  ".join(bits), end="", flush=True)

            if world.started and seconds and now - world.started >= seconds:
                break
    except KeyboardInterrupt:
        pass

    if world.not_ours:
        print(f"\n  ({world.not_ours} packets ignored -- wrong size or magic)")
    report(world.players, world.method, rate)


def self_test():
    """Checks the arithmetic that silently corrupts results if wrong."""
    from packet import Packet

    now = 100.0
    p = PlayerState(0, now)

    # First packet establishes state.
    p.apply(Packet(seq=400, pos=(0.0, 0.0, 0.0), vel=(100.0, 0.0, 0.0)),
            now)
    assert p.seq == 400 and p.received == 1
    assert p.missing == 0, "first packet must not count as a gap"

    # Jump to 412 after 0.2 s. Missing is 401..411 = ELEVEN.
    # True position is 20 units along; DR from vel 100 u/s for 0.2 s
    # predicts exactly 20, so DR error is 0 and freeze error is 20.
    gap = p.apply(Packet(seq=412, pos=(20.0, 0.0, 0.0), vel=(100.0, 0.0, 0.0)),
                  now + 0.2)
    assert gap is not None
    missing, elapsed = gap
    assert missing == 11, f"off-by-one: got {missing}, expected 11"
    assert abs(elapsed - 0.2) < 1e-9
    assert abs(p.errors_dr[0] - 0.0) < 1e-6, p.errors_dr
    assert abs(p.errors_freeze[0] - 20.0) < 1e-6, p.errors_freeze
    print("  gap arithmetic      OK  (400 -> 412 is 11 missing)")
    print("  dead reckoning      OK  (exact on constant velocity)")
    print("  freeze error        OK  (20.0 units)")

    # Consecutive packets are not a gap.
    q = PlayerState(1, now)
    q.apply(Packet(seq=5, pos=(0.0, 0.0, 0.0)), now)
    assert q.apply(Packet(seq=6, pos=(1.0, 0.0, 0.0)), now + 0.0156) is None
    assert q.missing == 0
    print("  consecutive seq     OK  (not counted as a gap)")

    # Reordered packet must not rewind state.
    before = q.pos
    assert q.apply(Packet(seq=5, pos=(999.0, 0.0, 0.0)), now + 0.02) is None
    assert q.pos == (1.0, 0.0, 0.0), "a reordered packet moved the player"
    print("  reordered packet    OK  (ignored, state not rewound)")

    # Units check: velocity is per second, not per tick. A player at
    # 250 u/s concealed for 500 ms must land 125 units away, not 2.
    r = dead_reckon((0.0, 0.0, 0.0), (250.0, 0.0, 0.0), 0.5)
    assert abs(r[0] - 125.0) < 1e-9, r
    print("  velocity units      OK  (250 u/s for 0.5 s = 125 units)")

    print("\n  all checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help="sender's tick rate, for converting packets to ms")
    ap.add_argument("--conceal", choices=["dr", "freeze"], default="dr",
                    help="what to believe during an outage")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 means until Ctrl-C or end-of-run packet")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--http", type=int, default=8080,
                    help="port for the JSON state endpoint, 0 to disable")
    ap.add_argument("--self-test", action="store_true",
                    help="check the arithmetic, no network")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    serve(args.port, args.rate, args.conceal, args.seconds, args.quiet,
          args.http)


if __name__ == "__main__":
    main()
