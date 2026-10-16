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
import gru_conceal

PORT = 27015
DEFAULT_RATE = 64.0

# A player is considered "in an outage" once this many tick intervals
# have passed with nothing from them. 2.5 rather than 1 because packets
# arrive slightly irregularly even on a clean link -- jitter alone would
# otherwise flip players in and out of concealment constantly.
OUTAGE_AFTER_TICKS = 2.5

# How many recent packets each player keeps, keyed by sequence number, so
# a 16-frame history at 1/4 rate can be assembled. 16 frames x stride 4 is
# 61 sequence numbers; 96 leaves room for the reordering and jitter that
# would otherwise make the oldest frame occasionally unavailable.
FRAME_BUFFER = 96

# A cached GRU prediction is reused while the outage grows by less than
# this. The dashboard polls at 10 Hz; without it every poll would run a
# forward pass per concealing player for no change in the answer.
GRU_RECOMPUTE_AFTER = 0.040


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

    def __init__(self, player_id, now, frame_time=None):
        self.id = player_id
        # When set, a gap's duration is computed from the sequence
        # numbers -- (missing + 1) * frame_time -- instead of the wall
        # clock.
        #
        # WHY THIS MATTERS FOR A REPLAY
        #     The truth a gap is scored against is the position in the
        #     next real packet, which is exactly (missing + 1) frames
        #     after the last one. The wall clock measures that interval
        #     plus whatever the OS scheduler added, so dead reckoning
        #     gets extrapolated over an interval a few milliseconds off
        #     the one the truth actually corresponds to. That noise
        #     moved this comparison by 7.7 points between two runs of
        #     identical traffic.
        #
        #     For a real network run, leave it None: there the wall
        #     clock IS the quantity of interest, because nothing else
        #     knows when the packets were sent.
        self.frame_time = frame_time
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
        self.errors_gru = []
        # DR's error on exactly the gaps the GRU answered. Without this,
        # the two medians are computed over different sets of gaps and
        # the comparison is decided by which gaps the model declined
        # rather than by how well it predicted.
        self.errors_dr_matched = []

        # One row per gap: (elapsed, err_freeze, err_dr, err_gru|None,
        # rung_rows|None, clamped). Kept so the comparison can be broken
        # down by gap length. A single median over all gaps blends three
        # different regimes -- gaps inside a rung's training range, gaps
        # the model declined, and gaps past the longest rung where the
        # hook is extrapolating rather than predicting -- and the blend
        # is dominated by whichever regime happens to be most common.
        self.gap_log = []
        self.intent_packets = 0

        # seq -> (pos, vel) for recent packets, so the GRU's 16-frame
        # history can be assembled by sequence number rather than by
        # arrival order. Arrival order is wrong the moment anything is
        # lost, which is the whole subject of this project.
        self.frames = {}

        # Why the GRU did not answer, when it did not. Counted rather
        # than ignored: a method that silently declines most gaps and
        # scores well on the rest is not comparable to one that answers
        # every gap, and this is the only way to see that happening.
        self.gru_skips = {"short": 0, "history": 0, "unavailable": 0}
        self.gru_clamped = 0          # answered beyond its trained horizon

        # (origin, predicted, horizon_seconds, elapsed_at_prediction)
        self._gru_cache = None

    def apply(self, pkt, now, gru=None):
        """Take a real packet. Returns (missing_count, gap_seconds) if
        this packet closed a gap, else None."""
        gap = None

        if self.seq is not None and pkt.seq > self.seq:
            # THE off-by-one. 400 -> 412 means 401..411 are gone: 11.
            missing = pkt.seq - self.seq - 1
            if missing > 0:
                if self.frame_time is not None:
                    elapsed = (missing + 1) * self.frame_time
                else:
                    elapsed = now - self.last_real
                self.missing += missing
                self.outages.append((missing, elapsed))

                # Score every concealment method against ground truth,
                # which is the position in THIS packet. All of them get
                # the same gaps, so none of them got easier cases.
                predicted = dead_reckon(self.pos, self.vel, elapsed)
                self.errors_dr.append(dist(predicted, pkt.pos))
                self.errors_freeze.append(dist(self.pos, pkt.pos))

                # The GRU is scored on the position it was ALREADY
                # showing at the instant the gap closed -- the same
                # function the dashboard draws, at the same elapsed
                # time. Not a fresh prediction made with hindsight.
                g = self.gru_position(now, gru, count_skip=True,
                                      elapsed=elapsed)
                e_gru = None
                if g is not None:
                    e_gru = dist(g, pkt.pos)
                    self.errors_gru.append(e_gru)
                    self.errors_dr_matched.append(self.errors_dr[-1])

                rows = clamped = None
                if gru is not None and gru.ladder.models:
                    rows, clamped = gru_conceal.choose_rung(
                        elapsed, gru.ladder.rows, gru.policy)
                self.gap_log.append((elapsed, self.errors_freeze[-1],
                                     self.errors_dr[-1], e_gru,
                                     rows, clamped))

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

        # Record the frame, then drop anything too old to be part of a
        # history window. Unbounded growth over a long run would be a
        # slow leak in the one process that must not stall.
        self.frames[pkt.seq] = (pkt.pos, pkt.vel)
        if len(self.frames) > FRAME_BUFFER * 2:
            cutoff = pkt.seq - FRAME_BUFFER
            for s in [s for s in self.frames if s < cutoff]:
                del self.frames[s]
        self._gru_cache = None        # new truth; any prediction is spent

        if pkt.flags & INTENT_MASK:
            self.intent_packets += 1
        return gap

    def gru_position(self, now, gru, count_skip=False, elapsed=None):
        """Where the GRU says this player is, or None if it will not say.

        None is a real answer, not a failure: the gap may be too short to
        be worth a model, the history may have a hole in it, or torch may
        not be installed. Each of those is counted separately so the
        report can state how often the method actually answered.
        """
        if elapsed is None:
            elapsed = now - self.last_real

        if gru is None or not gru.ladder.models:
            if count_skip:
                self.gru_skips["unavailable"] += 1
            return None

        rows, clamped = gru_conceal.choose_rung(elapsed, gru.ladder.rows,
                                                gru.policy)
        if rows is None:
            if count_skip:
                self.gru_skips["short"] += 1
            return None

        horizon = gru_conceal.rung_seconds(rows)
        if count_skip and clamped:
            self.gru_clamped += 1

        cached = self._gru_cache
        if (cached is not None and cached[2] == horizon
                and abs(elapsed - cached[3]) < GRU_RECOMPUTE_AFTER):
            origin, pred = cached[0], cached[1]
        else:
            hist = gru_conceal.build_history(self.frames, self.seq,
                                            gru.stride)
            if hist is None:
                if count_skip:
                    self.gru_skips["history"] += 1
                return None
            origin = self.pos
            pred = gru.ladder.predict(rows, hist)
            self._gru_cache = (origin, pred, horizon, elapsed)

        return gru_conceal.position_at(origin, pred, horizon, elapsed)

    def believed_position(self, now, method, gru=None):
        """Where the server currently thinks this player is.

        During an outage this is a guess. That guess is the product."""
        elapsed = now - self.last_real
        if method == "freeze":
            return self.pos
        if method == "dr":
            return dead_reckon(self.pos, self.vel, elapsed)
        if method == "gru":
            # Falls back to dead reckoning when the model declines, which
            # is what a real deployment would have to do. The fallback is
            # visible in the skip counters, not hidden.
            g = self.gru_position(now, gru)
            if g is None:
                return dead_reckon(self.pos, self.vel, elapsed)
            return g
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
    gr = [e for p in players.values() for e in p.errors_gru]

    print(f"\n  CONCEALMENT ERROR over {len(dr)} gaps, game units")
    print(f"  {'method':<16}{'median':>10}{'90th':>10}{'max':>10}{'gaps':>8}")
    print(f"  {'freeze':<16}{percentile(fz,0.5):>10.2f}"
          f"{percentile(fz,0.9):>10.2f}{max(fz):>10.2f}{len(fz):>8}")
    print(f"  {'dead reckoning':<16}{percentile(dr,0.5):>10.2f}"
          f"{percentile(dr,0.9):>10.2f}{max(dr):>10.2f}{len(dr):>8}")
    drm = [e for p in players.values() for e in p.errors_dr_matched]
    if gr:
        print(f"  {'GRU':<16}{percentile(gr,0.5):>10.2f}"
              f"{percentile(gr,0.9):>10.2f}{max(gr):>10.2f}{len(gr):>8}")
        if len(gr) < len(dr):
            # DR restricted to the gaps the GRU answered, so the two can
            # be read off the same table without the subset difference
            # quietly deciding the comparison.
            print(f"  {'  (DR, GRU gaps)':<16}{percentile(drm,0.5):>10.2f}"
                  f"{percentile(drm,0.9):>10.2f}{max(drm):>10.2f}"
                  f"{len(drm):>8}")

    med_f, med_d = percentile(fz, 0.5), percentile(dr, 0.5)
    if med_f > 0:
        gain = (med_f - med_d) / med_f * 100
        print(f"\n  Dead reckoning beats freeze by {gain:.1f}% at the median.")
        if gain < 0:
            print("  NEGATIVE -- dead reckoning is doing worse than doing")
            print("  nothing. That means velocity is being misread, or the")
            print("  gap duration is wrong. Check units before believing it.")

    if gr:
        med_g = percentile(gr, 0.5)
        med_dm = percentile(drm, 0.5)
        if med_dm > 0:
            g = (med_dm - med_g) / med_dm * 100
            print(f"\n  The GRU beats dead reckoning by {g:.1f}% at the "
                  f"median, on")
            print(f"  the {len(gr)} of {len(dr)} gaps it answered -- "
                  f"compared against")
            print("  dead reckoning on those same gaps, not on all of them.")
            if g < 0:
                print("  NEGATIVE -- the model is losing to dead reckoning on")
                print("  live traffic despite winning on the dataset. Most")
                print("  likely the history window: check the stride and that")
                print("  the generator's turn statistics resemble CSKnow's.")

        # Skips and clamps are part of the result, not diagnostics. A
        # method that declines the hard gaps and wins the easy ones is
        # not comparable to one that answers all of them.
        sk = {k: sum(p.gru_skips[k] for p in players.values())
              for k in ("short", "history", "unavailable")}
        cl = sum(p.gru_clamped for p in players.values())
        print(f"\n  GRU declined     : {sk['short']} too short "
              f"(<{gru_conceal.MIN_GAP_SECONDS*1000:.0f} ms), "
              f"{sk['history']} incomplete history, "
              f"{sk['unavailable']} unavailable")
        print(f"  beyond horizon   : {cl} gaps longer than "
              f"{gru_conceal.rung_seconds(max(gru_conceal.LADDER_ROWS))*1000:.0f}"
              f" ms, answered by the top rung outside its training range")
    elif any(p.gru_skips["unavailable"] for p in players.values()):
        print("\n  GRU not loaded, so it scored nothing. Pass --model-dir")
        print("  to score it on the same gaps as the other two.")

    # ------------------------------------------------------------------
    # Breakdown by gap length. The point of the ladder is that each rung
    # was trained at one horizon, so the only fair question about the
    # model is how it does on gaps that horizon covers. Gaps past the
    # longest rung are reported separately and never averaged in.
    # ------------------------------------------------------------------
    log = [g for p in players.values() for g in p.gap_log]
    if gr and log:
        longest_rung = gru_conceal.rung_seconds(max(gru_conceal.LADDER_ROWS))
        bands = [
            ("in range, <=250 ms", lambda e: e <= 0.250),
            ("in range, 250-500", lambda e: 0.250 < e <= 0.500),
            ("in range, 500-750", lambda e: 0.500 < e <= longest_rung),
            ("BEYOND 750 ms", lambda e: e > longest_rung),
        ]
        print(f"\n  BY GAP LENGTH -- GRU vs dead reckoning on the same gaps")
        print(f"  {'band':<20}{'n':>5}{'DR med':>9}{'GRU med':>9}"
              f"{'DR p90':>9}{'GRU p90':>9}{'median':>9}")
        print("  " + "-" * 69)
        for label, pred in bands:
            rows_ = [g for g in log if g[3] is not None and pred(g[0])]
            if len(rows_) < 5:
                print(f"  {label:<20}{len(rows_):>5}   (too few to read)")
                continue
            d = [g[2] for g in rows_]
            m = [g[3] for g in rows_]
            md, mm = percentile(d, 0.5), percentile(m, 0.5)
            delta = (md - mm) / md * 100 if md else float("nan")
            print(f"  {label:<20}{len(rows_):>5}{md:>9.2f}{mm:>9.2f}"
                  f"{percentile(d,0.9):>9.2f}{percentile(m,0.9):>9.2f}"
                  f"{delta:>8.1f}%")
        print("  Positive in the last column means the GRU was closer.")
        print("  The last band is outside every rung's training range: the")
        print("  hook extrapolates past the horizon there, so that row says")
        print("  nothing about how well the model predicts.")

    print("\n  Same gaps scored every way, so no method got easier cases.")
    print("=" * 66)


class GruRuntime:
    """The loaded ladder plus the one derived number the server needs.

    `stride` is how many packets apart the history frames must be. It is
    computed once from the live rate and the rate the models were trained
    at, and a rate that is not a whole multiple of 16 Hz is refused
    outright -- rounding it would change the span of the history window
    without changing anything visible.
    """

    def __init__(self, ladder, rate, policy="cover"):
        self.ladder = ladder
        self.rate = rate
        self.policy = policy
        self.stride = gru_conceal.stride_for(rate)
        if self.stride is None and ladder.models:
            ladder.error = (f"live rate {rate:g} Hz is not a whole "
                            f"multiple of the {gru_conceal.SOURCE_HZ:g} Hz "
                            f"the models were trained at")
            ladder.models = {}

    def describe(self):
        d = self.ladder.describe()
        if self.ladder.models:
            d += (f"  (history: every {self.stride} packet, "
                  f"rung policy: {self.policy})")
        return d


class World:
    """Everything the receive loop owns and the HTTP thread reads.

    The two run concurrently, so every access goes through `lock`.
    Without it the dashboard can read a player mid-update and render a
    position whose x is new and whose y is old -- a rare, intermittent
    jump that looks like a physics bug and is miserable to track down.
    """

    def __init__(self, rate, method, gru=None, frame_time=None):
        self.lock = threading.Lock()
        self.players = {}
        self.rate = rate
        self.interval = 1.0 / rate
        self.method = method
        self.gru = gru
        self.frame_time = frame_time
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
                bx, by, bz = p.believed_position(now, self.method, self.gru)

                # When the GRU is the drawn method, say per player whether
                # it actually answered. It declines a gap whose history
                # has a hole in it and falls back to dead reckoning --
                # correct behaviour, but on screen the dot just moves,
                # and a viewer cannot tell which method they are looking
                # at. An unlabelled fallback is a demo that claims more
                # than it is showing.
                src = self.method
                if self.method == "gru":
                    src = ("gru" if p.gru_position(now, self.gru) is not None
                           else "dr-fallback")
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
                    "source": src,
                    "concealed_for_ms": round(
                        max(0.0, now - p.last_real) * 1000.0, 1),
                    "received": p.received,
                    "missing": p.missing,
                })

            dr = [e for p in self.players.values() for e in p.errors_dr]
            fz = [e for p in self.players.values() for e in p.errors_freeze]
            gr = [e for p in self.players.values() for e in p.errors_gru]
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
                    # Scored on the gaps the model answered, which may be
                    # fewer than the total. The count is sent alongside so
                    # the dashboard can never show the error without it.
                    "error_median_gru": round(percentile(gr, 0.5), 2)
                                        if gr else None,
                    "error_p90_gru": round(percentile(gr, 0.9), 2)
                                     if gr else None,
                    "gru_scored": len(gr),
                    # DR on the same subset, so the dashboard compares
                    # like with like when the model declines gaps.
                    "error_median_dr_matched": round(percentile(
                        [e for p in self.players.values()
                         for e in p.errors_dr_matched], 0.5), 2)
                        if gr else None,
                    "gru_available": bool(self.gru
                                          and self.gru.ladder.models),
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
                if m not in ("dr", "freeze", "gru"):
                    self._send(
                        {"error": "method must be dr, freeze or gru"}, 400)
                    return
                if m == "gru" and not (world.gru
                                       and world.gru.ladder.models):
                    self._send({"error": "gru unavailable",
                                "detail": world.gru.ladder.error
                                if world.gru else "not loaded"}, 409)
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


def serve(port, rate, method, seconds, quiet, http_port,
          model_dir=None, seed=0, policy="cover", gap_time="clock"):
    gru = None
    if model_dir:
        gru = GruRuntime(gru_conceal.Ladder().load(model_dir, seed=seed),
                         rate, policy)
        print(gru.describe())
    frame_time = (1.0 / rate) if gap_time == "seq" else None
    world = World(rate, method, gru, frame_time)
    if frame_time:
        print(f"gap duration from sequence numbers "
              f"({frame_time*1000:.3f} ms per frame), not the wall clock")

    if method == "gru" and not (gru and gru.ladder.models):
        print("  refusing to start: --conceal gru was asked for but no")
        print("  model loaded. Pass --model-dir, or choose dr/freeze.")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    sock.settimeout(1.0)

    print(f"listening on :{port}, concealing with '{method}'")
    if http_port:
        start_http(world, http_port)
        print(f"state endpoint on http://0.0.0.0:{http_port}/state")
        print(f"  toggle   : /conceal?method=dr|freeze|gru")
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
                        pkt.player_id, now, world.frame_time)
                world.players[pkt.player_id].apply(pkt, now, world.gru)
                method_now = world.method

            if not quiet and now - last_draw > 0.5:
                last_draw = now
                with world.lock:
                    bits = []
                    for pid in sorted(world.players):
                        p = world.players[pid]
                        x, y, _ = p.believed_position(now, method_now,
                                                      world.gru)
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

    # Sequence-derived gap time must be exact, not clock-dependent: the
    # same packets delivered late must score identically.
    for lateness in (0.0, 0.037):
        w = PlayerState(7, now, frame_time=1.0 / 16.0)
        w.apply(Packet(seq=100, pos=(0.0, 0.0, 0.0), vel=(160.0, 0.0, 0.0)),
                now)
        w.apply(Packet(seq=108, pos=(80.0, 0.0, 0.0), vel=(160.0, 0.0, 0.0)),
                now + 0.5 + lateness)
        assert w.outages[0][0] == 7
        # 7 missing + 1 = 8 frames = 0.5 s exactly, whatever the clock said
        assert abs(w.outages[0][1] - 0.5) < 1e-12, w.outages
        assert abs(w.errors_dr[0] - 0.0) < 1e-4, w.errors_dr
    print("  seq gap time        OK  (exact, unaffected by 37 ms of jitter)")

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

    # -----------------------------------------------------------------
    # GRU wiring. A stub model stands in for torch so this runs
    # anywhere: it returns a position 120 units along x, whatever the
    # history. That is enough to check the plumbing, which is where the
    # bugs would be -- the model's accuracy is train.py's business.
    # -----------------------------------------------------------------
    class StubLadder:
        rows = (4, 8, 12)
        models = {4: None, 8: None, 12: None}
        error = None
        seen = []

        def predict(self, rows, history):
            assert len(history) == gru_conceal.HISTORY_LEN, len(history)
            assert len(history[0]) == 6, len(history[0])
            self.seen.append((rows, history[-1][0]))
            return (history[-1][0] + 120.0, history[-1][1], history[-1][2])

    stub = StubLadder()
    gru = GruRuntime(stub, 64.0)
    assert gru.stride == 4, gru.stride

    # Feed 80 consecutive packets at 64 Hz so a full history exists.
    s = PlayerState(9, 0.0)
    tick = 1.0 / 64.0
    for i in range(80):
        s.apply(Packet(seq=i, pos=(float(i), 0.0, 0.0),
                       vel=(200.0, 0.0, 0.0)), i * tick, gru)
    assert s.seq == 79 and not s.errors_gru

    # 60 ms gap: too short for any rung, must decline rather than guess.
    assert s.gru_position(79 * tick + 0.060, gru, count_skip=True) is None
    assert s.gru_skips["short"] == 1, s.gru_skips
    print("  gru short gap       OK  (declines under 125 ms)")

    # 500 ms gap: must use the 500 ms rung, built from every 4th packet.
    pos500 = s.gru_position(79 * tick + 0.500, gru)
    assert stub.seen and stub.seen[-1][0] == 8, stub.seen[-1]
    assert abs(pos500[0] - (79.0 + 120.0)) < 1e-6, pos500
    print("  gru rung + history  OK  (500 ms gap -> h8, 16 frames step 4)")

    # Halfway through the horizon it must be halfway to the prediction,
    # not sitting at it. This is the interpolation actually being used.
    # 400 ms picks the 500 ms rung (nearest), so it is 80% of the way
    # there: 0.8 x 120 = 96 units. At exactly 250 ms the ladder would
    # pick the 250 ms rung instead and sit at the full prediction, which
    # is correct behaviour and would test nothing.
    part = s.gru_position(79 * tick + 0.400, gru)
    assert stub.seen[-1][0] == 8, stub.seen[-1]
    assert abs(part[0] - (79.0 + 96.0)) < 1e-6, part
    print("  gru interpolation   OK  (400 ms into a 500 ms horizon = 80%)")

    # A hole in the history frames must make it decline, not substitute.
    del s.frames[79 - 4 * 3]
    s._gru_cache = None
    assert s.gru_position(79 * tick + 0.500, gru, count_skip=True) is None
    assert s.gru_skips["history"] == 1, s.gru_skips
    # And the server must still render something -- falling back to DR.
    fb = s.believed_position(79 * tick + 0.500, "gru", gru)
    dr_fb = dead_reckon(s.pos, s.vel, 0.500)
    assert abs(fb[0] - dr_fb[0]) < 1e-6, (fb, dr_fb)
    print("  gru missing history OK  (declines, falls back to DR)")

    # Scoring at gap close must pair GRU with DR on the SAME gap, or the
    # two medians describe different sets of gaps.
    t = PlayerState(10, 0.0)
    for i in range(80):
        t.apply(Packet(seq=i, pos=(float(i), 0.0, 0.0),
                       vel=(200.0, 0.0, 0.0)), i * tick, gru)
    closed = t.apply(Packet(seq=120, pos=(500.0, 0.0, 0.0),
                            vel=(200.0, 0.0, 0.0)),
                     79 * tick + 0.500, gru)
    assert closed is not None and closed[0] == 40
    assert len(t.errors_gru) == 1, t.errors_gru
    assert len(t.errors_dr_matched) == len(t.errors_gru)
    assert abs(t.errors_dr_matched[0] - t.errors_dr[-1]) < 1e-9
    print("  gru scoring         OK  (paired with DR on the same gap)")

    # A gap past the longest rung must be answered AND flagged.
    u = PlayerState(11, 0.0)
    for i in range(80):
        u.apply(Packet(seq=i, pos=(float(i), 0.0, 0.0),
                       vel=(200.0, 0.0, 0.0)), i * tick, gru)
    assert u.gru_position(79 * tick + 1.344, gru, count_skip=True) is not None
    assert u.gru_clamped == 1, u.gru_clamped
    print("  gru beyond horizon  OK  (answered, counted as clamped)")

    # With no ladder at all, 'gru' must degrade to DR rather than raise.
    none_rt = GruRuntime(gru_conceal.Ladder(), 64.0)
    v = PlayerState(12, 0.0)
    v.apply(Packet(seq=1, pos=(0.0, 0.0, 0.0), vel=(200.0, 0.0, 0.0)),
            0.0, none_rt)
    got = v.believed_position(0.5, "gru", none_rt)
    assert abs(got[0] - 100.0) < 1e-6, got
    print("  gru unavailable     OK  (degrades to DR, no exception)")

    # An unusable live rate must disable the model, not resample it.
    bad = GruRuntime(StubLadder(), 60.0)
    assert not bad.ladder.models, "60 Hz must disable the GRU"
    assert "multiple" in bad.ladder.error
    print("  rate mismatch       OK  (60 Hz disables, does not round)")

    print("\n  all checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--rate", type=float, default=DEFAULT_RATE,
                    help="sender's tick rate, for converting packets to ms")
    ap.add_argument("--conceal", choices=["dr", "freeze", "gru"],
                    default="dr",
                    help="what to believe during an outage")
    ap.add_argument("--model-dir", default=None,
                    help="directory holding gru_h{4,8,12}_s0.pt; "
                         "omit to run without the GRU")
    ap.add_argument("--gap-time", choices=["clock", "seq"], default="clock",
                    help="'seq' derives a gap's duration from the "
                         "sequence numbers: exact and reproducible, "
                         "correct for a replay at a known frame rate. "
                         "'clock' uses the wall clock, which is what a "
                         "real network run must use")
    ap.add_argument("--rung-policy", choices=list(gru_conceal.RUNG_POLICIES),
                    default="cover",
                    help="'cover' never asks a rung shorter than the gap, "
                         "so the model is never extrapolated past its "
                         "trained horizon. 'nearest' is the earlier rule, "
                         "kept so both can be measured on the same gaps")
    ap.add_argument("--seed", type=int, default=0,
                    help="which trained seed to load. Fixed, not "
                         "best-of: picking the best held-out score "
                         "would be selecting on the test set")
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
          args.http, args.model_dir, args.seed, args.rung_policy,
          args.gap_time)


if __name__ == "__main__":
    main()
