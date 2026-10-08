#!/usr/bin/env python3
"""
Did netem actually do what it was told?

WHY THIS EXISTS
    `tc qdisc add` returning 0 means the rule was accepted, not that it is
    affecting your traffic. It silently misses when: the rule is on the
    wrong interface, traffic stays on loopback while netem sits on ens5,
    the rule shapes egress when you meant the inbound path, or a namespace
    boundary gets in the way.

    The plan's own instruction is to verify the harness produces sane
    numbers under no load before trusting it under load. An uncalibrated
    tool is worse than no tool, because it reports wrong numbers
    confidently.

WHAT IT MEASURES
    Sends EGP1 game packets at a fixed rate and reports what arrived:

        loss rate        -- how many never showed up
        burst structure  -- were losses scattered or in runs
        one-way delay    -- only meaningful if the clocks agree, so it is
                            reported but flagged

    The run-length histogram is the one that matters. Uniform 5% loss and
    bursty 5% loss give the SAME loss rate and completely different
    histograms. If you configured burst mode and every run has length 1,
    the gemodel parameters did not take effect.

PACKET FORMAT
    This uses the agreed 56-byte EGP1 packet, not a private layout, so the
    same bytes Sangeet's client sends are the bytes this measures and the
    bytes the XDP classifier reads. When packet.py is factored out for the
    game client, this should import it instead of redefining PKT.

        offset 0  magic b"EGP1"   4   XDP checks this first
        offset 4  version          1
        offset 5  flags            1   the intent byte (bit0 firing, ...)
        offset 6  player_id        2
        offset 8  seq              4   loss detection -- used here
        offset 12 send_ns          8   CLOCK_REALTIME ns at send
        offset 20 pos              12  3x float32, game units
        offset 32 vel              12  3x float32, units per SECOND
        offset 44 view             8   2x float32 pitch, yaw
        offset 52 reserved         4

USAGE
    Two shells on the same box (netem must then be on lo, not ens5):
        python3 verify_netem.py --listen
        python3 verify_netem.py --host 127.0.0.1

    Across machines (netem on ens5):
        python3 verify_netem.py --listen            # on the receiver
        python3 verify_netem.py --host <receiver>   # on the sender
"""

import argparse
import collections
import socket
import struct
import sys
import time

PORT = 27015
MAGIC = b"EGP1"
VERSION = 1
PKT = struct.Struct("<4sBBHIQ3f3f2fI")     # 56 bytes
END_SEQ = 0xFFFFFFFF

# flags bits
FIRING, JUMPING, CROUCHING, RELOADING = 1, 2, 4, 8


def build(seq, player_id=0, flags=0):
    """One EGP1 packet. Position/velocity are plausible filler -- this tool
    measures delivery, not movement."""
    return PKT.pack(MAGIC, VERSION, flags, player_id, seq, time.time_ns(),
                    1050.5, -400.25, 64.0,      # pos
                    200.0, 150.0, 0.0,          # vel, units per second
                    12.5, -87.25,               # view
                    0)


def sender(host, port, rate_hz, seconds):
    n = int(rate_hz * seconds)
    interval = 1.0 / rate_hz
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"sending {n:,} EGP1 packets to {host}:{port} "
          f"at {rate_hz:g} Hz for {seconds:g}s")
    print(f"packet size {PKT.size} bytes, interval {interval*1000:.3f} ms\n")

    start = time.perf_counter()
    for i in range(n):
        # Every 8th packet carries an intent flag, so the XDP classifier has
        # something to separate when it is attached.
        flags = FIRING if i % 8 == 0 else 0
        sock.sendto(build(i, flags=flags), (host, port))

        target = start + (i + 1) * interval
        while True:
            remaining = target - time.perf_counter()
            if remaining <= 0:
                break
            if remaining > 0.002:
                time.sleep(remaining - 0.001)

    # Trailer, repeated in case it is itself dropped.
    for _ in range(5):
        sock.sendto(build(END_SEQ), (host, port))
        time.sleep(0.01)

    elapsed = time.perf_counter() - start
    print(f"sent {n:,} packets in {elapsed:.2f}s "
          f"(actual rate {n/elapsed:.1f} Hz)")


def receiver(port, rate_hz):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    print(f"listening on :{port} -- start the sender now "
          f"(Ctrl-C to stop early)\n")

    seen, delays = set(), []
    highest = -1
    not_ours = 0

    try:
        while True:
            sock.settimeout(10.0)
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                print("no packets for 10s, stopping")
                break

            if len(data) != PKT.size or data[:4] != MAGIC:
                not_ours += 1
                continue

            _, _, _, _, seq, send_ns = PKT.unpack(data)[:6]
            if seq == END_SEQ:
                break
            seen.add(seq)
            highest = max(highest, seq)
            delays.append((time.time_ns() - send_ns) / 1e6)
    except KeyboardInterrupt:
        print("\ninterrupted")

    if not_ours:
        print(f"({not_ours} packets ignored -- wrong size or magic)\n")
    report(seen, highest, delays, rate_hz)


def report(seen, highest, delays, rate_hz):
    if highest < 0:
        print("no EGP1 packets received at all.")
        print("  - is the sender pointing at the right address?")
        print("  - if across machines, is UDP 27015 open in the security group?")
        return

    total = highest + 1
    lost = [i for i in range(total) if i not in seen]
    interval_ms = 1000.0 / rate_hz

    print("=" * 62)
    print(f"  expected   : {total:,} packets (seq 0..{highest})")
    print(f"  received   : {len(seen):,}")
    print(f"  lost       : {len(lost):,}  ({len(lost)/total*100:.2f}%)")

    if delays:
        d = sorted(delays)
        print(f"\n  one-way delay, ms (ONLY meaningful if clocks agree):")
        print(f"    median {d[len(d)//2]:.2f}   "
              f"90th {d[int(len(d)*0.9)]:.2f}   max {d[-1]:.2f}")
        print("    Trust the SPREAD; treat the absolute value as suspect.")

    if not lost:
        print("\n  No loss. If you configured loss and see none, netem is not")
        print("  on the path this traffic takes. Loopback traffic never")
        print("  reaches ens5 -- put netem on lo for a same-box test.")
        print("=" * 62)
        return

    runs, run = [], 1
    for a, b in zip(lost, lost[1:]):
        if b == a + 1:
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)

    hist = collections.Counter(runs)
    mean_run = sum(runs) / len(runs)

    print(f"\n  loss events     : {len(runs)}")
    print(f"  mean run length : {mean_run:.1f} packets "
          f"= {mean_run*interval_ms:.0f} ms")
    print(f"  longest run     : {max(runs)} packets "
          f"= {max(runs)*interval_ms:.0f} ms")

    print(f"\n  run length histogram")
    for length in sorted(hist):
        bar = "#" * min(48, hist[length])
        print(f"    {length:>4} pkt ({length*interval_ms:>5.0f} ms) "
              f"x{hist[length]:<5} {bar}")

    print()
    if mean_run < 1.5:
        print("  -> UNIFORM loss: almost every drop is isolated.")
        print("     Correct for the control condition. If you meant burst")
        print("     mode, the gemodel parameters did not take effect.")
    elif mean_run * interval_ms < 200:
        print(f"  -> Bursty but short ({mean_run*interval_ms:.0f} ms mean).")
        print("     The ML measurements say dead reckoning handles anything")
        print("     under ~250 ms to within two centimetres, so a comparison")
        print("     here shows all methods overlapping. Lower r to lengthen.")
    else:
        print(f"  -> Bursty, {mean_run*interval_ms:.0f} ms mean outage.")
        print("     In the range where the methods measurably diverge.")

    print("\n  Loss rate varies a lot over short runs. Measure for a minute")
    print("  or more before quoting a number.")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true", help="run as receiver")
    ap.add_argument("--host", help="receiver address (sender mode)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--rate", type=float, default=64.0,
                    help="packets per second (default 64, a CS update rate)")
    ap.add_argument("--seconds", type=float, default=60.0)
    args = ap.parse_args()

    if args.listen:
        receiver(args.port, args.rate)
    elif args.host:
        sender(args.host, args.port, args.rate, args.seconds)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
