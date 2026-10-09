#!/usr/bin/env python3
"""
Kernel-timestamped receive latency, with SO_TIMESTAMPNS.

WHY THIS EXISTS
    verify_netem.py measures delay with time.time_ns() in userspace --
    the moment this Python process got around to LOOKING at the packet.
    That includes however long the process sat waiting to be scheduled,
    which on a loaded 2-vCPU box is easily a millisecond and is wildly
    variable.

    That is fine for counting lost packets, which is all verify_netem.py
    used it for, and it says so. It is not good enough to quote a latency
    figure next to "1.07 ms inference against a 16.67 ms frame budget".
    One precise number and one sloppy number in the same sentence is
    worse than no number.

    SO_TIMESTAMPNS makes the KERNEL stamp each packet as it arrives,
    before this process is involved at all.

WHAT IT SEPARATES
    Three quantities that a userspace-only measurement blends into one:

      kernel arrival -> userspace read   SCHEDULING DELAY
          How long the packet sat in the socket queue waiting for this
          process to be run. Pure software latency on this host. Clean:
          both timestamps come from the same clock on the same machine.

      sender send_ns -> kernel arrival   ONE-WAY NETWORK DELAY
          Only meaningful if the two clocks agree. Across machines they
          do not, so this is reported but flagged, and the SPREAD is the
          part worth trusting.

      kernel arrival -> kernel arrival   INTER-ARRIVAL JITTER
          Gaps between consecutive packets, measured entirely with kernel
          stamps. No clock skew, no scheduling noise. This is the honest
          view of what the network did, and the one to put in the report.

    It also reports the userspace-only number alongside, so the
    difference between the two methods is visible rather than asserted.
    That contrast is the point of the deliverable.

HOW IT WORKS
    setsockopt(SOL_SOCKET, SO_TIMESTAMPNS, 1) tells the kernel to attach
    a timestamp to each datagram as ancillary ("control") data. You then
    have to use recvmsg() rather than recvfrom() to collect it, and parse
    a struct timespec out of the cmsg.

    This is Linux-only. It will not run on Windows -- that is not a bug,
    the socket option does not exist there.

RUN
    On the receiving machine:
        python3 timing_harness.py --listen

    Then send traffic:
        python3 client.py --host <receiver> --seconds 60

USAGE NOTE FOR NUMBER #1
    For the eBPF vs standard-stack comparison, run this identically on
    both sides. The native-XDP configuration is single-queue with MTU
    1500, so the standard-stack run must use the same, or the comparison
    is not like-for-like.
"""

import argparse
import array
import socket
import struct
import sys
import time

# Linux socket constants. Python does not reliably expose these as named
# attributes across versions, so they are spelled out. Both are 35 on
# Linux; the duplication is not a typo.
SO_TIMESTAMPNS = 35
SCM_TIMESTAMPNS = 35

# struct timespec on 64-bit Linux: time_t tv_sec; long tv_nsec;
TIMESPEC = struct.Struct("qq")

PORT = 27015

# EGP1 fields this tool needs. packet.py is the source of truth; these
# are used only when it is not importable (it lives in src/backend and
# this file lives in src/aws). The offsets are asserted against it when
# the import does work, so the two cannot silently drift.
MAGIC = b"EGP1"
PKT_SIZE = 56
OFF_SEQ = 8
OFF_SEND_NS = 12

try:
    sys.path.insert(0, "../backend")
    from packet import SIZE as _S, OFF_SEQ as _Q, OFF_SEND_NS as _N, MAGIC as _M
    assert (_S, _Q, _N, _M) == (PKT_SIZE, OFF_SEQ, OFF_SEND_NS, MAGIC), \
        "EGP1 layout in packet.py no longer matches this file"
    _IMPORTED = True
except Exception:
    _IMPORTED = False


def percentile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def summarize(name, vals, unit="ms"):
    if not vals:
        print(f"  {name:<34} no samples")
        return
    print(f"  {name:<34}"
          f"{percentile(vals,0.5):>9.3f}"
          f"{percentile(vals,0.9):>9.3f}"
          f"{percentile(vals,0.99):>9.3f}"
          f"{max(vals):>9.3f}")


def listen(port, seconds, quiet):
    if not hasattr(socket, "AF_PACKET") and sys.platform != "linux":
        sys.exit("SO_TIMESTAMPNS is Linux-only. This will not run here.")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
    except OSError as e:
        sys.exit(f"could not enable SO_TIMESTAMPNS: {e}")
    sock.bind(("", port))
    sock.settimeout(5.0)

    print(f"listening on :{port} with kernel timestamping")
    print(f"EGP1 layout {'imported from packet.py' if _IMPORTED else 'using local constants'}")
    print("waiting for packets (Ctrl-C to stop)\n")

    sched = []        # kernel arrival -> userspace read
    oneway = []       # sender send_ns -> kernel arrival (clock-dependent)
    usonly = []       # sender send_ns -> userspace read (the naive method)
    gaps = []         # kernel arrival -> next kernel arrival
    prev_kern = None
    n = 0
    no_stamp = 0
    not_ours = 0
    started = None

    # Ancillary buffer: one cmsg holding a timespec, plus header slack.
    ancbufsize = socket.CMSG_SPACE(TIMESPEC.size)

    try:
        while True:
            try:
                data, anc, flags, _addr = sock.recvmsg(256, ancbufsize)
            except socket.timeout:
                if started:
                    break
                continue

            # Read the userspace clock IMMEDIATELY, before any parsing.
            # Anything done first is counted as scheduling delay and
            # inflates the very number this tool exists to isolate.
            user_ns = time.time_ns()

            if len(data) != PKT_SIZE or data[:4] != MAGIC:
                not_ours += 1
                continue

            kern_ns = None
            for level, ctype, cdata in anc:
                if level == socket.SOL_SOCKET and ctype == SCM_TIMESTAMPNS:
                    sec, nsec = TIMESPEC.unpack(cdata[:TIMESPEC.size])
                    kern_ns = sec * 1_000_000_000 + nsec
                    break
            if kern_ns is None:
                no_stamp += 1
                continue

            send_ns = struct.unpack_from("<Q", data, OFF_SEND_NS)[0]

            sched.append((user_ns - kern_ns) / 1e6)
            oneway.append((kern_ns - send_ns) / 1e6)
            usonly.append((user_ns - send_ns) / 1e6)
            if prev_kern is not None:
                gaps.append((kern_ns - prev_kern) / 1e6)
            prev_kern = kern_ns

            n += 1
            if started is None:
                started = time.monotonic()
            if not quiet and n % 256 == 0:
                print(f"\r  {n:,} packets", end="", flush=True)
            if seconds and time.monotonic() - started >= seconds:
                break
    except KeyboardInterrupt:
        pass

    print()
    report(n, sched, oneway, usonly, gaps, no_stamp, not_ours)


def report(n, sched, oneway, usonly, gaps, no_stamp, not_ours):
    print()
    print("=" * 70)
    print(f"  packets with kernel timestamps : {n:,}")
    if no_stamp:
        print(f"  packets WITHOUT a timestamp    : {no_stamp:,}  <-- "
              f"SO_TIMESTAMPNS not taking effect")
    if not_ours:
        print(f"  ignored (wrong size or magic)  : {not_ours:,}")
    if n == 0:
        print("\n  Nothing measured. Is the sender pointing here, and is")
        print("  UDP open inbound in the security group?")
        print("=" * 70)
        return

    print()
    print(f"  {'':<34}{'median':>9}{'90th':>9}{'99th':>9}{'max':>9}   ms")
    print("  " + "-" * 68)
    summarize("scheduling delay (kernel->user)", sched)
    summarize("inter-arrival gap (kernel only)", gaps)
    print("  " + "-" * 68)
    summarize("one-way delay (kernel stamp)", oneway)
    summarize("one-way delay (userspace only)", usonly)
    print("=" * 70)

    print()
    print("  READING THESE")
    print("   scheduling delay is pure host software latency: how long the")
    print("   packet waited in the socket queue for this process to run.")
    print("   Both stamps come from one clock on one machine, so it is the")
    print("   number to trust and the one to quote.")
    print()
    print("   inter-arrival gap is what the network delivered, measured")
    print("   entirely with kernel stamps -- no clock skew, no scheduling")
    print("   noise. At 64 Hz it should sit near 15.625 ms; the spread is")
    print("   the jitter.")
    print()
    print("   the two one-way rows differ ONLY by the scheduling delay.")
    med_k = percentile(oneway, 0.5)
    med_u = percentile(usonly, 0.5)
    p99_k = percentile(oneway, 0.99)
    p99_u = percentile(usonly, 0.99)
    print(f"   median {med_u - med_k:+.3f} ms, 99th {p99_u - p99_k:+.3f} ms.")
    print("   That difference is what a userspace-only measurement silently")
    print("   attributes to the network. Across machines the absolute values")
    print("   carry clock skew -- trust the spread, not the magnitude.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", action="store_true")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 means until the sender stops or Ctrl-C")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not args.listen:
        ap.print_help()
        print("\nThis is the receiving half. Send with client.py.")
        sys.exit(1)
    listen(args.port, args.seconds, args.quiet)


if __name__ == "__main__":
    main()
