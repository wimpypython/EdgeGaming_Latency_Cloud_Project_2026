#!/usr/bin/env python3
"""
Load and run the EGP1 XDP classifier.

WHY THIS IS SEPARATE FROM THE C
    The C is the deliverable ("eBPF/XDP intent classifier in C"). This is
    the harness that compiles it, attaches it, reads its map, and -- the
    part that is easy to skip and should not be -- detaches it on exit.

    An XDP program left attached survives this process dying. If you
    Ctrl-C out of a broken version and attach a new one, you can end up
    debugging the old program's behaviour. The cleanup below is not
    tidiness.

NATIVE VS GENERIC MODE -- READ THIS
    XDP can attach two ways:

        native (driver) -- runs inside the NIC driver, BEFORE the kernel
                           allocates an sk_buff. This is the mode the
                           project's whole eBPF argument depends on.
        generic (SKB)   -- a fallback that runs AFTER sk_buff allocation,
                           in the normal receive path.

    Generic mode works and classifies correctly, but it is NOT "before
    the kernel allocates packet memory." If the classifier is running
    generic, then number #1 is measuring something much less interesting
    and the report must say so.

    This script tries native first, falls back to generic, and prints
    which one it got in large letters. Believe it, not the plan.

USAGE
    sudo python3 run_classifier.py                  # ens5, native if possible
    sudo python3 run_classifier.py --iface lo
    sudo python3 run_classifier.py --mode generic   # force fallback
    sudo python3 run_classifier.py --seconds 60     # then exit

THEN, from the sending machine:
    python client.py --host <this box's public IP> --seconds 30

    UDP 27015 must be open inbound in the security group, or the packets
    never arrive and every counter stays at zero.
"""

import argparse
import ctypes
import os
import sys
import time

from bcc import BPF

# From uapi/linux/if_link.h. bcc exposes some of these but not reliably
# across versions, so they are spelled out.
XDP_FLAGS_SKB_MODE = 1 << 1      # generic
XDP_FLAGS_DRV_MODE = 1 << 2      # native

SLOTS = [
    ("total packets seen", 0),
    ("IPv4/UDP on 27015", 1),
    ("carrying EGP1 magic", 2),
    ("  intent-bearing", 3),
    ("  position-only", 4),
]


def load(source_path):
    if not os.path.exists(source_path):
        sys.exit(f"cannot find {source_path}")
    with open(source_path) as fh:
        text = fh.read()
    print(f"compiling {source_path} ...")
    return BPF(text=text)


def attach(b, fn, iface, mode):
    """Try native, fall back to generic. Returns the flags actually used."""
    attempts = []
    if mode in ("auto", "native"):
        attempts.append(("native (driver)", XDP_FLAGS_DRV_MODE))
    if mode in ("auto", "generic"):
        attempts.append(("generic (SKB)", XDP_FLAGS_SKB_MODE))

    last = None
    for name, flags in attempts:
        try:
            b.attach_xdp(iface, fn, flags)
            return name, flags
        except Exception as exc:          # bcc raises plain Exception
            last = exc
            print(f"  {name}: not available ({exc})")

    sys.exit(f"could not attach to {iface}: {last}")


def read_counts(b):
    table = b["counts"]
    return [table[ctypes.c_int(i)].value for _, i in SLOTS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="ens5")
    ap.add_argument("--source", default="ebpf_filter.c")
    ap.add_argument("--mode", choices=["auto", "native", "generic"],
                    default="auto")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="0 means run until Ctrl-C")
    args = ap.parse_args()

    if os.geteuid() != 0:
        sys.exit("needs root -- run with sudo")

    b = load(args.source)
    fn = b.load_func("xdp_classifier", BPF.XDP)

    print(f"attaching to {args.iface} ...")
    mode_name, flags = attach(b, fn, args.iface, args.mode)

    print()
    print("=" * 58)
    print(f"  XDP MODE: {mode_name}")
    if "generic" in mode_name:
        print("  This runs AFTER sk_buff allocation. Classification is")
        print("  correct, but the 'before the kernel allocates packet")
        print("  memory' claim does NOT hold in this mode. Say so in the")
        print("  report, or get native mode working first.")
    else:
        print("  Running inside the driver, before sk_buff allocation.")
        print("  This is the mode the project's eBPF argument needs.")
    print("=" * 58)
    print()
    print("Send traffic now. Ctrl-C to stop and detach.")
    print()

    start = time.time()
    try:
        while True:
            time.sleep(1.0)
            vals = read_counts(b)
            egp1, intent, position = vals[2], vals[3], vals[4]
            pct = (intent / egp1 * 100.0) if egp1 else 0.0
            print(f"\r  total {vals[0]:<8} udp27015 {vals[1]:<8} "
                  f"egp1 {egp1:<8} intent {intent:<7} "
                  f"pos {position:<8} intent {pct:5.1f}%", end="", flush=True)
            if args.seconds and time.time() - start >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        print()
        vals = read_counts(b)
        print("-" * 58)
        for (label, _), v in zip(SLOTS, vals):
            print(f"  {label:<22} {v:>12,}")
        print("-" * 58)

        egp1, intent, position = vals[2], vals[3], vals[4]
        if egp1 == 0:
            print("  No EGP1 packets seen. Check, in this order:")
            print("   - is UDP 27015 open inbound in the security group?")
            print("   - is the sender using this box's CURRENT public IP?")
            print("   - is the traffic arriving on this interface? Loopback")
            print("     traffic never reaches ens5.")
        elif intent + position != egp1:
            print(f"  INCONSISTENT: intent + position = {intent + position}")
            print(f"  but egp1 = {egp1}. Every EGP1 packet must land in")
            print("  exactly one class -- this is a bug in the C.")
        else:
            print(f"  Consistent: {intent:,} + {position:,} = {egp1:,}")
            print(f"  {intent / egp1 * 100:.1f}% intent-bearing. The client")
            print("  reports its own figure -- they should roughly match.")

        print()
        print(f"detaching from {args.iface}")
        b.remove_xdp(args.iface, flags)


if __name__ == "__main__":
    main()
