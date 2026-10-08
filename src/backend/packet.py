#!/usr/bin/env python3
"""
EGP1 -- the wire format for EdgeGaming player state.

WHY THIS FILE EXISTS
    Three separate pieces of this project read the same bytes off the wire:

        src/game_sim/client.py      writes them
        src/game_sim/server.py      reads them
        src/ebpf/classifier.c       reads two fields of them, in the kernel
        src/aws/verify_netem.py     reads the sequence number to find gaps

    If each of those carries its own copy of the layout, they will drift.
    The failure is nasty: everything works in isolation and the system only
    breaks when the pieces are joined, which is the week you have no time.
    So the layout is defined ONCE, here, and imported everywhere else.

    The C side cannot import Python, so instead this file states the byte
    offsets the C code must use, and the self-test at the bottom asserts
    them. If someone reorders a field, the self-test fails immediately
    rather than the classifier quietly reading the wrong byte.

THE LAYOUT -- 56 bytes, little-endian, no padding

    off  size  field       why it is where it is
    ---  ----  ---------   -----------------------------------------------
      0     4  magic       b"EGP1". First thing XDP checks, so it is first.
      4     1  version     Lets the format change later without guessing.
      5     1  flags       Player intent bits. XDP reads this to classify,
                           so it sits in the first 8 bytes where it is
                           cheap to reach.
      6     2  player_id   Up to 65535 players.
      8     4  seq         Per-player counter. Gaps in it ARE the packet
                           loss measurement -- this is how the receiver
                           knows something was dropped.
     12     8  send_ns     CLOCK_REALTIME nanoseconds at send.
     20    12  pos         3x float32, game units.
     32    12  vel         3x float32, units per SECOND (not per tick --
                           this caught us out once already).
     44     8  view        2x float32, pitch and yaw in degrees.
     52     4  reserved    Pads to 56 and leaves room to grow.

    Little-endian ("<") is not cosmetic: it also removes struct's automatic
    alignment padding, so Python's layout matches a C packed struct exactly.

WHY 56 BYTES
    Small enough that a 64 Hz 10-player server sends ~36 KB/s, which is
    nothing. Large enough to carry real state. Well under any MTU, so a
    packet is never fragmented -- fragmentation would make the loss
    measurements meaningless, because losing one fragment loses the packet
    but netem would have counted one drop.

USAGE
    from packet import Packet, encode, decode, FIRING

    wire = encode(Packet(seq=1, player_id=0, pos=(10.0, 20.0, 0.0),
                         vel=(250.0, 0.0, 0.0), flags=FIRING))
    p = decode(wire)
"""

import struct
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

MAGIC = b"EGP1"
VERSION = 1

# "<" little-endian AND no alignment padding. Do not drop it.
PKT = struct.Struct("<4sBBHIQ3f3f2fI")
SIZE = PKT.size                     # 56

# Sentinel the sender emits to say "that was the last one". Chosen as the
# maximum u32 so it can never collide with a real sequence number.
END_SEQ = 0xFFFFFFFF

# --- intent flags -------------------------------------------------------
# One bit each, so several can be set at once. The XDP classifier tests
# these against the single byte at offset 5.
FIRING = 1 << 0
JUMPING = 1 << 1
CROUCHING = 1 << 2
RELOADING = 1 << 3

# A packet is "intent-bearing" if any of these are set. Intent packets are
# the ones worth protecting: a missed position update can be interpolated,
# a missed shot cannot.
INTENT_MASK = FIRING | JUMPING | CROUCHING | RELOADING

FLAG_NAMES = {FIRING: "firing", JUMPING: "jumping",
              CROUCHING: "crouching", RELOADING: "reloading"}

# --- offsets the C side must match --------------------------------------
# The classifier reads a raw Ethernet frame, so each offset is shifted by
# the headers in front of the payload. These are for plain IPv4 + UDP with
# no IP options, which is what this project generates.
ETH_HDR = 14
IP_HDR = 20
UDP_HDR = 8
PAYLOAD_OFF = ETH_HDR + IP_HDR + UDP_HDR        # 42

OFF_MAGIC = 0
OFF_VERSION = 4
OFF_FLAGS = 5
OFF_PLAYER_ID = 6
OFF_SEQ = 8
OFF_SEND_NS = 12
OFF_POS = 20
OFF_VEL = 32
OFF_VIEW = 44
OFF_RESERVED = 52


@dataclass
class Packet:
    """One frame of player state.

    send_ns defaults to None and is stamped at encode() time, so the
    timestamp reflects when the packet actually went out rather than when
    the object happened to be constructed. Those differ, and the difference
    lands straight in your latency measurement.
    """
    seq: int
    player_id: int = 0
    pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    view: Tuple[float, float] = (0.0, 0.0)
    flags: int = 0
    send_ns: Optional[int] = None
    version: int = VERSION
    reserved: int = 0

    @property
    def has_intent(self) -> bool:
        return bool(self.flags & INTENT_MASK)

    def flag_list(self):
        return [name for bit, name in FLAG_NAMES.items() if self.flags & bit]


class BadPacket(ValueError):
    """Not an EGP1 packet, or a corrupted one."""


def encode(p: Packet, send_ns: Optional[int] = None) -> bytes:
    """Pack to 56 bytes. Stamps send_ns now unless one is supplied."""
    ts = send_ns if send_ns is not None else (
        p.send_ns if p.send_ns is not None else time.time_ns())
    return PKT.pack(MAGIC, p.version, p.flags & 0xFF, p.player_id & 0xFFFF,
                    p.seq & 0xFFFFFFFF, ts,
                    p.pos[0], p.pos[1], p.pos[2],
                    p.vel[0], p.vel[1], p.vel[2],
                    p.view[0], p.view[1],
                    p.reserved)


def decode(data: bytes) -> Packet:
    """Unpack 56 bytes. Raises BadPacket on anything that is not ours.

    The length and magic checks are not paranoia -- on a real port you
    will receive scans and stray traffic, and silently unpacking those
    produces garbage positions that look like data.
    """
    if len(data) != SIZE:
        raise BadPacket(f"wrong size: {len(data)} bytes, expected {SIZE}")
    if data[:4] != MAGIC:
        raise BadPacket(f"bad magic: {data[:4]!r}")

    (_, version, flags, player_id, seq, send_ns,
     px, py, pz, vx, vy, vz, pitch, yaw, reserved) = PKT.unpack(data)

    return Packet(seq=seq, player_id=player_id, pos=(px, py, pz),
                  vel=(vx, vy, vz), view=(pitch, yaw), flags=flags,
                  send_ns=send_ns, version=version, reserved=reserved)


def is_egp1(data: bytes) -> bool:
    """Cheap check with no exception, for hot receive loops."""
    return len(data) == SIZE and data[:4] == MAGIC


def _self_test():
    """Run with: python3 packet.py

    Checks the things that silently break integration: the size, every
    offset the C code hardcodes, and a full round trip.
    """
    assert SIZE == 56, f"size is {SIZE}, expected 56"
    print(f"size                     {SIZE} bytes  OK")

    # Build a packet with a distinct value in every field, so a swapped
    # pair of fields cannot pass by coincidence.
    p = Packet(seq=0x11223344, player_id=0xAABB,
               pos=(1.0, 2.0, 3.0), vel=(4.0, 5.0, 6.0),
               view=(7.0, 8.0), flags=FIRING | RELOADING,
               send_ns=1234567890123456789, reserved=0xDEADBEEF)
    wire = encode(p)
    assert len(wire) == SIZE

    # Offsets, read straight out of the bytes, the way C will read them.
    assert wire[OFF_MAGIC:OFF_MAGIC + 4] == MAGIC
    assert wire[OFF_VERSION] == VERSION
    assert wire[OFF_FLAGS] == (FIRING | RELOADING)
    assert struct.unpack_from("<H", wire, OFF_PLAYER_ID)[0] == 0xAABB
    assert struct.unpack_from("<I", wire, OFF_SEQ)[0] == 0x11223344
    assert struct.unpack_from("<Q", wire, OFF_SEND_NS)[0] == 1234567890123456789
    assert struct.unpack_from("<3f", wire, OFF_POS) == (1.0, 2.0, 3.0)
    assert struct.unpack_from("<3f", wire, OFF_VEL) == (4.0, 5.0, 6.0)
    assert struct.unpack_from("<2f", wire, OFF_VIEW) == (7.0, 8.0)
    assert struct.unpack_from("<I", wire, OFF_RESERVED)[0] == 0xDEADBEEF
    print("field offsets            OK")

    back = decode(wire)
    assert back.seq == p.seq and back.player_id == p.player_id
    assert back.pos == p.pos and back.vel == p.vel and back.view == p.view
    assert back.flags == p.flags and back.send_ns == p.send_ns
    assert back.has_intent and set(back.flag_list()) == {"firing", "reloading"}
    print("round trip               OK")

    for bad, why in [(b"", "empty"), (b"XXXX" + b"\0" * 52, "bad magic"),
                     (b"EGP1" + b"\0" * 10, "too short")]:
        try:
            decode(bad)
        except BadPacket:
            pass
        else:
            raise AssertionError(f"decode accepted {why}")
    print("rejects bad input        OK")

    print(f"\nC side: payload starts at byte {PAYLOAD_OFF} of the frame")
    print(f"  magic at frame offset  {PAYLOAD_OFF + OFF_MAGIC}")
    print(f"  flags at frame offset  {PAYLOAD_OFF + OFF_FLAGS}")
    print(f"  seq   at frame offset  {PAYLOAD_OFF + OFF_SEQ}")
    print("\nall checks passed")


if __name__ == "__main__":
    _self_test()
