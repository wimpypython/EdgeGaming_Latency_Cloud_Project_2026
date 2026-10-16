#!/usr/bin/env bash
# netem.sh -- apply, clear and inspect emulated network conditions.
#
# WHY ONE SCRIPT INSTEAD OF setup.sh + reset.sh
#   The plan asks for conditions that "apply and clear identically every
#   run". Two files drift apart the moment one is edited and the other is
#   not. One file with subcommands cannot.
#
# WHY THE BURST NUMBERS ARE WHAT THEY ARE
#   The ML track measured dead reckoning -- what game netcode already does
#   when packets stop arriving. At 62.5 ms of loss its median error is 0.70
#   game units. A unit is about an inch and a player hitbox is ~32 units
#   wide, so that is under two centimetres: invisible.
#
#   Error grows as burst^1.57 and only exceeds a player's own body width
#   between 250 and 500 ms. Below that every method gives the same answer
#   and a comparison chart shows three overlapping lines.
#
#   So the interesting range is 250-750 ms outages, which at a 64 Hz update
#   rate is 16 to 48 consecutive dropped packets.
#
# HOW BURST MODE WORKS
#   netem's Gilbert-Elliott model has two states. p is the chance per packet
#   of entering the bad state, r the chance of leaving it. Mean burst length
#   is 1/r packets, and the long-run loss rate is p/(p+r).
#
#   Solving both for a 64 Hz stream gives the table in `table`. Uniform mode
#   uses plain random loss for contrast -- the "real networks do not drop
#   packets evenly" comparison the report needs.
#
# INTERFACE
#   On this EC2 instance the NIC is ens5, NOT eth0. If client and server
#   both run on the box, their traffic never reaches ens5 -- use lo:
#       NETEM_IFACE=lo sudo -E ./netem.sh burst 500 5
#
# Usage
#   ./netem.sh status
#   ./netem.sh table
#   sudo ./netem.sh uniform 5
#   sudo ./netem.sh burst   500 5
#   sudo ./netem.sh burst   500 5 --delay 50 --jitter 10
#   sudo ./netem.sh clear

set -euo pipefail

IFACE="${NETEM_IFACE:-}"
RATE_HZ="${NETEM_RATE_HZ:-64}"

die() { echo "error: $*" >&2; exit 1; }

detect_iface() {
    if [[ -n "$IFACE" ]]; then echo "$IFACE"; return; fi
    local dev
    dev=$(ip route show default 2>/dev/null | awk '/default/ {print $5; exit}')
    [[ -n "$dev" ]] || die "could not detect an interface; set NETEM_IFACE"
    echo "$dev"
}

need_root() { [[ $EUID -eq 0 ]] || die "needs root -- run with sudo"; }

# r = 1/packets (mean time in the bad state); p = L*r/(1-L) so the long-run
# loss rate lands on L.
burst_params() {
    awk -v ms="$1" -v L="$2" -v hz="$RATE_HZ" 'BEGIN {
        packets = ms / (1000.0 / hz)
        r = 1.0 / packets
        l = L / 100.0
        printf "%.4f %.4f %.1f", (l*r/(1.0-l))*100, r*100, packets
    }'
}

run_tc() { echo "  + $*"; "$@"; }

apply() {
    local dev="$1"; shift
    run_tc tc qdisc del dev "$dev" root 2>/dev/null || true
    run_tc tc qdisc add dev "$dev" root netem "$@"
    echo
    echo "Applied. Verify it took effect before trusting it:"
    echo "    python3 verify_netem.py --listen            # one shell"
    echo "    python3 verify_netem.py --host 127.0.0.1    # another"
}

cmd="${1:-help}"; shift || true
DEV=""
[[ "$cmd" != "help" && "$cmd" != "table" ]] && DEV=$(detect_iface)

case "$cmd" in

status)
    echo "interface   : $DEV"
    echo "update rate : ${RATE_HZ} Hz ($(awk -v h=$RATE_HZ 'BEGIN{printf "%.3f", 1000/h}') ms per packet)"
    echo
    tc qdisc show dev "$DEV"
    ;;

clear)
    need_root
    run_tc tc qdisc del dev "$DEV" root 2>/dev/null \
        && echo "cleared on $DEV" || echo "nothing to clear on $DEV"
    ;;

uniform)
    need_root
    loss="${1:-5}"
    echo "Uniform random loss: ${loss}% on $DEV"
    echo "  (scattered single drops -- dead reckoning handles these well;"
    echo "   this is the control condition, not the interesting one)"
    echo
    apply "$DEV" loss random "${loss}%"
    ;;

burst)
    need_root
    ms="${1:-500}"; loss="${2:-5}"; shift 2 || true
    delay=""; jitter=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --delay)  delay="$2";  shift 2 ;;
            --jitter) jitter="$2"; shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done
    read -r p r packets <<<"$(burst_params "$ms" "$loss")"
    echo "Burst loss on $DEV"
    echo "  mean outage           : ${ms} ms (~${packets%.*} packets at ${RATE_HZ} Hz)"
    echo "  long-run loss         : ${loss}%"
    echo "  gemodel p (good->bad) : ${p}%"
    echo "  gemodel r (bad->good) : ${r}%"
    echo
    args=(loss gemodel "${p}%" "${r}%")
    if [[ -n "$delay" ]]; then
        args+=(delay "${delay}ms")
        [[ -n "$jitter" ]] && args+=("${jitter}ms" distribution normal)
    fi
    apply "$DEV" "${args[@]}"
    ;;

delay)
    need_root
    d="${1:-50}"; j="${2:-0}"
    echo "Delay ${d}ms jitter ${j}ms on $DEV"
    echo
    if [[ "$j" == "0" ]]; then apply "$DEV" delay "${d}ms"
    else apply "$DEV" delay "${d}ms" "${j}ms" distribution normal; fi
    ;;

table)
    echo "netem burst parameters for a ${RATE_HZ} Hz update stream"
    echo "(derived, not guessed: r = 1/packets, p = L*r/(1-L))"
    echo
    printf "  %-10s %-9s %-12s %-12s %-12s\n" "outage" "packets" "r" "p @5%" "p @10%"
    printf "  %s\n" "-------------------------------------------------------------"
    for ms in 125 250 375 500 750; do
        read -r p5 r pk <<<"$(burst_params "$ms" 5)"
        read -r p10 _ _ <<<"$(burst_params "$ms" 10)"
        printf "  %-10s %-9s %-12s %-12s %-12s\n" "${ms} ms" "${pk%.*}" "${r}%" "${p5}%" "${p10}%"
    done
    echo
    echo "  The ML measurements say 250-750 ms is where the methods diverge."
    echo "  125 ms is shown only to illustrate 'too short to matter'."
    ;;

help|*)
    sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
