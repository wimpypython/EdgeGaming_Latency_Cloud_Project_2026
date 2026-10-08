#!/usr/bin/env python3
"""
bcc hello-world -- prove the eBPF toolchain works before writing the real thing.

WHY THIS EXISTS
    The XDP classifier can fail for two completely different reasons:

        1. The toolchain is broken  -- headers missing, kernel mismatch,
           no permission to load BPF, bcc not finding clang.
        2. Your C is wrong          -- verifier rejection, bad offsets,
           unbounded access.

    Those are debugged in totally different ways, and when you hit the
    first error on a 100-line classifier you cannot tell which one you
    have. This program removes category 1 from the search. If this runs,
    the toolchain is fine and every later failure is your logic.

WHAT IT DOES
    Attaches to the execve syscall -- the thing the kernel does whenever a
    program starts. Every time you run any command on this box, this
    fires.

    It does two things on each event, deliberately:

      a) prints a line via bpf_trace_printk      -- the classic hello
      b) increments a counter in a BPF MAP       -- the thing that matters

    (b) is the point. A BPF map is shared memory between kernel space and
    user space: the kernel program writes, your Python reads. The XDP
    classifier reports its intent-vs-position counts exactly this way, so
    getting maps working on something trivial means maps are not a second
    unknown when the verifier starts rejecting your packet parsing.

WHY A TRACEPOINT AND NOT A KPROBE
    The classic bcc hello-world attaches a kprobe to `sys_clone`. The
    symbol name for that has changed across kernel versions and now
    commonly fails on 6.x with a confusing "failed to attach" error --
    which is precisely the false alarm this program exists to avoid.
    Tracepoints are a stable kernel ABI, so `syscalls:sys_enter_execve`
    works the same across versions.

RUN IT
    sudo python3 hello.py

    Needs sudo: loading BPF requires CAP_BPF/CAP_SYS_ADMIN.

    Then open a SECOND ssh session and run anything -- `ls`, `date`. Each
    command you run there appears here. Ctrl-C to stop and print totals.

IF IT FAILS
    "Failed to compile"      -> kernel headers missing:
                                sudo apt install linux-headers-$(uname -r)
    "Operation not permitted"-> you forgot sudo
    "No such file or directory" on /sys/kernel/debug
                             -> sudo mount -t debugfs none /sys/kernel/debug
"""

from bcc import BPF

# This string is C. bcc compiles it with clang, checks it with the kernel's
# verifier, and loads it. It is NOT Python, and it does not have Python's
# freedoms: no unbounded loops, no arbitrary memory access, no calling
# libc. The verifier rejects anything it cannot prove safe.
BPF_PROGRAM = r"""
// A map: an array of exactly one 64-bit counter, shared with user space.
// The classifier will use the same mechanism with more slots -- one per
// packet class.
BPF_ARRAY(counts, u64, 1);

TRACEPOINT_PROBE(syscalls, sys_enter_execve) {
    // Write to the shared counter.
    //
    // lookup_or_try_init returns a POINTER, and it can return NULL. The
    // verifier will reject the program outright if you dereference it
    // without checking -- this null check is not defensive style, it is
    // mandatory. Expect to write a lot of these in the classifier.
    int key = 0;
    u64 zero = 0;
    u64 *val = counts.lookup_or_try_init(&key, &zero);
    if (val) {
        (*val)++;
    }

    // And print, so there is something to watch live.
    bpf_trace_printk("exec\n");
    return 0;
}
"""


def main():
    print("compiling and loading BPF program...")
    b = BPF(text=BPF_PROGRAM)
    print("loaded and attached to syscalls:sys_enter_execve")
    print()
    print("Open a SECOND ssh session and run some commands (ls, date).")
    print("Each one should appear below. Ctrl-C to stop.")
    print("-" * 56)

    seen = 0
    try:
        while True:
            try:
                task, pid, cpu, flags, ts, msg = b.trace_fields()
            except ValueError:
                continue
            seen += 1
            name = task.decode(errors="replace")
            print(f"  {ts:14.6f}  pid {pid:<7} {name}")
    except KeyboardInterrupt:
        print()

    # Read the map back. This is the part worth paying attention to: the
    # kernel wrote this number, user space is reading it.
    total = b["counts"][0].value

    print("-" * 56)
    print(f"  events printed  : {seen}")
    print(f"  map counter     : {total}")
    print()

    if total == 0:
        print("  Counter is zero. The program loaded but nothing ran execve")
        print("  while it was attached -- run a command in another session")
        print("  next time. Loading worked, which is what this test is for.")
    else:
        print("  TOOLCHAIN OK. Program compiled, loaded, attached, ran in")
        print("  kernel space, wrote to a map, and user space read it back.")
        print("  Every one of those steps is needed by the XDP classifier.")
        if seen and total < seen:
            print()
            print(f"  (counter {total} < printed {seen}: the trace pipe is")
            print("   global, so some lines may be from an earlier run.")
            print("   The map is the trustworthy number.)")


if __name__ == "__main__":
    main()
