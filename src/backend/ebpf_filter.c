/*
 * EGP1 intent classifier -- XDP program.
 *
 * WHAT IT DOES
 *     Runs on every packet arriving at the interface, at the earliest
 *     point the kernel gives you, and sorts EGP1 game packets into two
 *     classes:
 *
 *         intent   -- the player is firing, jumping, crouching, reloading
 *         position -- a plain position update
 *
 *     It counts them into a map and passes every packet through
 *     untouched. It is a classifier, not a filter: dropping real game
 *     traffic would be a bug, not a feature.
 *
 * WHY THIS IS THE ARGUMENT FOR eBPF
 *     XDP runs in the network driver, before the kernel allocates an
 *     sk_buff for the packet. Userspace filtering happens after the
 *     packet has been copied, queued, and handed up the stack. That gap
 *     is what Dhyaan's benchmark (number #1) measures.
 *
 *     This only holds in NATIVE mode. If the program loads in generic
 *     (SKB) mode instead, it runs AFTER sk_buff allocation and the whole
 *     claim collapses. The loader reports which mode it got -- read it.
 *
 * WHY EVERY READ IS BOUNDS-CHECKED
 *     The verifier rejects any program where it cannot prove a pointer
 *     stays inside the packet. A packet can be truncated or malformed,
 *     so "the UDP header is obviously there" is not something the kernel
 *     will take on faith. Hence the wall of
 *
 *         if ((void *)(x + 1) > data_end) return XDP_PASS;
 *
 *     before each access. This is not defensive style. Remove one and
 *     the program will not load.
 *
 * THE BYTES IT READS
 *     Offsets come from packet.py, whose self-test asserts them:
 *
 *         frame + 0   Ethernet header          14 bytes
 *         frame + 14  IPv4 header              20 bytes (more with options)
 *         frame + 34  UDP header                8 bytes
 *         frame + 42  EGP1 magic "EGP1"         4 bytes
 *         frame + 46  version                   1 byte
 *         frame + 47  flags   <-- the one that matters
 *
 *     The flags byte is deliberately early in the EGP1 layout so this
 *     program reaches it with one short bounds check.
 */

/* These MUST be the uapi/ variants, not <linux/...>.
 *
 * <linux/udp.h> is the kernel-internal header: it pulls in inet_sock.h,
 * netdevice.h, security.h and eventually linux/bpf.h. On a 6.17 kernel
 * with a distro bcc, that last one fails to compile -- incomplete
 * struct bpf_wq, undeclared BPF_LOAD_ACQ -- because bcc's bundled
 * headers are older than the running kernel. The errors appear to be in
 * kernel code and have nothing to do with this program.
 *
 * The uapi/ headers define the same packet structs with no dependency
 * chain. Use them. */
#include <uapi/linux/bpf.h>
#include <uapi/linux/if_ether.h>
#include <uapi/linux/ip.h>
#include <uapi/linux/udp.h>
#include <uapi/linux/in.h>

#define EGP_PORT 27015

/* Intent bits, mirroring packet.py's FIRING/JUMPING/CROUCHING/RELOADING.
 * If a flag is added there, add it here -- these two definitions are the
 * one place the Python and the C have to agree by hand. */
#define INTENT_MASK 0x0F

/* Counter slots. The loader prints these by name. */
#define C_TOTAL    0   /* every packet the interface saw */
#define C_UDP      1   /* IPv4 + UDP on our port */
#define C_EGP1     2   /* ...and carrying the EGP1 magic */
#define C_INTENT   3   /* ...with at least one intent bit set */
#define C_POSITION 4   /* ...with none */
#define N_COUNTERS 5

BPF_ARRAY(counts, u64, N_COUNTERS);

static inline void bump(int key)
{
    u64 *v = counts.lookup(&key);
    /* lookup can return NULL and the verifier knows it. Dereferencing
     * without this check is an instant rejection. */
    if (v)
        (*v)++;
}

int xdp_classifier(struct xdp_md *ctx)
{
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    bump(C_TOTAL);

    /* --- Ethernet ------------------------------------------------- */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;
    if (bpf_ntohs(eth->h_proto) != ETH_P_IP)
        return XDP_PASS;

    /* --- IPv4 ------------------------------------------------------ */
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;
    if (ip->protocol != IPPROTO_UDP)
        return XDP_PASS;

    /* ihl counts 32-bit words, so the header is 20 bytes normally and up
     * to 60 with options. Our own traffic never uses options, but the
     * interface sees everyone's packets -- assuming 20 here would read
     * the wrong bytes on anything that does. Bounding it explicitly also
     * helps the verifier, which otherwise has to reason about a variable
     * offset derived from packet contents. */
    u32 ihl = ip->ihl * 4;
    if (ihl < sizeof(struct iphdr) || ihl > 60)
        return XDP_PASS;

    /* --- UDP ------------------------------------------------------- */
    struct udphdr *udp = (void *)ip + ihl;
    if ((void *)(udp + 1) > data_end)
        return XDP_PASS;
    if (bpf_ntohs(udp->dest) != EGP_PORT)
        return XDP_PASS;

    bump(C_UDP);

    /* --- EGP1 payload ---------------------------------------------- */
    unsigned char *pay = (void *)(udp + 1);

    /* Need magic (4) + version (1) + flags (1) = 6 bytes. Checking for
     * the whole 56-byte packet would be stricter, but then a truncated
     * packet would be counted as not-EGP1 rather than as a malformed
     * one, and the counts would quietly disagree with the sender's. */
    if ((void *)(pay + 6) > data_end)
        return XDP_PASS;

    if (pay[0] != 'E' || pay[1] != 'G' || pay[2] != 'P' || pay[3] != '1')
        return XDP_PASS;

    bump(C_EGP1);

    /* --- the actual classification --------------------------------- */
    unsigned char flags = pay[5];

    if (flags & INTENT_MASK)
        bump(C_INTENT);
    else
        bump(C_POSITION);

    return XDP_PASS;
}
