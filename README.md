# Machine Learning-Based Network Latency Optimization Framework for Online Gaming Platforms using Edge Intelligence

**Course:** Cloud Architecture Design · **Supervisor:** Dr. Priya V

---

## Team

| Member | Track | Owns |
|---|---|---|
| **Atharva** | ML & Data | `src/ml_model/`, `dataset/` — dataset pipeline, sequence model, baselines, evaluation |
| **Dhyaan** | Cloud & Networking | `src/aws/` — netem harness, timing instrumentation, AWS |
| **Sangeet** | Kernel & Frontend | `src/backend/`, `src/frontend/` — eBPF/XDP classifier, UDP traffic, dashboard |

---

## Problem Statement

In fast-paced multiplayer architectures, three distinct sources of delay degrade play: physical transit distance to a centralized server, host-side kernel queuing where critical inputs wait behind non-critical updates, and burst packet loss on wireless links. Traditional client-side dead reckoning extrapolates linearly and diverges badly during sustained loss, producing visible rubber-banding.

**Note on scope:** the ML component conceals the *effects* of packet loss. It does not reduce round-trip latency — that is what the edge-placement and kernel-classification components address.

---

## Objectives

1. Reduce host-side packet classification latency using an intent-aware eBPF/XDP scheduler, measured against the standard Linux network stack.
2. Reduce spatial divergence error during burst packet loss using a sequence model, measured against linear dead reckoning.
3. Keep model inference within the 16.7 ms frame budget at 60 fps, measured on a single CPU core.
4. Evaluate across burst durations rather than at a single horizon, since the useful range depends on how long the outage lasts.
5. Store telemetry and training artifacts on Amazon S3 with IAM-scoped access; monitor network health via Amazon CloudWatch.

---

## Architecture

Three layers, each attacking a different component of end-to-end latency:

- **Edge placement** — an EC2-hosted edge node reduces physical transit distance. *Note: this **emulates** a 5G MEC/Wavelength deployment. Real AWS Wavelength requires telecom carrier partnership and is out of scope; edge conditions are emulated with `tc-netem`.*
- **Intent-aware kernel classification (eBPF/XDP)** — packets are classified at the NIC driver level, before the kernel allocates packet memory, so high-intent inputs are not queued behind ambient updates.
- **Edge state synthesis** — when packet loss is detected, a sequence model synthesizes the missing player state from recent trajectory history rather than leaving the client to extrapolate.

---

## Results

Measured values only. Game units: 1 unit ≈ 1 inch; a player hitbox is ~32 units wide.

| Metric | Baseline | Ours | Result |
|---|---|---|---|
| Kernel classification
