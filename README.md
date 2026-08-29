# Machine Learning-Based Network Latency Optimization Framework for Online Gaming Platforms using Edge Intelligence

**Course:** Cloud Architecture Design · **Supervisor:** Dr. Priya V

---

## Team

| Member | Track | Owns |
|---|---|---|
| **Atharva** | ML & Data | `src/ml/` — dataset pipeline, SSM model, baselines, evaluation |
| **Dhyaan** | Cloud & Networking | `src/net/` — netem harness, timing instrumentation, AWS |
| **Sangeet** | Kernel & Frontend | `src/kernel/`, `demo/` — eBPF/XDP classifier, UDP traffic, dashboard |

---

## Problem Statement

In fast-paced multiplayer architectures, three distinct sources of delay degrade play: physical transit distance to a centralized server, host-side kernel queuing where critical inputs wait behind non-critical updates, and burst packet loss on wireless links. Traditional client-side dead reckoning extrapolates linearly and diverges badly during sustained loss, producing visible rubber-banding.

---

## Objectives

1. Reduce host-side packet classification latency using an intent-aware eBPF/XDP scheduler, measured against the standard Linux network stack.
2. Reduce spatial divergence error during packet loss using a state-space sequence model, measured against both linear dead-reckoning and a published transformer baseline.
3. Keep model inference within the 16.6ms frame budget at 60fps, measured on a single CPU core.
4. Evaluate under both uniform and bursty packet loss, since real networks drop packets in runs rather than uniformly.
5. Store telemetry and training artifacts on Amazon S3 with IAM-scoped access; monitor network health via Amazon CloudWatch.

---

## Architecture

Three layers, each attacking a different component of end-to-end latency:

- **Edge placement** — an EC2-hosted edge node reduces physical transit distance. *Note: this **emulates** a 5G MEC/Wavelength deployment. Real AWS Wavelength requires telecom carrier partnership and is out of scope; edge conditions are emulated with `tc-netem`.*
- **Intent-aware kernel classification (eBPF/XDP)** — packets are classified at the NIC driver level, before the kernel allocates packet memory, so high-intent inputs are not queued behind ambient updates.
- **Edge state synthesis (SSM)** — when packet loss is detected, a state-space model synthesizes the missing player state from recent trajectory history rather than leaving the client to extrapolate.

---

## Results

The project is evaluated on three measured numbers. **Populate with real values as they are produced — do not fill in projections.**

| Metric | Baseline | Ours | Status |
|---|---|---|---|
| Kernel classification latency (µs) | standard stack | eBPF/XDP | pending |
| Spatial divergence error (uniform loss) | dead-reckoning, CSKnow transformer | SSM | pending |
| Spatial divergence error (burst loss) | dead-reckoning, CSKnow transformer | SSM | pending |
| Inference latency (ms/step, 1 CPU core) | CSKnow transformer (<0.5ms published) | SSM | pending |

---

## Dataset

**CSKnow** — from Durst et al., *Learning to Move Like Professional Counter-Strike Players*, Computer Graphics Forum, 2024 ([arXiv:2408.13934](https://arxiv.org/abs/2408.13934)).

| Field | Value |
|---|---|
| Contents | 123 hours of professional CS:GO gameplay traces, curated for team movement learning |
| Sampling | 128 Hz (competitive CS:GO tick rate) |
| Format | Structured binary tables via CSKnow, a custom C++ parsing and spatial-indexing engine; C++ and Python loaders provided for direct PyTorch tensor loading |
| Sample subset | ~4% of full dataset, publicly downloadable |
| Full dataset | 30 GB compressed / ~230 GB uncompressed, hosted on S3 as **Requester Pays** — egress billed to the downloader |
| Project page | https://mlmove.github.io |
| License | Code MIT; underlying match replay data subject to Valve's Counter-Strike data distribution terms |

**This project uses the sample subset unless demonstrably data-limited**, to avoid Requester Pays egress charges. ~4% of 123 hours at 128 Hz is ample for pipeline development and evaluation.

The dataset also supplies a **published baseline**: the CSKnow transformer performs inference for all players in under 0.5 ms per game step on a single CPU core, giving a citable comparison point for the inference-latency claim.

---


## Technology Stack

- **ML:** PyTorch, state-space models, pandas, NumPy
- **Kernel/Networking:** eBPF, XDP, C, bcc, UDP sockets, `tc-netem`
- **Cloud:** Amazon EC2, S3, CloudWatch, IAM
- **Frontend:** HTML/JS canvas dashboard

---

## Conventions

- Each member owns one `src/` subdirectory; cross-directory edits go through a PR.
- All measured results land in `results/` as CSV. Charts are regenerated from CSVs, never hand-edited.
- Every claim in the report traces to a measured number or a paper actually read.
- Branches: `main` ← `develop` ← `feature/<name>`.
