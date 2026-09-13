Machine Learning-Based Network Latency Optimization Framework for Online Gaming Platforms using Edge Intelligence

Course: Cloud Architecture Design · Supervisor: Dr. Priya V

Team
Member	Track	Owns
Atharva	ML & Data	src/ml_model/, dataset/ — dataset pipeline, sequence model, baselines, evaluation
Dhyaan	Cloud & Networking	src/aws/ — netem harness, timing instrumentation, AWS
Sangeet	Kernel & Frontend	src/backend/, src/frontend/ — eBPF/XDP classifier, UDP traffic, dashboard
Problem Statement

In fast-paced multiplayer architectures, three distinct sources of delay degrade play: physical transit distance to a centralized server, host-side kernel queuing where critical inputs wait behind non-critical updates, and burst packet loss on wireless links. Traditional client-side dead reckoning extrapolates linearly and diverges badly during sustained loss, producing visible rubber-banding.

Note on scope: the ML component conceals the effects of packet loss. It does not reduce round-trip latency — that is what the edge-placement and kernel-classification components address.

Objectives
Reduce host-side packet classification latency using an intent-aware eBPF/XDP scheduler, measured against the standard Linux network stack.
Reduce spatial divergence error during burst packet loss using a sequence model, measured against linear dead reckoning.
Keep model inference within the 16.7 ms frame budget at 60 fps, measured on a single CPU core.
Evaluate across burst durations rather than at a single horizon, since the useful range depends on how long the outage lasts.
Store telemetry and training artifacts on Amazon S3 with IAM-scoped access; monitor network health via Amazon CloudWatch.
Architecture

Three layers, each attacking a different component of end-to-end latency:

Edge placement — an EC2-hosted edge node reduces physical transit distance. Note: this emulates a 5G MEC/Wavelength deployment. Real AWS Wavelength requires telecom carrier partnership and is out of scope; edge conditions are emulated with tc-netem.
Intent-aware kernel classification (eBPF/XDP) — packets are classified at the NIC driver level, before the kernel allocates packet memory, so high-intent inputs are not queued behind ambient updates.
Edge state synthesis — when packet loss is detected, a sequence model synthesizes the missing player state from recent trajectory history rather than leaving the client to extrapolate.
Results

Measured values only. Game units: 1 unit ≈ 1 inch; a player hitbox is ~32 units wide.

Metric	Baseline	Ours	Result
Kernel classification latency (µs)	standard stack	eBPF/XDP	pending
Spatial divergence, median	dead reckoning	GRU	8.0–10.0% lower
Spatial divergence, 90th percentile	dead reckoning	GRU	21.7–24.3% lower
Inference, 10 players, 1 CPU thread	DR 0.032 ms	GRU 1.07 ms median, 2.76 ms p99	within 16.7 ms budget

Measured across burst durations of 250, 500 and 750 ms (16, 32 and 48 consecutive dropped updates at 64 Hz), on games held out from training. Replicated over seven training runs: three burst lengths, two seeds each, two model sizes. The 90th-percentile gain never fell outside 21.3–24.3%.

Supporting findings:

Dead reckoning is already near-exact at short outages — 0.70 units median error at 62.5 ms, under two centimetres. There is nothing to improve below roughly 250 ms.
Dead reckoning error grows super-linearly, as burst^1.57, crossing a player's own width between 250 and 500 ms.
The gain is a constant factor, not a growing one. Both methods degrade at the same rate, so the model lowers the error curve without flattening it.
Model capacity is not the bottleneck. Increasing parameters 3.75× (14k → 53k) yielded under 2 percentage points; validation plateaued in every run while training loss kept falling. The limit is the input representation.

Full run logs and figures: src/ml_model/results/.

Dataset

CSKnow — from Durst et al., Learning to Move Like Professional Counter-Strike Players, Computer Graphics Forum, 2024 (arXiv:2408.13934).

Field	Value
File in use	behaviorTreeTeamFeatureStore_28.hdf5
Contents	136,376 rows from 25 professional Dust2 demos, 359 continuous blocks across 25 games
Sampling	128-tick competitive demos, feature store sampled at 16 Hz (62.5 ms per step)
Traces	645, all human, zero bot
History available	48 steps
Format	HDF5 feature store, ~3,000 separate 1-D arrays
Sample subset	~4% of the full dataset, publicly downloadable (1.1 GB compressed)
Full dataset	30 GB compressed / ~230 GB uncompressed, hosted on S3 as Requester Pays — egress billed to the downloader
Project page	https://mlmove.github.io
License	Code MIT; underlying match replay data subject to Valve's Counter-Strike data distribution terms

This project uses the sample subset unless demonstrably data-limited, to avoid Requester Pays egress charges.

Measured properties, verified rather than assumed (src/ml_model/check_alignment.py):

Rows are 8 game ticks apart — 16 Hz, not the 128 Hz of the underlying demos.
The feature store's own t-N history fields step two rows (125 ms), not one. The parser therefore builds history from the row timeline instead of using those fields.
Velocity is in units per second, confirmed directly.
Train/test splits are by game, never by round or randomly: consecutive examples share 15 of their 16 history frames.
Median distance travelled in one 62.5 ms step: 6.13 units.

Block lengths average ~27 s, which is shorter than a full competitive round. The CSKnow model generates movement for "Retakes" scenarios; whether this feature store is retakes-derived is not yet confirmed and should be verified before describing the data as full professional rounds.

The dataset ships a published model that performs inference for all players in under 0.5 ms per game step on a single CPU core. It is a behaviour policy evaluated by human raters, not a positional predictor scored on Euclidean error, so it is context rather than a like-for-like baseline.

Technology Stack
ML: PyTorch, GRU sequence model, NumPy, h5py, matplotlib
Kernel/Networking: eBPF, XDP, C, bcc, UDP sockets, tc-netem
Cloud: Amazon EC2, S3, CloudWatch, IAM
Frontend: HTML/JS canvas dashboard
Conventions
Each member owns one src/ subdirectory; cross-directory edits go through a PR.
All measured results land in results/. Charts are regenerated from the scripts, never hand-edited.
Every claim in the report traces to a measured number or a paper actually read.
Branches: main ← develop ← feature/<name>.
Notes for the team
Loss injection should target 250–750 ms bursts, not scattered single drops. At 62.5 ms all methods are within two centimetres of each other and any comparison chart will show identical lines.
Batch the whole team into one inference call. Per-player cost falls from 0.76 ms at batch 1 to 0.11 ms at batch 10; calling per player costs roughly 10× more for no benefit.
The earlier claim that a sequence model wins on turns is contradicted by measurement. Straight-line error compounds faster (h^1.60) than turning error (h^1.45), and the turn ratio falls from 2.19× to 1.41× as the horizon grows. Do not repeat it.
