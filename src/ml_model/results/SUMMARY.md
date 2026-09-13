\# ML track — measured results



\*\*Task:\*\* given 16 frames (1.0 s) of a player's position and velocity,

predict their position after a burst of lost updates. Compared against

linear dead reckoning, the standard netcode technique.



\*\*Data:\*\* CSKnow feature store, 25 professional Dust2 demos, 359 continuous

blocks. Rows 62.5 ms apart (16 Hz). Split by GAME — never by round, never

randomly — because consecutive examples share 15 of 16 history frames.



\## Headline



| | result |

|---|---|

| Median error vs dead reckoning | \*\*8.0–10.0% lower\*\* |

| 90th-percentile error vs DR | \*\*21.7–24.3% lower\*\* |

| Burst range tested | 250–750 ms (16–48 dropped updates at 64 Hz) |

| Inference, 10 players, 1 CPU thread | 1.07 ms median, 2.76 ms at 99th |

| Frame budget at 60 fps | 16.67 ms |



Replicated over 7 training runs: 3 horizons, 2 seeds each, 2 model sizes.

The 90th-percentile gain never fell outside 21.3–24.3%.



\## Findings



\*\*Lead with the 90th percentile, not the median.\*\* The median is dominated

by easy cases where both methods nearly agree, so it is noisy — consecutive

epochs gave 9.5%, 2.8%, 0.2% in one run. The tail is where the methods

actually differ and it moved monotonically in almost every run.



\*\*The gain is constant, not growing.\*\* Both methods degrade at the same rate

(burst^1.53 median, burst^1.36 at the 90th). The model lowers the error

curve; it does not flatten it.



\*\*Capacity is not the bottleneck.\*\* 3.75× the parameters (14k → 53k) bought

under 2 points. Validation plateaued in all runs while training loss kept

falling. The limit is the input representation.



\*\*The turn hypothesis in the plan document is REJECTED.\*\* The plan claims a

sequence model wins on turns. Measured: straight-line error compounds faster

(h^1.60) than turning error (h^1.45), and the turn ratio \*falls\* from 2.19×

to 1.41× as the horizon grows. Do not repeat the plan's claim.



\*\*Dead reckoning is unbeatable at short horizons.\*\* 0.70 units median error

at 62.5 ms — under two centimetres, against a 32-unit player. Nothing to

improve below \~250 ms.



\## Caveats



\- Horizon 4 was still improving at epoch 10 in both seeds, so 250 ms figures

&#x20; are likely slight underestimates.

\- Latency measured on a laptop CPU under Windows, single thread. Honest for

&#x20; this machine; not a server benchmark.

\- One prediction across the blackout, not an autoregressive rollout.

\- Where the 90th-percentile error crosses a player's width (32 units) falls

&#x20; between 250 and 500 ms for both methods and was not resolved.



\## Files



`h{horizon}\_hid{size}\_seed{n}.txt` — training runs

`horizon\_sweep.\*` — dead reckoning across 7 horizons, error grows as h^1.57

`baselines\_h1.txt` — null and DR at 62.5 ms, with turn breakdown

`latency\_h12.txt` — number #3

`burst\_loss.\*` — number #2

