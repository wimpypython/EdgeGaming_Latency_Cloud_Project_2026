# ML track — results

*Every figure below came out of a recorded run. Where a number is a single
observation rather than a replicated one, it says so.*

---

## 1. The problem, stated as a measurement

When packets are lost, the server must believe something about where the
player is. Current netcode dead reckons: `pos + vel * dt`. The question
this track answers is whether a sequence model does better, and by how
much.

Everything is measured in **game units** of Euclidean error against
ground truth, which is the player's true position in the next packet that
arrives. A player hitbox is about 32 units wide.

## 2. The baselines, before any model

Measured on the 5 held-out games (112,613 examples), at the 750 ms
horizon:

| method | median | 90th | mean |
|---|---|---|---|
| null — assume no movement | 61.24 | 166.58 | 70.12 |
| dead reckoning | **33.59** | 111.44 | 46.95 |

Dead reckoning is 45.2% better than doing nothing, so **33.59 is the bar**.
Beating the null baseline is not a result and is not reported as one.

The split is **by whole game**, never randomly. Consecutive examples share
15 of their 16 history frames, so a random split would place near
duplicates on both sides and inflate the score enormously. The split
function lives in one file and is imported by every script that needs it.

## 3. Where dead reckoning fails, and why a sequence model should help

Dead reckoning uses one velocity vector and discards the other 15 frames.
Its error is therefore concentrated where velocity is about to change.
Measured on held-out data at 750 ms, split by the player's current speed:

| speed (u/s) | n | null | dead reckoning |
|---|---|---|---|
| 0–10 (standing still) | 26,434 | 0.65 | 0.94 |
| 10–100 | 28,647 | 39.34 | 31.72 |
| 100–200 | 34,019 | 83.01 | 46.02 |
| 200+ | 23,513 | 161.64 | 65.73 |

Two things follow. **23.5% of real frames are a player standing still**,
where every method is near-perfect and nothing can be won. And DR's error
grows with speed, which is where a model that can see deceleration and
turning has something to contribute.

## 4. The model

A single-layer GRU, 64 hidden units, ~14k parameters. Input is 16 frames
of position and velocity spanning 1000 ms; output is a position at a
fixed horizon.

The one design decision that matters is **delta framing**: the current
position is subtracted from every history frame and added back to the
output, inside `forward()`. Without it the model spends its capacity
memorising the map — that mid is near x=500, B site near y=2500 — and
would fail on any other map. With it, the model sees only displacement,
which also makes the comparison with dead reckoning like-for-like, since
DR is itself a displacement method.

Framing invariance was verified, not assumed: the same motion moved 1500
units across the map changes the prediction by **0.0001 units**.

## 5. Offline result — three horizons, clean validation split

Each horizon was trained twice. The first runs selected the saved epoch on
the **held-out games**, which is model selection on the test set. The
retrain splits the 20 non-test games into 16 train and 4 validation,
chooses the epoch on validation, and touches the test set **once**, after
training has finished. The test games are identical in both
(6, 7, 11, 14, 17), so the two are directly comparable.

| horizon | DR median | GRU median | clean gain | earlier, biased | DR p90 | GRU p90 | p90 gain |
|---|---|---|---|---|---|---|---|
| 250 ms | 6.32 | 5.76 | **9.0%** | 9.5% | 25.13 | 19.42 | **22.7%** |
| 500 ms | 18.62 | 17.21 | **7.6%** | 8.0% | 68.25 | 53.05 | **22.3%** |
| 750 ms | 33.59 | 29.80 | **11.3%** | 10.0% | 111.44 | 84.62 | **24.1%** |

**The selection bias was worth about half a point.** 250 ms fell 0.5 and
500 ms fell 0.4; 750 ms rose 1.3, which is confounded — its best epoch
was the last one, so that run trained longer than the original's ten
epochs. The old numbers were not carrying a meaningful inflation, and
the figures above need no caveat about how the epoch was chosen.

The clean retrain also trains on less data — 16 games instead of 20,
439,866 examples instead of 543,011 — so these numbers are conservative
on two counts at once.

**The gain is 7.6-11.3% at the median and 22.3-24.1% at the 90th
percentile.** The median is roughly flat across horizons; what grows is
the absolute error removed, 0.6 units at 250 ms to 3.8 at 750 ms, because
DR itself degrades from 6.3 to 33.6.

Doubling the hidden size to 128 improved the biased 500 ms result from
17.14 to 16.82 — **1.8% for 4x the parameters** — which is why the ladder
uses hidden 64.

**Note for the next run:** at 750 ms the chosen epoch was the twelfth of
twelve, so the model had not stopped improving. Training longer is
legitimate here because the epoch is selected on validation, not test.

## 6. Latency — is it usable at all

Single CPU thread, no GPU, measured over 200 calls:

| | result |
|---|---|
| inference, median | 1.07 ms |
| inference, 99th percentile | 2.76 ms |
| frame budget at 60 fps | 16.67 ms |

**6.4% of one frame at the median.** The claim that this could run in a
real server loop is measured, not asserted.

## 7. Live result — the part that tests the offline number

The offline figure is a claim about a dataset. To make it a claim about
the system, the recorded trajectories were replayed over UDP through the
real server, with burst loss whose parameters reproduce what the netem
run measured on EC2 (5.69% loss, 610 ms mean burst).

**Nothing was synthesised.** The dataset is 16 Hz and the replay sends at
16 Hz, one packet per recorded frame, positions and velocities exactly as
recorded. Held-out games only, using the same split function as training.

Six runs: two rung-selection policies x three loss seeds, ~280 gaps each
on ~145 real trajectories. The same seed under both policies produces the
identical gaps, so the policies are compared on identical traffic.

### The headline

| metric | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|
| median error, GRU vs DR | +12.2% | +17.1% | +4.9% | **+11.4%** |
| 90th percentile | +19.1% | +16.2% | +29.5% | **+21.6%** |
| maximum | +19.5% | +61.0% | +9.6% | **+30.0%** |

*(`nearest` policy; DR measured on the same gaps the model answered.)*

**The 90th percentile improved in all six runs, across both policies,
between 16% and 31%.** That is the result the project rests on, because
it is the one that survived every run.

**And it agrees with the offline measurement.** The clean retrain puts
the offline p90 gain at 22.3-24.1% across the three horizons; the live
runs put it at 21.6% mean. Two measurements of the same quantity -- one
over 112,613 dataset windows, one over ~280 gaps through a UDP socket
under burst loss -- within two points of each other. The shrinkage
mechanism predicts a tail-weighted gain, and both measurements show one. The median improved in all three
runs of this policy, mean 11.4%, which is consistent with the 8-10%
measured offline — so the dataset result does transfer, within the
uncertainty three seeds can establish.

**The median is noisy and the range must be quoted with it.** Across
seeds it moved from +4.9% to +17.1%; under the other policy it ranged
from -10.0% to +15.1%. A single live run is not a result, and the first
two runs of this experiment -- before gap duration was derived from
sequence numbers -- disagreed by 7.7 points on identical traffic. Those
runs are superseded by the reproducible ones, not hidden: they measured a
quantity contaminated by scheduler jitter.

### Rung selection: a prediction that failed

Each checkpoint predicts at one fixed horizon; live gaps are any length.
Two rules for choosing which rung to ask were implemented and measured:

- **nearest** - the closest horizon in absolute time, which extrapolates
  past the horizon when the gap is longer than the rung.
- **cover** - the smallest horizon at least as long as the gap, so the
  model is never used beyond its trained range.

`cover` was expected to win, on the principle that a fixed-horizon model
should not be extrapolated. Under the measured loss, `nearest`
extrapolates on 53.6% of gaps by a mean factor of 1.59x, which looked
like an obvious defect.

It lost, on every seed:

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| GRU median, cover | 14.04 | 13.97 | 17.04 |
| GRU median, nearest | **13.04** | **13.64** | **14.73** |
| GRU p90, cover | **89.74** | **90.03** | **85.21** |
| GRU p90, nearest | 91.39 | 91.40 | 86.51 |

`nearest` is better at the median in all three (by 0.33-2.31 units);
`cover` is better at the 90th percentile in all three (by 1.3-1.7 units).
The maximum is identical under both, because gaps beyond 750 ms clamp to
the top rung either way -- which is a useful consistency check that the
policies differ only where they are supposed to.

The reason `cover` loses at the median is that interpolating back from a
longer-horizon prediction is a worse approximation than mildly
extrapolating a shorter-horizon one. The model's 750 ms displacement is
not 1.5x its 500 ms displacement -- real players decelerate and turn --
so scaling the long answer back understates near-term travel. The
prediction was reasonable and the measurement contradicted it, which is
why both policies are reported rather than one being chosen in advance.

### Where the model declines

The history must be 16 consecutive sequence numbers. Under bursty loss a
second burst close behind the first leaves a hole, and the model declines
rather than substituting a frame it does not have: **13.4-15.3% of gaps**
across the six runs. Those gaps fall back to dead reckoning, and the
count is reported with every result so the model is never credited with
gaps it did not answer.

### Why the gain is in the tail

The model systematically predicts shorter displacement than DR - 0.97x at
250 ms, 0.91x at 500 ms, 0.84x at 750 ms on straight motion - because it
was trained with squared error on players who frequently slow and stop,
so the conditional mean displacement really is shorter than
constant-velocity extrapolation. The shrinkage is **conditional, not a
constant**: at 750 ms it is 0.84x on straight motion and 0.59x at 180/s
of turning. The model learned to shorten more when it sees a player
cutting, which is learned structure rather than a scale factor.

That is why it bounds the tail. Dead reckoning's failure mode is
unbounded overshoot during a long outage - its worst case reached 1,178
units in one run against the model's 460 - and a shrunk prediction cannot
overshoot as far. It is also why the median gain is modest: 42% of live
gaps are under 375 ms, where DR's error is about 3 units and there is
almost nothing to win.

## 8. Honest limitations

**Checkpoint selection touched the test set, and has been fixed.** The
original `train.py` saved the epoch scoring best on the held-out games,
which is selection on the test set. `train_clean.py` adds a validation
split and the retrained figures in section 5 are free of it. The bias
turned out to be about half a point. The earlier numbers are reported
alongside rather than deleted, since the comparison is what establishes
the size of the bias.

**The live comparison is over hundreds of gaps, not hundreds of
thousands.** The offline figure rests on 112,613 examples; each live run
on about 280 gaps, of which the model answers ~240. Deriving gap duration
from sequence numbers made runs bit-identical on repeat, but three seeds
is a small sample: the median gain ranged from +4.9% to +17.1% across
them. Any live median quoted without that range is overstated.

**The replay runs at 16 Hz, the dataset's own rate.** Sending at the game
tick rate of 64 Hz would require inventing three positions between every
pair of recorded frames, so it was not done. The 64 Hz figures in this
project come from the synthetic generator, and the two are not
interchangeable.

**Three seeds is the minimum, not a comfortable number.** The spread on
the median is wide enough that a fourth seed could move the mean by
several points. The 90th-percentile figure is far steadier - positive in
all six runs, 16-31% - which is why it leads. More seeds would narrow
both and cost about five minutes each.

**The synthetic generator is not a substitute for recorded play.** It was
built to CSKnow's speed envelope but not its turn and stop statistics:
median 200 u/s and essentially never stationary, against real play's
median 100 u/s and 25.3% of frames standing still. On that traffic DR is
near-optimal and the comparison cannot discriminate — measured, as a GRU
loss of 40% against DR, before the replay client existed.
