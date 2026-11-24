# Rare patterns in recursive generation

**Research problem.** Can a generator learn where to spend a fixed budget of verified real examples so it retains rare patterns and still adapts when the source changes?

The 24-generation pilot compared eight methods across 256 independent worlds per regime. Each recursive learner using real data trained on 384 synthetic and 64 real examples per round. The rare-context intervention chose where those 64 examples came from.

| Regime | Balanced recursive rare KL | Tail-enriched recursive rare KL | Paired reduction and 95% interval | Change in population KL |
|---|---:|---:|---:|---:|
| Stationary | 0.394 | 0.295 | 0.099 [0.092, 0.106] | +0.123 [0.114, 0.133] |
| Conditional source shift | 0.398 | 0.297 | 0.101 [0.094, 0.109] | +0.127 [0.118, 0.137] |

Lower KL is better. Targeting rare contexts improved their density estimates and failed the population guardrail. Balanced accumulation of real examples gave lower rare-context and overall loss in both regimes than any recursive method. On changed contexts alone, the four-round real-only buffer reached **0.224 nats** versus **0.363** for cumulative real-only learning. The rolling buffer lost to accumulation on the stationary population.

These two controls motivate the next question. **Can a learner retain a small stable real archive and detect when a context needs its own short memory?** Compare adaptive windows with tuned fixed-window, cumulative, discounting and Bayesian online change-point controls. Separate abrupt, gradual and recurrent shifts using fresh evaluation worlds. Fix real query counts, archive capacity and generation budget. In a second stage, replace known context labels with observed sequence prefixes and score factual recovery, calibration and rare-composition likelihood separately.

The first pilot uses a smoothed table and selectively queryable simulator data. It isolates conditional sampling, with sequence-model representation learning still to test. The existing recursive pseudo-labelling pilot studies a different mechanism on real inputs. Shumailov et al. motivate tracking rare patterns and Gerstgrasser et al. motivate an accumulation control. [Model collapse paper](https://doi.org/10.1038/s41586-024-07566-y) · [Accumulation paper](https://arxiv.org/abs/2404.01413)

## Reproduce

With Python and NumPy installed, run from this directory:

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run.py --output replay.json
cmp results.json replay.json
```

The saved replay reproduced byte for byte. Peak process RSS was about 163 MiB on the replay.

[Protocol](Protocol.md) · [All outcomes](results.json)
