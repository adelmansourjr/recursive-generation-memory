# Selective archive reuse in autoregressive sequence models

**Ask when a single archive helps some prefixes and hurts others.** A model may recognise a returning source from common passages and still recover the wrong rare continuation. This experiment moves the previous conditional-count study into actual autoregressive sequences and introduces a seemingly familiar source whose rare structure is absent from the verified archive.

## What held up

The locked test fitted trigram models on **768 fresh sequence simulations**. Every recursive arm received **20 verified sequences and 48 self-generated sequences** per round for **30 rounds**. Each learner had the same three candidate archives. The candidate compared a retrieved archive with a diffuse prior separately for each two-token prefix, with candidate segment and source probabilities updated from verified observations. The tuned, diffuse-reset Bayesian selector was fixed as the strongest comparator on separate development seeds.

| Measure | Comparison | Reference → selective archive | Paired reduction [95% interval], nats |
|---|---|---:|---:|
| Rare-prefix KL during mismatched returns | Development-selected diffuse-reset detector | 0.235155 → 0.235658 | −0.00050 [−0.00328, +0.00240] |
| Population KL across all six regimes | Development-selected diffuse-reset detector | 0.126722 → 0.113781 | **+0.01294 [+0.01246, +0.01341]** |
| Rare-prefix KL during mismatched returns | Whole-archive resets | 0.373811 → 0.235658 | **+0.13815 [+0.13117, +0.14515]** |
| Rare-prefix KL during exact archived returns | Development-selected diffuse-reset detector | 0.241171 → 0.138099 | +0.10307 [+0.09764, +0.10854] |

The preregistered improvement target against the diffuse detector **did not pass** on the primary rare-prefix mismatch endpoint. Average population KL fell **10.2%** and passed its separate safeguard. The grouped whole-archive comparison is a descriptive ablation under the same endpoints and budgets. Selective use reduced mismatch loss by **37.0%** against whole-archive resets. On fully matched returns, whole-archive resets scored **0.119094** rare-prefix KL and retained their advantage over selective reuse (**0.138099**). The conditional guard protected against wrong positive transfer at the cost of being less decisive when the full archive was correct.

| Regime | Early rare KL, diffuse / whole archive / selective | Population KL, diffuse / selective |
|---|---:|---:|
| Stationary | 0.1102 / 0.0826 / 0.0858 | 0.0767 / 0.0661 |
| Known return | 0.2412 / 0.1191 / 0.1381 | 0.1338 / 0.1198 |
| Partly changed return | 0.2389 / 0.3763 / 0.2390 | 0.1342 / 0.1234 |
| New source with a misleading archive | 0.2314 / 0.3713 / 0.2323 | 0.1342 / 0.1232 |
| Gradual drift | 0.1951 / 0.1997 / 0.1818 | 0.1347 / 0.1080 |
| Unarchived return | 0.2525 / 0.2523 / 0.2442 | 0.1467 / 0.1422 |

The verified-only control reproduced the main mechanism. Its mismatch rare-prefix reduction versus diffuse reset was **0.00027 nats** with interval **[−0.00220, +0.00275]**. Versus whole-archive resets, it was **0.13513 [0.12879, 0.14153]** in the descriptive grouped check. The advantage comes from how evidence is reused and does not require recursive synthetic feedback. Rolling windows, discounting, cumulative estimation, and the global and prefix-level retrieval baselines are included in the ledger.

## Why the sequence setup matters

Each sample is a real draw from an autoregressive token process: predict a token given the preceding two, append it, then condition the next prediction on that updated prefix. Verified and self-generated inputs are fitted as sequence transition counts. A wrong transition therefore affects both the fitted distribution and which states the model encounters in its next training data. The evaluator computes expected KL per next-token prediction using the true occupancy of each prefix, summing over all generated positions. This avoids scoring an arbitrary uniform mixture of frequent and rare prefix states.

The next scientific question is whether the learner can infer these local disagreements from learned representations on text, where prefix identity, verified provenance and recurrence are imperfectly observed. This synthetic benchmark isolates the archive-reuse mechanism in small Markov sequence learners. It does not evaluate neural language models or preservation of natural-language facts. Use the [protocol](PROTOCOL.md), [frozen selection](freeze.json), and [all runs and paired world-level data](results.json) to inspect variations.

[Prior verified-archive experiment](../Remembering%20real%20regimes/README.md) · [PhD advert](https://www.findaphd.com/phds/project/when-ai-learns-from-ai-understanding-and-preventing-model-collapse/?p202576) · [Bayesian changepoint inference](https://arxiv.org/abs/0710.3742) · [Primary model-collapse study](https://doi.org/10.1038/s41586-024-07566-y)

## Reproduce

Use Python 3.12 and the pinned dependencies:

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
scratch=$(mktemp -d)
python run.py freeze --output "$scratch/freeze.json"
python run.py confirm --freeze "$scratch/freeze.json" --output "$scratch/results.json"
cmp freeze.json "$scratch/freeze.json"
cmp results.json "$scratch/results.json"
```

The freeze hashes the experimental code and protocol. Independent checks reproduced the prefix-occupancy calculation for an IID source, the soft acceptance posterior for a single prefix, the cumulative and whole-archive limiting cases, and the paired effects from the world-level ledgers. The independently bootstrapped intervals for the grouped whole-archive ablation also use 20,000 draws. Both saved outputs reproduced byte for byte in 36.2 seconds at **201 MiB peak RSS**. The original confirmation process peaked at **226 MiB RSS**.
