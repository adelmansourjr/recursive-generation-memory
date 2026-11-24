# Recovering rare patterns when a verified source returns

Can a recursive generator reuse trustworthy old examples when a source returns, while adapting to a source it has never archived? This follow-up tests a context-level Bayesian predictor that considers either a diffuse prior or the verified original archive at each possible reset.

## Result

The locked experiment ran **960 fresh simulated worlds** across five regimes. Each fitted recursive method received the same **64 verified examples** and generated **384 samples** per round for 36 rounds. The candidate and the validation-selected diffuse-reset Bayesian detector used the same hazard, evidence and run-length horizon. Both kept the same verified initial archive, while the candidate could also use it as a reset prior.

When a known source returned, the candidate reduced affected rare-context excess log loss from **0.2331 to 0.1662 nats**, a **28.7% reduction**. The paired reduction was **0.0669 nats** with a 95% bootstrap interval of **[0.0605, 0.0736]**. This endpoint covers the first eight post-return rounds. Average population KL across the five regimes also fell by **0.00218 nats** with a 95% interval of **[0.00170, 0.00268]**. Both endpoints cleared the criteria set before confirmation.

| Regime | Focal rare KL, diffuse reset | Focal rare KL, archive reset | Paired rare-KL reduction, nats | Whole-stream population KL, diffuse / archive |
|---|---:|---:|---:|---:|
| Stationary | 0.0808 | 0.0803 | +0.0005 [−0.0010, +0.0020] | 0.0894 / 0.0891 |
| Abrupt change | 0.2632 | 0.2677 | −0.0045 [−0.0079, −0.0012] | 0.1341 / 0.1355 |
| Gradual change | 0.2087 | 0.2034 | +0.0053 [+0.0025, +0.0081] | 0.1356 / 0.1382 |
| Return to an archived source | 0.2331 | 0.1662 | **+0.0669 [+0.0605, +0.0736]** | 0.1555 / 0.1376 |
| Return to an unarchived source | 0.2275 | 0.2256 | +0.0019 [−0.0021, +0.0059] | 0.1714 / 0.1746 |

On the archived-source return, the fixed archive blend and the cumulative estimator achieved similar focal rare KL (0.1681 and 0.1626, respectively). The adaptive archive reduced whole-stream population KL in that regime relative to both (0.1376 versus 0.1723 and 0.2135). This distinction matters because focusing only on recovery after a return can favour methods that never forgot the earlier source. The discounting baseline was stronger in the early rare-context windows for abrupt and gradual shifts.

The improvement after an archived return persisted in the verified-only control: **0.0671 nats** lower focal rare KL versus the diffuse detector, with a 95% interval of **[0.0609, 0.0735]**. The evidence supports better reuse of verified observations under the tested recurrence assumptions. At the archived return, true motif mass whose probability was underestimated by at least half fell from 18.6% to 13.2%. See the full ledger for results for all comparators, regime means and unadjusted secondary intervals.

## Method

At each round, the predictor tracks the posterior probability of each possible last-reset time. The proposed version also tracks whether a new segment starts from the context's original verified counts or a symmetric Dirichlet prior. Both reset hypotheses are weighted by the exact Dirichlet-multinomial predictive likelihood. The forecast is the weighted mixture of conditional motif probabilities. The recovered archive contributes to the real-data estimate. Generated examples enter only the separate recursive learner and never update the verified archive or the reset likelihood.

The earlier study showed that accumulating real examples and dropping stale observations had different advantages. This one isolates whether the same archive can be used conditionally. The mixture weight was selected on disjoint development worlds with the same objective used to select the comparator. The synthetic data are categorical draws, with the source probability supplied only to the scorer. Language-model training, non-oracle context assignments and recurrence from a large collection of historical sources remain open experiments.

[Protocol](Protocol.md) · [Development freeze](freeze.json) · [Full results and world ledger](results.json) · [Prior pilot](../Recursive%20generation/README.md)

## Reproduce

With Python 3.12 and the dependencies in requirements.txt installed, run from this directory:

```sh
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
scratch=$(mktemp -d)
python replay.py develop --output "$scratch/freeze.json"
python replay.py confirm --freeze "$scratch/freeze.json" --output "$scratch/results.json"
cmp freeze.json "$scratch/freeze.json"
cmp results.json "$scratch/results.json"
```

Both files reproduced byte for byte in the pinned environment. The memory-bounded replay batches the trusted Bayesian forecast across independent worlds and preserves the full generation RNG and ledger. The reference confirmation implementation peaked at 493 MiB RSS. The batched development and confirmation replay together peaked at **241 MiB RSS**. A zero-hazard check recovered cumulative estimation. Independently enumerating all reset branches over four updates reproduced the Bayesian forecasts, and a second paired bootstrap from the world ledger gave [0.0606, 0.0735] nats for the primary reduction. The freeze stores the code and protocol SHA-256 fingerprints.

[Model-collapse advert](https://www.findaphd.com/phds/project/when-ai-learns-from-ai-understanding-and-preventing-model-collapse/?p202576) · [Shumailov et al.](https://doi.org/10.1038/s41586-024-07566-y) · [Gerstgrasser et al.](https://arxiv.org/abs/2404.01413) · [Adams and MacKay](https://arxiv.org/abs/0710.3742)
