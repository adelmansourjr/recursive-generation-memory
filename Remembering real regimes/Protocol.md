# Reusing verified examples after a source returns

## Research question

The previous pilot found that permanent accumulation preserved rare contexts, while a short real-data window recovered faster after a shift. Test whether a learner can restore a useful verified archive when a familiar context returns, while preserving familiar-context coverage and adapting to sources it has never archived.

An online changepoint model can place probability on two kinds of reset. One starts with a diffuse prior and the other starts with the original verified counts for that context. Both are updated with exactly the same new verified observations. The archive is never refreshed using generated examples. This is a comparison of reset priors in a finite model, based on Bayesian online changepoint detection. It is not a novel general-purpose changepoint algorithm.

## Budget and data

Use 12 contexts and 12 motifs. Eight contexts are common and four are rare. Draw one immutable archive of 2,048 real examples per world. After that, every fitted method receives the same 64 fresh examples per round for 36 rounds. The quotas range from five to six examples per context. Learners see verified counts, generated samples in the recursive comparison, and context IDs. The latent motif distributions, event times and shifted-context identities are never passed to learners.

Each world has two affected rare contexts and four affected common contexts. Independently sample a stationary regime, a permanent abrupt shift, a gradual transition, a partial return to the initial source (A → B → A′), or a recurrence without a matching initial archive (A → B → C → B′). The initial archive and subsequent probabilities are newly generated, including independent motif vectors rather than swapping two fixed motif ranks as in the first pilot. Return distributions have 0–12% contamination from a different source. The comparison includes 192 worlds per reported regime.

## Models and freeze

Compare cumulative verified counts, a tuned fixed rolling window, tuned exponential forgetting, a diffuse-reset Bayesian online changepoint detector (BOCPD), and a fixed blend of a rolling estimate with the verified archive. The candidate uses the same changepoint inference and chooses between diffuse and archived reset hypotheses. Include a latest-batch-only and a generated-only control as descriptive diagnostics.

In the main recursion experiment each fitted policy generates exactly 384 samples from its own previous-round conditional model under the same estimated context prior. It mixes these counts with 24 per-context pseudo-observations from its current trusted estimator. At every round the synthetic draw supplies the same sample count and the verified estimators have the same acquisition budget. Verified-only controls remove synthetic feedback. This design tests equal evidence and generation budgets, not equal runtime. Both BOCPD variants maintain the same immutable archive and have the same run-length horizon. The candidate evaluates one additional prior hypothesis per possible segment.

Choose hyperparameters on five distinct 64-world development regimes using equal weighting of all-round population KL and the focal rare-context KL. The criterion uses the recursive branch. Select one fixed strongest competitor on development, using the same objective across established baselines, and freeze all variants and source hashes before scoring confirmation seeds. Do not pick a winner or reconfigure after reading confirmation results. All comparisons with individual methods are retained.

## Endpoints

The primary endpoint is mean exact conditional KL (natural logs) on affected rare contexts over the first eight rounds starting at the return in A → B → A′. This evaluates each post-update conditional prediction under the true current source. Lower is better. Take a paired interval by resampling 192 whole worlds 5,000 times. The practical target is a reduction of at least 0.015 nats with a positive lower 95% endpoint against the validation-selected competitor. Require the upper endpoint of the equally weighted population-KL degradation across the five regimes to remain below +0.015 nats. Report rare motif mass underestimated by half, the five regime-specific comparisons, the direct verified-only contrast, and the A → B → C → B′ setting even if the primary target passes. The primary and guardrail are specified together. Remaining intervals are descriptive and are not multiplicity-adjusted.

This is a finite conditional-density study. The generated text analogy is deliberate, but the benchmark uses synthetic categorical counts and known contexts. The statement about return-to-archive prediction depends on the existence of informative verified provenance.

[Model-collapse advert](https://www.findaphd.com/phds/project/when-ai-learns-from-ai-understanding-and-preventing-model-collapse/?p202576) · [Shumailov et al.](https://doi.org/10.1038/s41586-024-07566-y) · [Gerstgrasser et al.](https://arxiv.org/abs/2404.01413) · [Adams and MacKay](https://arxiv.org/abs/0710.3742)
