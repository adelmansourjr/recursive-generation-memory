# Which real examples preserve rare patterns in recursive generation?

Can we prevent repeated sampling and refitting from eroding rare conditional structures without sacrificing the ability to learn a changed source? Use the first experiment to separate renewed real evidence, diversity of the real prompts, and the value or harm of synthetic examples.

## Simulation

Use an explicit two-token generative model: draw a context, then draw a motif given that context. There are 12 contexts and 12 motifs, with four known low-frequency contexts. For each of 256 independent worlds, vary the motif probabilities and draw a finite 2,048-example real anchor. Give every method the same anchor. Fix the empirical context prior for synthesis and identify rare contexts from the anchor alone. Every new generation of a recursive reader samples 384 complete synthetic examples from its predecessor and refits its conditional table. Its real-data budget is 64 fresh examples per generation, including diagnostic controls that use less. Real examples are from a simulator that permits queries for a particular context. Use identical within-context real sample prefixes for methods that request different counts. Keep the true distributions inaccessible to the learners.

Use two regimes with independent seed lists. Keep the source fixed in one. In the other, shift half the rare contexts and half the common contexts at round 9, by swapping each selected context's most and least likely motif. No learner is informed of this change. Dynamics and the final evaluation measure use the complete true conditional distribution, so the comparison introduces no test-sample noise.

## Comparisons

All methods use Dirichlet smoothing at 0.25 per motif. Compare recursion alone, recursion with fresh random examples, recursion with a balanced context quota, and recursion with 38 of 64 real acquisitions allocated to the four rare contexts. Drop synthetic examples in a matched-acquisition ablation of the rare-context rule. Include a stronger rolling real-only estimator on the same targeted acquisitions, a cumulative real-only estimator on balanced acquisitions, and the untouched initial anchor. The real-only methods spend the same fresh acquisition budget and use fewer or equal training records. Their time and memory costs differ. Every generative recursion arm has the same per-round training budget when fresh data are supplied.

## Readouts and decision rule

Use the mean excess log loss against the true distribution on the anchor-defined rare contexts across rounds 21–24. Lower is better. Compare the tail-enriched recursive arm to the otherwise identical balanced arm on paired worlds. The practical criterion is a 0.02-nat reduction in rare excess log loss, with a positive lower endpoint on a 95% interval and an upper confidence endpoint below 0.03 nats for whole-population degradation. Check the real-only baselines even if this matched recursive contrast passes. Report population excess log loss, probability mass of motifs produced at under half their true conditional rate, and loss restricted to changed contexts. Use 5,000 paired bootstrap draws over worlds. Report both regimes, all controls, and the generated and acquired counts. The bootstrap quantifies uncertainty within this simulator.

A positive result here would separate acquisition from repeated refitting in a finite generative model. The model isolates sampling and estimation, while decoding heuristics, learned representations and semantic errors are left to a sequence-model experiment. The context-conditioned query interface is an experimental assumption. Contrast this with the existing Bristol workspace study that recursively pseudo-labels real inputs, and with the accumulation/replacement distinction in the literature.

[Moreno-Pino's advert](https://www.findaphd.com/phds/project/when-ai-learns-from-ai-understanding-and-preventing-model-collapse/?p202576) · [Shumailov et al.](https://doi.org/10.1038/s41586-024-07566-y) · [Gerstgrasser et al.](https://arxiv.org/abs/2404.01413)
