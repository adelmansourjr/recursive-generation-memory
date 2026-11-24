"""Replay the frozen study in separate 32-world Bayesian batches."""
import numpy as np

import run


full_bayes_forecasts = run.bayes_forecasts


def bounded_bayes_forecasts(params, archive, evidence):
    # Worlds are conditionally independent. Keep the full joint ledger and its
    # generation RNG unchanged, but split evaluation of their trusted predictor.
    compact = np.empty((len(evidence), len(params), *archive.shape))
    for start in range(0, len(archive), 32):
        row_slice = slice(start, start + 32)
        forecasts = full_bayes_forecasts(params, archive[row_slice], evidence[:, row_slice])
        for step, forecast in enumerate(forecasts):
            compact[step, :, row_slice] = forecast
    yield from compact


if __name__ == "__main__":
    run.bayes_forecasts = bounded_bayes_forecasts
    run.main()
