"""Paired conditional-density experiment for recursively generated data."""

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np


WORLDS = 256
CONTEXTS = MOTIFS = 12
ROUNDS = 24
FRESH = 64
SYNTHETIC = 384
SMOOTH = 0.25
METHODS = (
    "anchor", "synthetic_only", "random_recursive", "balanced_recursive",
    "tail_recursive", "tail_real_only", "tail_rolling_real", "balanced_cumulative_real",
)
ACQUISITIONS = {
    "random_recursive": "random",
    "balanced_recursive": "balanced",
    "tail_recursive": "tail",
    "tail_real_only": "tail",
    "tail_rolling_real": "tail",
    "balanced_cumulative_real": "balanced",
}
RECURSIVE = ("synthetic_only", "random_recursive", "balanced_recursive", "tail_recursive")
METRICS = ("rare_kl", "population_kl", "underproduced_mass", "shifted_kl", "rare_real_count")


def normalize(table):
    return table / table.sum(axis=-1, keepdims=True)


def estimate(counts):
    return normalize(counts + SMOOTH)


def quota(group_count, total):
    """Allocate a fixed count with randomised order already supplied."""
    return np.full(group_count, total // group_count) + (np.arange(group_count) < total % group_count)


def real_counts(truth, rare, prior, rng):
    """Reuse the first n real draws from each context for all policies."""
    worlds = len(truth)
    order = rng.random((worlds, CONTEXTS)).argsort(axis=1)
    rare_order = np.take_along_axis(rare, order, axis=1)
    rare_positions = np.cumsum(rare_order, axis=1) - 1
    common_positions = np.cumsum(~rare_order, axis=1) - 1
    targeted = np.where(
        rare_order, quota(4, 38)[rare_positions], quota(8, FRESH - 38)[common_positions],
    )
    inverse = order.argsort(axis=1)
    budget = {
        "random": rng.multinomial(FRESH, prior),
        "balanced": np.broadcast_to(quota(CONTEXTS, FRESH), (worlds, CONTEXTS)),
        "tail": np.take_along_axis(targeted, inverse, axis=1),
    }
    # Vary which contexts receive the balanced quota's remainder.
    budget["balanced"] = np.take_along_axis(budget["balanced"], inverse, axis=1)
    assert all(np.all(x.sum(axis=1) == FRESH) for x in budget.values())
    draws = rng.random((worlds, CONTEXTS, FRESH))
    labels = (draws[..., None] >= truth.cumsum(axis=2)[:, :, None, :]).sum(axis=3)
    one_hot = np.eye(MOTIFS, dtype=np.int16)[labels]
    prefixes = np.concatenate(
        (np.zeros((worlds, CONTEXTS, 1, MOTIFS), dtype=np.int16),
         np.cumsum(one_hot, axis=2, dtype=np.int16)),
        axis=2,
    )
    rows, groups = np.indices((worlds, CONTEXTS))
    return {name: prefixes[rows, groups, count] for name, count in budget.items()}


def measure(q, truth, prior, rare, shifted, acquired):
    excess = (truth * np.log(truth / q)).sum(axis=2)
    missing = (truth * (q < truth / 2)).sum(axis=2)
    return np.column_stack((
        (excess * rare).sum(axis=1) / 4,
        (excess * prior).sum(axis=1),
        (missing * rare).sum(axis=1) / 4,
        (excess * shifted).sum(axis=1) / 6,
        (acquired.sum(axis=2) * rare).sum(axis=1),
    ))


def simulate(seed, drift):
    rng = np.random.default_rng(seed)
    prior = rng.dirichlet([30] * 8 + [3] * 4, size=WORLDS)
    truth = rng.dirichlet([0.7] * MOTIFS, size=(WORLDS, CONTEXTS))
    rows, groups = np.indices((WORLDS, CONTEXTS))
    anchor = rng.multinomial(rng.multinomial(2048, prior), truth)
    initial = estimate(anchor)
    estimated_prior = (anchor.sum(axis=2) + 1) / (2048 + CONTEXTS)
    ranking = estimated_prior.argsort(axis=1)
    rare = np.zeros_like(prior, dtype=bool)
    rare[np.arange(WORLDS)[:, None], ranking[:, :4]] = True
    # Shift 2 rare and 4 common contexts. The chosen contexts vary by world.
    shifted = np.zeros_like(rare)
    shifted[np.arange(WORLDS)[:, None], ranking[:, [0, 2, 4, 6, 8, 10]]] = True
    swapped = truth.copy()
    best = truth.argmax(axis=2)
    worst = truth.argmin(axis=2)
    swapped[rows, groups, best] = truth[rows, groups, worst]
    swapped[rows, groups, worst] = truth[rows, groups, best]
    changed = np.where(shifted[..., None], swapped, truth)
    states = {name: initial for name in METHODS}
    zeros = np.zeros_like(anchor)
    accumulated = anchor.copy()
    recent = deque(maxlen=4)
    history = {name: [] for name in METHODS}
    for round_no in range(1, ROUNDS + 1):
        current = changed if drift and round_no >= 9 else truth
        evidence = real_counts(current, rare, prior, rng)
        generated_contexts = rng.multinomial(SYNTHETIC, estimated_prior)
        synth_seed = int(rng.integers(1 << 62))
        accumulated += evidence["balanced"]
        recent.append(evidence["tail"])
        for name in METHODS:
            real = evidence[ACQUISITIONS[name]] if name in ACQUISITIONS else zeros
            if name in RECURSIVE:
                simulated = np.random.default_rng(synth_seed).multinomial(generated_contexts, states[name])
                states[name] = estimate(real + simulated)
            elif name == "tail_real_only":
                states[name] = estimate(real)
            elif name == "tail_rolling_real":
                states[name] = estimate(sum(recent, start=zeros))
            elif name == "balanced_cumulative_real":
                states[name] = estimate(accumulated)
            history[name].append(measure(states[name], current, prior, rare, shifted, real))
    return {name: np.stack(values) for name, values in history.items()}


def paired_interval(values, rng):
    draws = rng.integers(len(values), size=(5000, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), (0.025, 0.975))
    return [float(values.mean()), float(low), float(high)]


def summarize(history, seed):
    rng = np.random.default_rng(seed)
    endings = {name: values[-4:].mean(axis=0) for name, values in history.items()}
    candidate = endings["tail_recursive"]
    results = {
        "last_four_rounds": {name: dict(zip(METRICS, v.mean(axis=0).tolist())) for name, v in endings.items()},
        "rare_nll_improvement_against": {
            name: paired_interval(v[:, 0] - candidate[:, 0], rng)
            for name, v in endings.items() if name != "tail_recursive"
        },
        "population_nll_degradation_vs_balanced": paired_interval(candidate[:, 1] - endings["balanced_recursive"][:, 1], rng),
        "curves": {
            name: [dict(zip(METRICS, point)) for point in values.mean(axis=1).tolist()]
            for name, values in history.items()
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {
        "worlds_per_regime": WORLDS, "rounds": ROUNDS,
        "real_budget_per_round": FRESH, "synthetic_budget_per_round": SYNTHETIC,
        "stationary": summarize(simulate(63127, False), 179),
        "context_shift": summarize(simulate(71489, True), 181),
    }
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
