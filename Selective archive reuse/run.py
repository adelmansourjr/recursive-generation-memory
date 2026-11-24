"""Controlled autoregressive sequence experiment with decoy verified archives."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import gammaln, softmax


V = 6
STATES = V * V
STEPS = 14
ROUNDS = 30
REAL = 20
SYNTHETIC = 48
ARCHIVE = 160
PSEUDO = 16
ALPHA = 0.25
BY_WORLD = 16
HEAD = (np.arange(STATES) // V < 4) & (np.arange(STATES) % V < 4)
INITIAL = np.array([.34, .26, .19, .15, .04, .02])
TOKEN_MASS = np.array([.34, .27, .19, .14, .04, .02])
REGIMES = ("stationary", "known_return", "partial_return", "decoy_return", "gradual", "unarchived_return")
TARGET = ("partial_return", "decoy_return")
DEV_SEEDS = (224701, 224737, 224771, 224809, 224863, 224879)
TEST_SEEDS = (716047, 716071, 716101, 716117, 716243, 716269)
DEV_N, TEST_N = 48, 128
MODELS = {
    "cumulative": ({},),
    "rolling": tuple({"window": w} for w in (3, 8, 14)),
    "discount": tuple({"decay": d} for d in (.60, .82, .94)),
    "global_retrieval": tuple({"window": w} for w in (4, 10)),
    "prefix_retrieval": tuple({"window": w} for w in (4, 10)),
    "diffuse_reset": tuple({"hazard": h} for h in (.04, .10)),
    "whole_archive_reset": tuple({"hazard": h} for h in (.04, .10)),
    "prefix_gated_reset": tuple({"hazard": h, "trust": r} for h in (.04, .10) for r in (.55, .85)),
}
CANDIDATE = "prefix_gated_reset"
METRICS = ("population_kl", "focal_rare_kl", "focal_rare_underprediction_mass", "whole_stream_rare_kl")
HERE = Path(__file__).resolve().parent


def normalize(values):
    return values / values.sum(axis=-1, keepdims=True)


def source_fingerprint():
    return {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in ("run.py", "PROTOCOL.md")
    }


def transition_counts(q, n_samples, rng):
    """Fit a trigram count table to actual independently drawn token sequences."""
    worlds = len(q)
    row = np.arange(worlds)[:, None]
    first = rng.choice(V, size=(worlds, n_samples), p=INITIAL)
    second = rng.choice(V, size=(worlds, n_samples), p=INITIAL)
    state = first * V + second
    out = np.zeros((worlds, STATES, V), dtype=np.int32)
    for _ in range(STEPS):
        cut = q[row, state].cumsum(axis=-1)[..., :-1]
        token = (rng.random((worlds, n_samples, 1)) >= cut).sum(axis=-1)
        index = row * STATES * V + state * V + token
        out += np.bincount(index.ravel(), minlength=worlds * STATES * V).reshape(worlds, STATES, V)
        state = (state % V) * V + token
    assert np.all(out.sum(axis=(1, 2)) == n_samples * STEPS)
    return out


def occupation(q):
    """Exact state-frequency weights per conditional prediction."""
    p = np.broadcast_to((INITIAL[:, None] * INITIAL).ravel(), (len(q), STATES))
    total = p.copy()
    for _ in range(STEPS - 1):
        flow = (p[..., None] * q).reshape(len(q), V, V, V)
        p = flow.sum(axis=1).reshape(len(q), STATES)
        total += p
    return total / STEPS


def make_worlds(regime, worlds, seed):
    rng = np.random.default_rng(seed)
    a, b, c, novel, wrong_tail, e = rng.dirichlet(4.3 * TOKEN_MASS, size=(6, worlds, STATES))
    # Smooth the generator slightly to keep all motifs in support.
    a, b, c, novel, wrong_tail, e = [
        .95 * source + .05 * TOKEN_MASS for source in (a, b, c, novel, wrong_tail, e)
    ]
    decoy = np.where(HEAD[None, :, None], novel, wrong_tail)
    first = np.where(rng.integers(2, size=(worlds, 1, 1)) == 0, a, b)
    archived = np.stack([transition_counts(src, ARCHIVE, rng) for src in (a, b, decoy)], axis=1)
    onset = rng.integers(8, 12, (worlds, 1, 1))
    intermediate = rng.integers(15, 18, (worlds, 1, 1))
    returned = rng.integers(20, 23, (worlds, 1, 1))
    ramp = rng.integers(9, 14, (worlds, 1, 1))
    labels = []
    targets = []
    weights = []
    for t in range(ROUNDS):
        if regime == "stationary":
            q = first
        elif regime == "gradual":
            coefficient = np.clip((t - onset + 1) / ramp, 0., 1.)
            q = (1 - coefficient) * first + coefficient * novel
        else:
            q = np.where(t >= onset, c, first)
            if regime == "known_return":
                last = first
            elif regime == "partial_return":
                last = np.where(HEAD[None, :, None], first, e)
            elif regime == "decoy_return":
                last = novel
            elif regime == "unarchived_return":
                q = np.where(t >= intermediate, e, q)
                last = c
            else:
                raise ValueError(regime)
            q = np.where(t >= returned, last, q)
        labels.append(transition_counts(q, REAL, rng))
        targets.append(q)
        weights.append(occupation(q))
    focal_start = (onset if regime in ("stationary", "gradual") else returned).reshape(worlds)
    focus = (np.arange(ROUNDS)[:, None] >= focal_start) & (np.arange(ROUNDS)[:, None] < focal_start + 8)
    assert np.all(focus.sum(axis=0) == 8)
    return {
        "archive": archived,
        "labels": np.stack(labels),
        "targets": np.stack(targets),
        "weight": np.stack(weights),
        "generation_seeds": rng.integers(1 << 62, size=ROUNDS),
        "focus": focus,
    }


class DirichletBank:
    """Conditional Markov likelihood from transition counts, integrating row parameters."""

    def __init__(self, archive):
        self.priors = np.concatenate((np.zeros_like(archive[:, :1]), archive), axis=1)[:, None]
        max_count = int(archive.sum(axis=-1).max()) + REAL * STEPS * ROUNDS
        self.gamma = gammaln(np.arange(max_count + 1) + ALPHA)
        self.gamma_row = gammaln(np.arange(max_count + 1) + V * ALPHA)
        self.normalizer = self.row_log(self.priors)

    def row_log(self, counts):
        return self.gamma[counts].sum(axis=-1) - self.gamma_row[counts.sum(axis=-1)]

    def marginal(self, new):
        posterior = new[:, :, None] + self.priors
        return self.row_log(posterior) - self.normalizer, normalize(posterior + ALPHA)


def recent_retrieval(method, param, env):
    model = DirichletBank(env["archive"])
    for t in range(ROUNDS):
        new = env["labels"][max(0, t - param["window"] + 1): t + 1].sum(axis=0)
        logs, forecasts = model.marginal(new[:, None])
        logs, forecasts = logs[:, 0], forecasts[:, 0]
        if method == "global_retrieval":
            yield (softmax(logs.sum(axis=-1), axis=1)[:, :, None, None] * forecasts).sum(axis=1)
        else:
            yield (softmax(logs, axis=1)[..., None] * forecasts).sum(axis=1)


def simple_forecasts(method, param, env):
    seed = env["archive"].mean(axis=1) * .12
    current = seed.copy()
    for t in range(ROUNDS):
        fresh = env["labels"][t]
        if method == "rolling":
            current = env["labels"][max(0, t - param["window"] + 1):t + 1].sum(axis=0)
        elif method == "discount":
            current = param["decay"] * current + fresh
        else:
            current += fresh
        yield normalize(current + ALPHA)


def confidence_rows(logs, estimates, weight):
    """Integrate one independent archive-trust indicator per context row."""
    if weight == 1.:
        return logs, estimates
    with_archive = np.log(weight) + logs
    without = np.log1p(-weight) + logs[:, :, :1]
    union = np.logaddexp(with_archive, without)
    prob = np.exp(with_archive - union)
    return union, prob[..., None] * estimates + (1 - prob)[..., None] * estimates[:, :, :1]


def reset_inference(bank, labels, recipes):
    """Track the posterior over the most recent reset and archive source."""
    prefix = np.concatenate((np.zeros_like(labels[:1]), labels.cumsum(axis=0, dtype=np.int32)))
    worlds = len(labels[0])
    priors = np.full((len(recipes), 4), .25)
    for k, (family, _) in enumerate(recipes):
        if family == "diffuse_reset":
            priors[k] = [1., 0., 0., 0.]
    mass = np.zeros((len(recipes), worlds, ROUNDS, 4))
    mass[:, :, 0, :] = priors[:, None, :]
    previous = [
        np.zeros((worlds, 0, 4))
        for _ in recipes
    ]
    for t in range(ROUNDS):
        tensor = (prefix[t + 1] - prefix[:t + 1]).transpose(1, 0, 2, 3)
        logs, estimates = bank.marginal(tensor)
        preds = np.empty((len(recipes), worlds, STATES, V))
        # Hazard variants share the same likelihood terms and prediction rows.
        trust_cache = {}
        for j, (_, param) in enumerate(recipes):
            hazard = param["hazard"]
            trust = param.get("trust", 1.)
            if trust not in trust_cache:
                mixed_log, mixed_rows = confidence_rows(logs, estimates, trust)
                trust_cache[trust] = mixed_log.sum(axis=-1), mixed_rows
            log, rows = trust_cache[trust]
            increments = log.copy()
            increments[:, :t] -= previous[j]
            previous[j] = log
            if t:
                mass[j, :, :t] *= 1 - hazard
                mass[j, :, t] = hazard * priors[j]
            block = mass[j, :, :t + 1]
            # Work in logs for the large likelihood gap between candidate sources.
            with np.errstate(divide="ignore"):
                log_weights = np.log(block) + increments
            block[:] = softmax(log_weights.reshape(worlds, -1), axis=1).reshape(block.shape)
            preds[j] = (block[..., None, None] * rows).sum(axis=(1, 2))
        yield preds


def batched_bayes(recipes, env):
    worlds = len(env["archive"])
    shape = (len(recipes), ROUNDS, worlds, STATES, V)
    compact = np.empty(shape)
    for j in range(0, worlds, BY_WORLD):
        segment = slice(j, j + BY_WORLD)
        bank = DirichletBank(env["archive"][segment])
        process = reset_inference(bank, env["labels"][:, segment], recipes)
        for t, forecasts in enumerate(process):
            compact[:, t, segment] = forecasts
    return compact


def evaluate(env, trusted):
    model = normalize(env["archive"].mean(axis=1) + ALPHA)
    losses = np.empty((2, ROUNDS, 3, len(model)))
    for t, verified in enumerate(trusted):
        generated = transition_counts(model, SYNTHETIC, np.random.default_rng(env["generation_seeds"][t]))
        model = normalize(generated + PSEUDO * verified)
        truth = env["targets"][t]
        weights = env["weight"][t]
        tail = normalize(weights[:, ~HEAD])
        for i, q in enumerate((model, verified)):
            kl = (truth * np.log(truth / q)).sum(axis=-1)
            losses[i, t, 0] = (weights * kl).sum(axis=-1)
            losses[i, t, 1] = (tail * kl[:, ~HEAD]).sum(axis=-1)
            under = (truth * (q < truth / 2)).sum(axis=-1)
            losses[i, t, 2] = (tail * under[:, ~HEAD]).sum(axis=-1)
    focused = (losses[:, :, 1:] * env["focus"][None, :, None, :]).sum(axis=1) / 8
    result = np.stack((losses[:, :, 0].mean(axis=1), focused[:, 0], focused[:, 1],
                       losses[:, :, 1].mean(axis=1)), axis=1)
    return {name: result[i] for i, name in enumerate(("recursive", "verified_only"))}


def run_suite(seeds, n, settings):
    result = {}
    for regime, seed in zip(REGIMES, seeds, strict=True):
        env = make_worlds(regime, n, seed)
        runs = {}
        recipes = []
        for method, params in settings.items():
            for i, config in enumerate(params):
                if method.endswith("_reset"):
                    recipes.append((method, i, config))
                elif method.endswith("_retrieval"):
                    runs[method, i] = evaluate(env, recent_retrieval(method, config, env))
                else:
                    runs[method, i] = evaluate(env, simple_forecasts(method, config, env))
        batch = batched_bayes([(m, i) for m, _, i in recipes], env)
        for (method, i, _), trusted in zip(recipes, batch, strict=True):
            runs[method, i] = evaluate(env, iter(trusted))
        result[regime] = runs
    return result


def freeze():
    bank = run_suite(DEV_SEEDS, DEV_N, MODELS)
    grid = {}
    selected = {}
    for method, configs in MODELS.items():
        scored = []
        for i, config in enumerate(configs):
            overall = [results[method, i]["recursive"][0].mean() for results in bank.values()]
            tail = [bank[k][method, i]["recursive"][1].mean() for k in TARGET]
            scored.append({"score": float(.5 * np.mean(overall) + .5 * np.mean(tail)), "config": config})
        grid[method] = scored
        selected[method] = min(scored, key=lambda c: c["score"])
    baseline = min((m for m in MODELS if m != CANDIDATE), key=lambda m: selected[m]["score"])
    return {
        "source_sha256": source_fingerprint(),
        "dev_seeds": DEV_SEEDS,
        "test_seeds": TEST_SEEDS,
        "selected": selected,
        "scores": grid,
        "best_reference": baseline,
        "worlds_per_family": [DEV_N, TEST_N],
    }


def interval(differences, rng):
    samples = np.zeros(10000)
    point = 0.
    for diff in differences:
        paired = rng.integers(len(diff), size=(len(samples), len(diff)))
        samples += diff[paired].mean(axis=1) / len(differences)
        point += diff.mean() / len(differences)
    return [float(point), *np.quantile(samples, [.025, .975]).tolist()]


def confirm(frozen):
    assert frozen["source_sha256"] == source_fingerprint()
    assert tuple(frozen["test_seeds"]) == TEST_SEEDS
    rng = np.random.default_rng(227611)
    recipes = {method: [record["config"]] for method, record in frozen["selected"].items()}
    suite = run_suite(TEST_SEEDS, TEST_N, recipes)
    best = frozen["best_reference"]
    regimes = {}
    for regime, data in suite.items():
        regimes[regime] = {}
        for arm in ("recursive", "verified_only"):
            ledger = {family: data[family, 0][arm].tolist() for family in recipes}
            absolute = {
                family: dict(zip(METRICS, np.mean(values, axis=1).tolist()))
                for family, values in ledger.items()
            }
            uncertainty = {
                family: dict(zip(METRICS, [interval([row], rng) for row in np.asarray(values) - ledger[CANDIDATE]]))
                for family, values in ledger.items() if family != CANDIDATE
            }
            regimes[regime][arm] = {"worlds": ledger, "absolute": absolute, "reduction": uncertainty}
    def effect(regime, arm, metric):
        d = suite[regime]
        return d[best, 0][arm][metric] - d[CANDIDATE, 0][arm][metric]
    rare = interval([effect(r, "recursive", 1) for r in TARGET], rng)
    population = interval([-effect(r, "recursive", 0) for r in REGIMES], rng)
    verified = interval([effect(r, "verified_only", 1) for r in TARGET], rng)
    return {
        "freeze": frozen, "regimes": regimes, "metric_order": METRICS,
        "focal_rare_kl_reduction": rare,
        "macro_population_kl_degradation": population,
        "verified_only_focal_rare_kl_reduction": verified,
        "criteria_met": bool(rare[0] >= .010 and rare[1] > 0 and population[2] < .010),
        "budgets": {"rounds": ROUNDS, "verified": REAL, "generated": SYNTHETIC, "archive_each": ARCHIVE},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for mode in ("freeze", "confirm"):
        item = sub.add_parser(mode)
        item.add_argument("--output", required=True, type=Path)
        if mode == "confirm":
            item.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args()
    result = freeze() if args.stage == "freeze" else confirm(json.loads(args.freeze.read_text()))
    with args.output.open("x") as destination:
        json.dump(result, destination, sort_keys=True, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
