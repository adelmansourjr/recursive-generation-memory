"""Online density tracking with matched verified and generated budgets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln

CONTEXTS = MOTIFS = 12
ROUNDS = 36
VERIFIED = 64
GENERATED = 384
TRUST = 24
ALPHA = 0.25
DEVELOPMENT = [280973, 280987, 281023, 281047, 281063]
CONFIRMATION = [538891, 538921, 538943, 538957, 538979]
REGIMES = ["stationary", "abrupt", "gradual", "known_return", "novel_return"]
BASELINES = ["latest", "cumulative", "rolling", "discounted", "archive_blend", "diffuse_reset"]
CANDIDATE = "archive_reset"
METRICS = ["population_kl", "rare_kl", "focal_rare_kl", "focal_underpredicted_mass", "focal_changed_kl"]
HERE = Path(__file__).resolve().parent
GRIDS = {
    "latest": [{}],
    "cumulative": [{}],
    "rolling": [{"window": w} for w in (2, 4, 8, 16)],
    "discounted": [{"discount": d} for d in (0.50, 0.75, 0.88, 0.94, 0.98)],
    "archive_blend": [{"window": w, "archive_weight": a} for w in (4, 8, 16) for a in (0.10, 0.30, 0.55)],
    "diffuse_reset": [{"hazard": h, "recall": 0.0} for h in (0.02, 0.06, 0.14)],
    "archive_reset": [{"hazard": h, "recall": a} for h in (0.02, 0.06, 0.14) for a in (0.20, 0.50, 0.80)],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_fingerprint():
    return {name: digest(HERE / name) for name in ("run.py", "Protocol.md")}


def normalize(x):
    return x / x.sum(axis=-1, keepdims=True)


def environment(regime, worlds, seed):
    rng = np.random.default_rng(seed)
    prior = rng.dirichlet([30] * 8 + [3] * 4, size=worlds)
    original, first, second = rng.dirichlet([0.60] * MOTIFS, size=(3, worlds, CONTEXTS))
    p = rng.uniform(0.60, 0.93, (worlds, CONTEXTS, 1))
    first = p * first + (1 - p) * original
    second = p * second + (1 - p) * original
    shifted = np.zeros((worlds, CONTEXTS), dtype=bool)
    by_world = np.arange(worlds)[:, None]
    order = rng.random((worlds, 8)).argsort(axis=1)
    shifted[by_world, order[:, :4]] = True
    order = rng.random((worlds, 4)).argsort(axis=1)
    shifted[by_world, 8 + order[:, :2]] = True
    onset = rng.integers(8, 13, worlds)
    midpoint = rng.integers(17, 21, worlds)
    returned = rng.integers(25, 29, worlds) if regime == "novel_return" else rng.integers(21, 26, worlds)
    length = rng.integers(6, 12, worlds)
    contamination = rng.uniform(0, 0.12, (worlds, CONTEXTS, 1))
    archive = rng.multinomial(rng.multinomial(2048, prior), original)
    context_prior = normalize(archive.sum(axis=-1) + 1)
    truths, evidence, generated_sizes = [], [], []
    for t in range(ROUNDS):
        weight = np.clip((t - onset + 1) / length, 0, 1) if regime == "gradual" else (t >= onset)
        weight = weight[:, None, None] * shifted[:, :, None]
        if regime == "stationary":
            truth = original
        else:
            truth = weight * first + (1 - weight) * original
        if regime == "known_return":
            active = ((t >= returned)[:, None] & shifted)[:, :, None]
            truth = np.where(active, (1 - contamination) * original + contamination * second, truth)
        elif regime == "novel_return":
            active = ((t >= midpoint)[:, None] & shifted)[:, :, None]
            truth = np.where(active, second, truth)
            active = ((t >= returned)[:, None] & shifted)[:, :, None]
            truth = np.where(active, (1 - contamination) * first + contamination * original, truth)
        quota = 5 + (rng.random((worlds, CONTEXTS)).argsort(axis=1) < 4)
        truths.append(truth)
        evidence.append(rng.multinomial(quota, truth))
        generated_sizes.append(rng.multinomial(GENERATED, context_prior))
    assert all(np.all(e.sum(axis=(1, 2)) == VERIFIED) for e in evidence)
    assert all(np.all(g.sum(axis=1) == GENERATED) for g in generated_sizes)
    focal_start = returned if regime in ("known_return", "novel_return") else onset
    focal = (np.arange(ROUNDS)[:, None] >= focal_start) & (np.arange(ROUNDS)[:, None] < focal_start + 8)
    return {
        "archive": archive, "truths": np.stack(truths), "evidence": np.stack(evidence),
        "sizes": generated_sizes, "generation_seeds": rng.integers(1 << 62, size=ROUNDS).tolist(),
        "prior": prior, "shifted": shifted, "focal": focal,
    }


def ordinary(method, param, archive, evidence):
    cumul = archive.copy()
    discount = archive.astype(float)
    for t, verified in enumerate(evidence):
        if method == "latest":
            yield normalize(verified + ALPHA)
        elif method == "cumulative":
            cumul += verified
            yield normalize(cumul + ALPHA)
        elif method == "discounted":
            discount = param["discount"] * discount + verified
            yield normalize(discount + ALPHA)
        else:
            counts = evidence[max(0, t - param["window"] + 1): t + 1].sum(axis=0)
            q = normalize(counts + ALPHA)
            if method == "archive_blend":
                w = param["archive_weight"]
                q = (1 - w) * q + w * normalize(archive + ALPHA)
            yield q


def bayes_forecasts(params, archive, evidence):
    """One marginal likelihood update per candidate last-reset time and prior."""
    prefix = np.concatenate((np.zeros_like(evidence[:1]), np.cumsum(evidence, axis=0)), axis=0)
    # Integer counts plus a fixed Dirichlet offset permit exact log-gamma lookup.
    max_total = int(archive.sum(axis=-1).max()) + 6 * ROUNDS
    lut = gammaln(np.arange(max_total + 1) + ALPHA)
    lut_total = gammaln(np.arange(max_total + 1) + MOTIFS * ALPHA)
    prior_known = np.moveaxis(archive[None, ...], 0, 2)
    shape = (len(params), *archive.shape[:2], ROUNDS)
    diffuse_mass = np.zeros(shape)
    known_mass = np.zeros(shape)
    known_mass[..., 0] = 1
    hazard = np.array([p["hazard"] for p in params])[:, None, None]
    recall = np.array([p["recall"] for p in params])[:, None, None]

    for t in range(ROUNDS):
        before = np.moveaxis(prefix[t] - prefix[: t + 1], 0, 2)
        after = before + evidence[t, :, :, None, :]
        known_before = before + prior_known
        known_after = after + prior_known

        def likelihood(prior_counts, updated_counts):
            return (
                (lut[updated_counts] - lut[prior_counts]).sum(axis=-1)
                - lut_total[updated_counts.sum(axis=-1)]
                + lut_total[prior_counts.sum(axis=-1)]
            )

        log_d = likelihood(before, after)
        log_a = likelihood(known_before, known_after)
        shift = np.maximum(log_d.max(axis=-1, keepdims=True), log_a.max(axis=-1, keepdims=True))
        if t:
            diffuse_mass[..., :t] *= (1 - hazard)[..., None]
            known_mass[..., :t] *= (1 - hazard)[..., None]
            diffuse_mass[..., t] = hazard * (1 - recall)
            known_mass[..., t] = hazard * recall
        d_mass = diffuse_mass[..., : t + 1] * np.exp(log_d - shift)
        a_mass = known_mass[..., : t + 1] * np.exp(log_a - shift)
        norm = (d_mass + a_mass).sum(axis=-1, keepdims=True)
        assert np.all(norm > 0)
        diffuse_mass[..., : t + 1] = d_mass / norm
        known_mass[..., : t + 1] = a_mass / norm
        yield (
            np.einsum("hwtl,wtlk->hwtk", diffuse_mass[..., : t + 1], normalize(after + ALPHA))
            + np.einsum("hwtl,wtlk->hwtk", known_mass[..., : t + 1], normalize(known_after + ALPHA))
        )


def score(forecast, truth, prior, shifted):
    error = (truth * np.log(truth / forecast)).sum(axis=-1)
    suppressed = (truth * (forecast < truth / 2)).sum(axis=-1)
    focus_rare = shifted[:, -4:]
    return np.array([
        (error * prior).sum(axis=-1),
        error[:, -4:].mean(axis=-1),
        (error[:, -4:] * focus_rare).sum(axis=-1) / 2,
        (suppressed[:, -4:] * focus_rare).sum(axis=-1) / 2,
        (error * shifted).sum(axis=-1) / 6,
    ])


def run_policy(forecasts, env, conserve=True):
    history = {"recursive": [], "verified_only": []}
    model = normalize(env["archive"] + ALPHA)
    for t, trusted in enumerate(forecasts):
        rng = np.random.default_rng(env["generation_seeds"][t])
        generated = rng.multinomial(env["sizes"][t], model)
        model = normalize(generated + TRUST * trusted) if conserve else normalize(generated + ALPHA)
        for name, prediction in (("recursive", model), ("verified_only", trusted)):
            history[name].append(score(prediction, env["truths"][t], env["prior"], env["shifted"]))
    result = {}
    for name, values in history.items():
        array = np.stack(values)
        focal_means = (array[:, 2:, :] * env["focal"][:, None, :]).sum(axis=0) / 8
        result[name] = np.concatenate((array[:, :2, :].mean(axis=0), focal_means))
    return result


def run_candidates(seeds, worlds, selected):
    result = {}
    for regime, seed in zip(REGIMES, seeds, strict=True):
        env = environment(regime, worlds, seed)
        scores = {}
        bayes_grid = []
        for method, params in selected.items():
            for param in params:
                key = (method, tuple(param.items()))
                if method.endswith("_reset"):
                    bayes_grid.append((key, param))
                else:
                    scores[key] = run_policy(ordinary(method, param, env["archive"], env["evidence"]), env)
        if bayes_grid:
            # Materialize compact W×C×K trusted forecasts for all settings, not run-length history.
            bank = np.stack(list(bayes_forecasts([p for _, p in bayes_grid], env["archive"], env["evidence"])), axis=1)
            for (key, _), values in zip(bayes_grid, bank, strict=True):
                scores[key] = run_policy(values, env)
        result[regime] = scores
        if seeds == CONFIRMATION:
            q = normalize(env["archive"] + ALPHA)
            control = run_policy((q for _ in range(ROUNDS)), env, conserve=False)
            scores["generated_only", ()] = {"recursive": control["recursive"]}
    return result


def development():
    regimes = run_candidates(DEVELOPMENT, 64, GRIDS)
    scores = {}
    selected = {}
    for method, configs in GRIDS.items():
        choices = []
        for config in configs:
            key = (method, tuple(config.items()))
            averages = np.stack([r[key]["recursive"][:3].mean(axis=-1) for r in regimes.values()])
            objective = float(averages[:, [0, 2]].mean())
            choices.append({"config": config, "score": objective})
        scores[method] = choices
        selected[method] = min(choices, key=lambda item: item["score"])
    competitor = min(BASELINES, key=lambda method: selected[method]["score"])
    return {
        "development_seeds": DEVELOPMENT, "confirmation_seeds": CONFIRMATION,
        "development_worlds_per_regime": 64, "test_worlds_per_regime": 192,
        "competitor": competitor, "selected": selected, "development_scores": scores,
        "source_sha256": protocol_fingerprint(),
        "parameters": {"rounds": ROUNDS, "verified": VERIFIED, "generated": GENERATED, "trust": TRUST},
    }


def interval(values, rng):
    count = len(values)
    draws = rng.integers(count, size=(5000, count))
    low, high = np.quantile(values[draws].mean(axis=1), (0.025, 0.975))
    return [float(values.mean()), float(low), float(high)]


def confirmation(frozen):
    if frozen["source_sha256"] != protocol_fingerprint():
        raise ValueError("Source differs from the freeze record")
    if frozen["confirmation_seeds"] != CONFIRMATION or frozen["test_worlds_per_regime"] != 192:
        raise ValueError("Confirmation plan differs from the fixed plan")
    selected = {method: [frozen["selected"][method]["config"]] for method in GRIDS}
    regimes = run_candidates(CONFIRMATION, 192, selected)
    rng = np.random.default_rng(421933)
    serializable = {}
    guardrail = []
    chosen = frozen["competitor"]
    for regime, rows in regimes.items():
        arms = {method: result for (method, _), result in rows.items()}
        serializable[regime] = {}
        for branch in ("recursive", "verified_only"):
            for metric, metric_id in (("focal_rare_kl", 2), ("population_kl", 0)):
                estimates = {}
                comparisons = {}
                worlds = {}
                candidate = arms[CANDIDATE][branch][metric_id]
                for method, result in arms.items():
                    if branch in result:
                        data = result[branch][metric_id]
                        estimates[method] = float(data.mean())
                        worlds[method] = data.tolist()
                        if method != CANDIDATE:
                            comparisons[method] = interval(data - candidate, rng)
                serializable[regime].setdefault(branch, {"absolute": {}, "paired_reduction": {}, "world_values": {}})
                serializable[regime][branch]["absolute"][metric] = estimates
                serializable[regime][branch]["paired_reduction"][metric] = comparisons
                serializable[regime][branch]["world_values"][metric] = worlds
            other_metrics = [1, 3, 4]
            serializable[regime][branch]["secondary"] = {
                method: {METRICS[i]: float(result[branch][i].mean()) for i in other_metrics}
                for method, result in arms.items() if branch in result
            }
            if branch == "recursive":
                guardrail.append(arms[CANDIDATE][branch][0] - arms[chosen][branch][0])
    resamples = [
        values[rng.integers(len(values), size=(5000, len(values)))].mean(axis=1) for values in guardrail
    ]
    low, high = np.quantile(np.mean(resamples, axis=0), (0.025, 0.975))
    return {
        "freeze": frozen, "metrics": METRICS, "regimes": serializable,
        "primary_known_return_focal_rare_kl_reduction": serializable["known_return"]["recursive"]["paired_reduction"]["focal_rare_kl"][chosen],
        "macro_population_kl_degradation_guardrail": [float(np.mean(guardrail)), float(low), float(high)],
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    dev = sub.add_parser("develop")
    dev.add_argument("--output", type=Path, required=True)
    test = sub.add_parser("confirm")
    test.add_argument("--freeze", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = development() if args.stage == "develop" else confirmation(json.loads(args.freeze.read_text()))
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
        destination.write("\n")


if __name__ == "__main__":
    main()
