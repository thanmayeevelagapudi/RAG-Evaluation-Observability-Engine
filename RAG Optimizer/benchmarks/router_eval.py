"""Offline routing-policy evaluation on a labelled BEIR run.

Every strategy is scored on every query, so we know what each routing choice would have earned.
That lets us compare a type-conditioned router against static baselines and a per-query oracle
without bandit exploration, with paired-bootstrap confidence intervals.

Read the result as a ceiling, not a forecast: the reward here is nDCG@10 against human qrels
(the live bandit sees noisier RAGAS/judge scores), and benchmark retrieval runs before reranking.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean

import numpy as np

# HyDE and multi-query each make one LLM call per query; routing adds one for classification.
LLM_CALLS_PER_QUERY: Mapping[str, int] = {"hyde": 1, "multi_query": 1}
CLASSIFIER_LLM_CALLS = 1


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def paired_bootstrap(
    a: list[float], b: list[float], *, n_boot: int = 10_000, seed: int = 0
) -> BootstrapResult:
    """Bootstrap the mean of per-query differences a - b: 95% percentile CI, two-sided p."""
    if len(a) != len(b) or not a:
        raise ValueError("paired_bootstrap needs two equal-length, non-empty samples")
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 1_000):  # chunked to bound memory on large query sets
        size = min(1_000, n_boot - start)
        means[start : start + size] = diff[rng.integers(0, n, size=(size, n))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    p_value = 2.0 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return BootstrapResult(
        mean_diff=float(diff.mean()),
        ci_low=float(low),
        ci_high=float(high),
        p_value=min(1.0, max(p_value, 1.0 / n_boot)),
    )


def _common_queries(
    per_query: Mapping[str, Mapping[str, float]], query_types: Mapping[str, str]
) -> list[str]:
    if not per_query:
        raise ValueError("no strategy results to evaluate")
    return sorted(set(query_types).intersection(*(set(v) for v in per_query.values())))


def _best(strategies: list[str], per_query: Mapping[str, Mapping[str, float]], qids: list[str]) -> str:
    return max(strategies, key=lambda s: fmean(per_query[s][q] for q in qids))


def cross_validated_router(
    per_query: Mapping[str, Mapping[str, float]],
    query_types: Mapping[str, str],
    *,
    folds: int = 5,
    seed: int = 0,
    min_type_samples: int = 5,
) -> dict[str, str]:
    """Pick, per query type, the strategy with the best mean nDCG on the other folds.

    Returns {query_id: chosen_strategy}. Types with fewer than ``min_type_samples`` training
    queries fall back to the best strategy overall, mirroring the bandit's cold-start rule.
    """
    strategies = sorted(per_query)
    qids = _common_queries(per_query, query_types)
    if folds < 2 or len(qids) < folds:
        raise ValueError(f"need at least {max(folds, 2)} labelled queries for {folds}-fold evaluation")

    order = list(qids)
    random.Random(seed).shuffle(order)
    fold_of = {q: i % folds for i, q in enumerate(order)}

    choice: dict[str, str] = {}
    for fold in range(folds):
        train = [q for q in order if fold_of[q] != fold]
        by_type: dict[str, list[str]] = defaultdict(list)
        for q in train:
            by_type[query_types[q]].append(q)
        overall = _best(strategies, per_query, train)
        per_type = {
            t: _best(strategies, per_query, qs)
            for t, qs in by_type.items()
            if len(qs) >= min_type_samples
        }
        for q in order:
            if fold_of[q] == fold:
                choice[q] = per_type.get(query_types[q], overall)
    return choice


@dataclass
class PolicyStats:
    name: str
    mean_ndcg: float
    mean_latency_ms: float | None = None
    mean_llm_calls: float | None = None
    router_minus_policy: BootstrapResult | None = None


@dataclass
class RoutingReport:
    n_queries: int
    folds: int
    type_counts: dict[str, int]
    best_static: str
    router_choice_share: dict[str, float]
    policies: list[PolicyStats]
    exhaustive_llm_calls: int
    exhaustive_latency_parallel_ms: float | None
    exhaustive_latency_sequential_ms: float | None


def evaluate_routing_policies(
    per_query: Mapping[str, Mapping[str, float]],
    query_types: Mapping[str, str],
    *,
    latency_ms: Mapping[str, float] | None = None,
    classifier_ms: float | None = None,
    llm_calls: Mapping[str, int] = LLM_CALLS_PER_QUERY,
    folds: int = 5,
    seed: int = 0,
    n_boot: int = 10_000,
    min_type_samples: int = 5,
) -> RoutingReport:
    """Compare the cross-validated router with every static strategy and the per-query oracle.

    ``per_query`` is {strategy: {query_id: nDCG@10}}. The best static strategy is picked
    in-sample, which flatters the baseline and makes the router comparison conservative.
    """
    strategies = sorted(per_query)
    qids = _common_queries(per_query, query_types)
    choice = cross_validated_router(
        per_query, query_types, folds=folds, seed=seed, min_type_samples=min_type_samples
    )
    latency = dict(latency_ms or {})

    router_scores = [per_query[choice[q]][q] for q in qids]
    oracle_scores = [max(per_query[s][q] for s in strategies) for q in qids]

    static_scores: dict[str, list[float]] = {}
    statics: list[PolicyStats] = []
    for s in strategies:
        scores = [per_query[s][q] for q in qids]
        static_scores[f"always_{s}"] = scores
        statics.append(
            PolicyStats(f"always_{s}", fmean(scores), latency.get(s), float(llm_calls.get(s, 0)))
        )
    statics.sort(key=lambda p: p.mean_ndcg, reverse=True)

    router_latency = None
    if classifier_ms is not None and set(choice.values()).issubset(latency):
        router_latency = classifier_ms + fmean(latency[choice[q]] for q in qids)
    router = PolicyStats(
        "router_cv",
        fmean(router_scores),
        router_latency,
        CLASSIFIER_LLM_CALLS + fmean(llm_calls.get(choice[q], 0) for q in qids),
    )

    for p in statics:
        p.router_minus_policy = paired_bootstrap(
            router_scores, static_scores[p.name], n_boot=n_boot, seed=seed
        )
    oracle = PolicyStats(
        "oracle",
        fmean(oracle_scores),
        router_minus_policy=paired_bootstrap(router_scores, oracle_scores, n_boot=n_boot, seed=seed),
    )

    has_all_latency = all(s in latency for s in strategies)
    return RoutingReport(
        n_queries=len(qids),
        folds=folds,
        type_counts=dict(sorted(Counter(query_types[q] for q in qids).items())),
        best_static=statics[0].name,
        router_choice_share={
            s: sum(1 for q in qids if choice[q] == s) / len(qids) for s in sorted(set(choice.values()))
        },
        policies=[router, *statics, oracle],
        exhaustive_llm_calls=sum(llm_calls.get(s, 0) for s in strategies),
        exhaustive_latency_parallel_ms=max(latency[s] for s in strategies) if has_all_latency else None,
        exhaustive_latency_sequential_ms=sum(latency[s] for s in strategies) if has_all_latency else None,
    )


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def render_markdown(report: RoutingReport, *, dataset: str) -> str:
    shares = ", ".join(f"{s} {share:.0%}" for s, share in report.router_choice_share.items())
    types = ", ".join(f"{t}={n}" for t, n in report.type_counts.items())
    lines = [
        f"# Router policy evaluation - {dataset}",
        "",
        f"- Queries: {report.n_queries} ({report.folds}-fold cross-validation)",
        f"- Query types (production classifier): {types}",
        f"- Router choice share: {shares}",
        f"- Best static strategy (picked in-sample, which favours the baseline): {report.best_static}",
        "",
        "| Policy | nDCG@10 | Router minus policy (95% CI) | Bootstrap p | Retrieval ms | LLM calls/query |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for p in report.policies:
        b = p.router_minus_policy
        delta = "-" if b is None else f"{b.mean_diff:+.3f} [{b.ci_low:+.3f}, {b.ci_high:+.3f}]"
        pval = "-" if b is None else f"{b.p_value:.3f}"
        calls = "n/a" if p.mean_llm_calls is None else f"{p.mean_llm_calls:.2f}"
        lines.append(
            f"| {p.name} | {p.mean_ndcg:.3f} | {delta} | {pval} | {_fmt_ms(p.mean_latency_ms)} | {calls} |"
        )
    lines += [
        "",
        "Exhaustive reference (run every strategy, cost only): "
        f"{report.exhaustive_llm_calls} LLM calls/query, "
        f"{_fmt_ms(report.exhaustive_latency_parallel_ms)} ms if parallel, "
        f"{_fmt_ms(report.exhaustive_latency_sequential_ms)} ms if sequential.",
        "",
        "Caveats:",
        "- Ceiling, not forecast: reward is nDCG@10 against human qrels; the live bandit sees noisier RAGAS/judge scores.",
        "- The per-query oracle is optimistic: it takes a max over noisy per-query strategy scores, so router-to-oracle gap is diagnostic only, not directly achievable headroom.",
        "- Benchmark retrieval is pre-rerank and pre-compression; a reranker narrows strategy differences.",
        "- Router retrieval ms includes the measured, uncached classification latency.",
        "",
    ]
    return "\n".join(lines)
