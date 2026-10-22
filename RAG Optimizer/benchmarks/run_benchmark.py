"""Benchmark CLI — orchestrates BEIR evaluation, prints comparison table, enforces CI gate.

Usage examples:
    python -m benchmarks.run_benchmark --dataset nfcorpus --max-queries 50
    python -m benchmarks.run_benchmark --dataset nfcorpus --strategies dense hybrid
    python -m benchmarks.run_benchmark --dataset nfcorpus --ci --baseline-file benchmarks/baselines/nfcorpus.json
    python -m benchmarks.run_benchmark --dataset nfcorpus --ci --baseline-file benchmarks/baselines/nfcorpus.json --quality-threshold 0.02 --latency-regression-pct 20
    python -m benchmarks.run_benchmark --dataset nfcorpus --save-baseline
    python -m benchmarks.run_benchmark --dataset nfcorpus --router-eval

Exit codes: 0 = pass, 1 = regression detected, 2 = infrastructure error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import anthropic

from benchmarks.beir_eval import BEIREvaluator, MetricsResult, check_regression
from benchmarks.router_eval import evaluate_routing_policies, render_markdown
from config.settings import get_settings
from retrieval.strategies.bm25 import BM25Retriever
from retrieval.strategies.dense import DenseRetriever
from retrieval.strategies.hybrid import HybridRetriever
from retrieval.strategies.hyde import HyDERetriever
from retrieval.strategies.multi_query import MultiQueryRetriever
from retrieval.strategies.parent_child import ParentChildRetriever
from router.classifier import QueryClassifier
from router.registry import StrategyRegistry
from router.router import AdaptiveRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_RESULTS_DIR = Path("benchmarks/results")
_BASELINES_DIR = Path("benchmarks/baselines")
_ALL_STRATEGIES = ["dense", "bm25", "hybrid", "hyde", "multi_query", "parent_child"]
_K_VALUES = [5, 10, 100]


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------

async def build_router() -> tuple[AdaptiveRouter, list[str]]:
    """Build and warm up all strategies. Returns (router, list_of_available_strategies)."""
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    child_dense = DenseRetriever(collection="rag_dense_children")

    candidates = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": HybridRetriever(dense=dense, bm25=bm25),
        "hyde": HyDERetriever(dense=dense),
        "multi_query": MultiQueryRetriever(dense=dense),
        "parent_child": ParentChildRetriever(dense=child_dense),
    }

    available: dict = {}
    for name, retriever in candidates.items():
        try:
            await retriever.warm_up()
            available[name] = retriever
            log.info("  ✓ %s", name)
        except Exception as exc:
            log.warning("  ✗ %s — warm_up failed: %s", name, exc)

    if not available:
        log.error("No strategies could be warmed up — check that Qdrant and ES are running.")
        sys.exit(2)

    registry = StrategyRegistry()  # no DB pool — fine for forced-strategy benchmarking
    router = AdaptiveRouter(strategies=available, registry=registry)
    return router, list(available.keys())


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _print_table(
    dataset: str,
    results: dict[str, MetricsResult],
    baseline: dict[str, MetricsResult] | None = None,
) -> None:
    best = max(results, key=lambda s: results[s].primary()) if results else None

    col_w = {"strategy": 14, "ndcg": 10, "mrr": 10, "recall": 11, "ms": 6}
    h_top = "╔" + "═" * col_w["strategy"] + "╦" + "═" * col_w["ndcg"] + "╦" + "═" * col_w["mrr"] + "╦" + "═" * col_w["recall"] + "╦" + "═" * col_w["ms"] + "╗"
    h_sep = "╠" + "═" * col_w["strategy"] + "╬" + "═" * col_w["ndcg"] + "╬" + "═" * col_w["mrr"] + "╬" + "═" * col_w["recall"] + "╬" + "═" * col_w["ms"] + "╣"
    h_bot = "╚" + "═" * col_w["strategy"] + "╩" + "═" * col_w["ndcg"] + "╩" + "═" * col_w["mrr"] + "╩" + "═" * col_w["recall"] + "╩" + "═" * col_w["ms"] + "╝"

    def _row(strategy: str, m: MetricsResult) -> str:
        marker = " ↑" if strategy == best else "  "
        ndcg = f"{m.primary():.3f}{marker}"
        mrr = f"{m.metrics.get('mrr', 0.0):.3f}"
        recall = f"{m.metrics.get('recall@100', m.metrics.get('recall@10', 0.0)):.3f}"
        ms = f"{m.avg_latency_ms:.0f}"
        return (
            f"║ {strategy:<{col_w['strategy']-2}} ║"
            f" {ndcg:>{col_w['ndcg']-2}} ║"
            f" {mrr:>{col_w['mrr']-2}} ║"
            f" {recall:>{col_w['recall']-2}} ║"
            f" {ms:>{col_w['ms']-1}}║"
        )

    print(f"\n╔{'═' * (sum(col_w.values()) + len(col_w) - 1)}╗")
    print(f"║  BEIR Evaluation — {dataset:<{sum(col_w.values()) + len(col_w) - 22}}║")
    print(h_top)
    print(f"║ {'Strategy':<{col_w['strategy']-2}} ║ {'nDCG@10':>{col_w['ndcg']-2}} ║ {'MRR':>{col_w['mrr']-2}} ║ {'R@100':>{col_w['recall']-2}} ║ {'AvgMs':>{col_w['ms']-1}}║")
    print(h_sep)

    sorted_results = sorted(results.items(), key=lambda x: x[1].primary(), reverse=True)
    for strategy, m in sorted_results:
        print(_row(strategy, m))

    print(h_bot)

    if best and baseline and best in baseline:
        delta = results[best].primary() - baseline[best].primary()
        sign = "+" if delta >= 0 else ""
        print(f"Best: {best} ({sign}{delta:.3f} vs baseline {best})")
    elif best:
        print(f"Best: {best} ({results[best].primary():.3f} nDCG@10)")


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def save_baseline(dataset: str, results: dict[str, MetricsResult]) -> Path:
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = _BASELINES_DIR / f"{dataset}.json"
    payload = {s: m.to_dict() for s, m in results.items()}
    path.write_text(json.dumps(payload, indent=2))
    log.info("Baseline saved → %s", path)
    return path


def load_baseline(path: str | Path) -> dict[str, MetricsResult]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {s: MetricsResult.from_dict(v) for s, v in data.items()}


def _strategy_comparison(
    current: MetricsResult,
    baseline: MetricsResult,
) -> dict[str, float | None]:
    latency_delta_ms = current.avg_latency_ms - baseline.avg_latency_ms
    latency_delta_pct = None
    if baseline.avg_latency_ms > 0:
        latency_delta_pct = ((current.avg_latency_ms / baseline.avg_latency_ms) - 1.0) * 100.0
    return {
        "ndcg_delta": current.primary() - baseline.primary(),
        "latency_delta_ms": latency_delta_ms,
        "latency_delta_pct": latency_delta_pct,
    }


def _fmt_signed(value: float, decimals: int = 3) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{decimals}f}"


def _build_markdown_summary(
    *,
    dataset: str,
    timestamp_iso: str,
    max_queries: int | None,
    results: dict[str, MetricsResult],
    baseline: dict[str, MetricsResult],
    ci_enabled: bool,
    gate_passed: bool | None,
    gate_message: str,
    quality_threshold: float,
    latency_regression_pct: float | None,
) -> str:
    best = max(results, key=lambda s: results[s].primary()) if results else ""
    lines = [
        f"# Benchmark Summary - {dataset}",
        "",
        f"- Timestamp: {timestamp_iso}",
        f"- Max queries: {max_queries if max_queries is not None else 'all'}",
        f"- Retrieval depth: top-{max(_K_VALUES)} per query, before reranking (latency is measured at this depth)",
        f"- Strategies evaluated: {', '.join(sorted(results.keys()))}",
        f"- Best strategy: {best or 'n/a'}",
        f"- Baseline compared: {'yes' if baseline else 'no'}",
        "",
        "## CI Gate",
        "",
        f"- Enabled: {'yes' if ci_enabled else 'no'}",
        f"- Passed: {gate_passed if ci_enabled else 'n/a'}",
        f"- Quality threshold (nDCG@10 drop): {quality_threshold}",
        f"- Latency regression threshold: {'disabled' if latency_regression_pct is None else f'+{latency_regression_pct:.1f}%'}",
        f"- Message: {gate_message}",
        "",
        "## Strategy Metrics",
        "",
        "| Strategy | nDCG@10 | Delta nDCG vs baseline | MRR | Recall@100 | Avg latency ms | P50 ms | P95 ms | Latency delta vs baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    sorted_results = sorted(results.items(), key=lambda x: x[1].primary(), reverse=True)
    for strategy, metric in sorted_results:
        ndcg = metric.primary()
        mrr = metric.metrics.get("mrr", 0.0)
        recall = metric.metrics.get("recall@100", metric.metrics.get("recall@10", 0.0))
        latency_ms = metric.avg_latency_ms

        ndcg_delta = "n/a"
        latency_delta = "n/a"
        if strategy in baseline:
            cmp = _strategy_comparison(metric, baseline[strategy])
            ndcg_delta = _fmt_signed(cmp["ndcg_delta"], decimals=3)
            if cmp["latency_delta_pct"] is None:
                latency_delta = _fmt_signed(cmp["latency_delta_ms"], decimals=1) + " ms"
            else:
                latency_delta = (
                    f"{_fmt_signed(cmp['latency_delta_ms'], decimals=1)} ms "
                    f"({_fmt_signed(cmp['latency_delta_pct'], decimals=1)}%)"
                )

        lines.append(
            "| "
            f"{strategy} | {ndcg:.3f} | {ndcg_delta} | {mrr:.3f} | {recall:.3f} | {latency_ms:.1f} | "
            f"{metric.p50_latency_ms:.1f} | {metric.p95_latency_ms:.1f} | {latency_delta} |"
        )

    lines.append("")
    return "\n".join(lines)


def save_report(
    dataset: str,
    results: dict[str, MetricsResult],
    baseline: dict[str, MetricsResult],
    *,
    max_queries: int | None,
    ci_enabled: bool,
    gate_passed: bool | None,
    gate_message: str,
    quality_threshold: float,
    latency_regression_pct: float | None,
) -> tuple[Path, Path]:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    timestamp_iso = now.isoformat()

    timestamped_json = _RESULTS_DIR / f"{dataset}_{ts}.json"
    latest_json = _RESULTS_DIR / f"{dataset}_latest.json"
    timestamped_md = _RESULTS_DIR / f"{dataset}_{ts}.md"
    latest_md = _RESULTS_DIR / f"{dataset}_latest.md"

    best = max(results, key=lambda s: results[s].primary()) if results else ""
    baseline_strategy = "bm25" if "bm25" in results else (best or "")
    improvement = (
        results[best].primary() - results[baseline_strategy].primary()
        if best and baseline_strategy and best != baseline_strategy
        else 0.0
    )

    baseline_comparison: dict[str, dict[str, float | None]] = {}
    for strategy, metric in results.items():
        if strategy in baseline:
            baseline_comparison[strategy] = _strategy_comparison(metric, baseline[strategy])

    gate = {
        "enabled": ci_enabled,
        "passed": gate_passed,
        "message": gate_message,
        "quality_threshold": quality_threshold,
        "latency_regression_pct": latency_regression_pct,
    }

    payload = {
        "dataset": dataset,
        "timestamp": timestamp_iso,
        "max_queries": max_queries,
        "strategies": {
            s: {
                **m.metrics,
                "avg_latency_ms": m.avg_latency_ms,
                "p50_latency_ms": m.p50_latency_ms,
                "p95_latency_ms": m.p95_latency_ms,
            }
            for s, m in results.items()
        },
        "baseline_comparison": baseline_comparison,
        "best_strategy": best,
        "baseline_strategy": baseline_strategy,
        "improvement_over_baseline": round(improvement, 4),
        "ci_gate": gate,
    }

    timestamped_json.write_text(json.dumps(payload, indent=2))
    latest_json.write_text(json.dumps(payload, indent=2))

    summary = _build_markdown_summary(
        dataset=dataset,
        timestamp_iso=timestamp_iso,
        max_queries=max_queries,
        results=results,
        baseline=baseline,
        ci_enabled=ci_enabled,
        gate_passed=gate_passed,
        gate_message=gate_message,
        quality_threshold=quality_threshold,
        latency_regression_pct=latency_regression_pct,
    )
    timestamped_md.write_text(summary)
    latest_md.write_text(summary)

    log.info("Report saved → %s", timestamped_json)
    log.info("Latest JSON → %s", latest_json)
    log.info("Summary saved → %s", timestamped_md)
    log.info("Latest summary → %s", latest_md)
    return timestamped_json, timestamped_md


async def _classify_queries(items: list[tuple[str, str]]) -> tuple[dict[str, str], float]:
    """Label queries with the production classifier, uncached. Returns (types, mean latency ms)."""
    client = anthropic.AsyncAnthropic(api_key=get_settings().anthropic_api_key)
    classifier = QueryClassifier(anthropic_client=client)
    semaphore = asyncio.Semaphore(4)
    latencies: list[float] = []

    async def _one(query_id: str, text: str) -> tuple[str, str]:
        async with semaphore:
            start = time.perf_counter()
            query_type = await classifier.classify(text)
            latencies.append((time.perf_counter() - start) * 1_000)
            return query_id, query_type.value

    types = dict(await asyncio.gather(*[_one(qid, text) for qid, text in items]))
    unknown = sum(1 for t in types.values() if t == "unknown") / max(len(types), 1)
    if unknown > 0.5:
        raise RuntimeError(
            f"classifier returned 'unknown' for {unknown:.0%} of queries; check the API key and network"
        )
    return types, (sum(latencies) / len(latencies) if latencies else 0.0)


def save_router_report(dataset: str, report, *, max_queries: int | None) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {"dataset": dataset, "max_queries": max_queries, **asdict(report)}
    markdown = render_markdown(report, dataset=dataset)
    for suffix in (stamp, "latest"):
        (_RESULTS_DIR / f"{dataset}_router_eval_{suffix}.json").write_text(json.dumps(payload, indent=2))
        (_RESULTS_DIR / f"{dataset}_router_eval_{suffix}.md").write_text(markdown)
    latest = _RESULTS_DIR / f"{dataset}_router_eval_latest.md"
    log.info("Router evaluation saved → %s", latest)
    print(markdown)
    return latest


async def _run_router_eval(args: argparse.Namespace, results: dict[str, MetricsResult]) -> bool:
    """Offline routing-policy evaluation over the per-query results of this run."""
    per_query = {s: m.per_query_ndcg10 for s, m in results.items() if m.per_query_ndcg10}
    if len(per_query) < 2:
        log.error("--router-eval needs at least two evaluated strategies")
        return False
    if not get_settings().anthropic_api_key:
        log.error("--router-eval needs ANTHROPIC_API_KEY to label query types")
        return False

    _, queries, _ = BEIREvaluator._load_dataset(args.dataset)
    items = list(queries.items())
    if args.max_queries:
        items = items[: args.max_queries]
    log.info("Labelling %d queries with the production classifier …", len(items))
    try:
        types, classifier_ms = await _classify_queries(items)
        report = evaluate_routing_policies(
            per_query,
            types,
            latency_ms={s: m.avg_latency_ms for s, m in results.items()},
            classifier_ms=classifier_ms,
        )
    except (RuntimeError, ValueError) as exc:
        log.error("Router evaluation failed: %s", exc)
        return False
    save_router_report(args.dataset, report, max_queries=args.max_queries)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    log.info("Building router …")
    router, available = await build_router()

    target_strategies = args.strategies or available
    invalid = [s for s in target_strategies if s not in available]
    if invalid:
        log.warning("Strategies not available (infra may be down): %s", invalid)
        target_strategies = [s for s in target_strategies if s in available]

    if not target_strategies:
        log.error("No strategies available.")
        return 2

    evaluator = BEIREvaluator(router=router)
    log.info("Evaluating %s on %s (max_queries=%s) …", target_strategies, args.dataset, args.max_queries)

    results = await evaluator.evaluate(
        dataset=args.dataset,
        strategies=target_strategies,
        k_values=_K_VALUES,
        max_queries=args.max_queries,
    )

    baseline = load_baseline(args.baseline_file) if args.baseline_file else {}
    baseline_exists = bool(args.baseline_file) and Path(args.baseline_file).exists()
    _print_table(args.dataset, results, baseline or None)

    latency_regression_pct: float | None = args.latency_regression_pct
    gate_passed: bool | None = None
    gate_message = "CI gate not enabled"

    if args.ci:
        if not args.baseline_file:
            gate_passed = False
            gate_message = "--ci requires --baseline-file so results can be compared to a saved baseline"
        elif not baseline_exists:
            gate_passed = False
            gate_message = f"Baseline file not found: {args.baseline_file}"
        elif not baseline:
            gate_passed = False
            gate_message = f"Baseline file is empty or invalid: {args.baseline_file}"
        else:
            gate_passed, gate_message = check_regression(
                results,
                baseline,
                threshold=args.quality_threshold,
                latency_regression_pct=latency_regression_pct,
            )

    save_report(
        args.dataset,
        results,
        baseline,
        max_queries=args.max_queries,
        ci_enabled=args.ci,
        gate_passed=gate_passed,
        gate_message=gate_message,
        quality_threshold=args.quality_threshold,
        latency_regression_pct=latency_regression_pct,
    )

    if args.save_baseline:
        save_baseline(args.dataset, results)

    if args.router_eval and not await _run_router_eval(args, results):
        return 2

    if args.ci:
        print(f"CI gate: {'PASSED' if gate_passed else 'FAILED'}")
        if gate_message:
            print(gate_message)
        if not gate_passed:
            return 1

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG benchmark runner with CI regression gate")
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument(
        "--strategies", nargs="+", default=None, choices=_ALL_STRATEGIES,
        help="Strategies to evaluate. Default: all available.",
    )
    p.add_argument("--ci", action="store_true", help="Enable CI regression gate (exit 1 on regression)")
    p.add_argument("--baseline-file", default=None, help="Path to baseline JSON for CI comparison")
    p.add_argument("--save-baseline", action="store_true", help="Write results as new baseline")
    p.add_argument(
        "--router-eval",
        action="store_true",
        help="Also evaluate routing policies offline (needs >=2 strategies and ANTHROPIC_API_KEY)",
    )
    p.add_argument(
        "--quality-threshold",
        type=float,
        default=0.02,
        help="Maximum allowed nDCG@10 drop per strategy before CI fails (default: 0.02)",
    )
    p.add_argument(
        "--latency-regression-pct",
        type=float,
        default=20.0,
        help="Maximum allowed avg latency increase (percent) vs baseline before CI fails (default: 20)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(_run(args)))
