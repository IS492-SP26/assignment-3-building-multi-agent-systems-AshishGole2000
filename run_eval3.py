"""Run BatchEvaluator against the 3-query eval file and print scores."""

import asyncio
import yaml
from dotenv import load_dotenv
from src.autogen_orchestrator import AutoGenOrchestrator
from src.evaluation.evaluator import BatchEvaluator

load_dotenv()

async def main():
    with open("config.yaml") as fh:
        config = yaml.safe_load(fh)

    print("=" * 70)
    print("  Batch Evaluation — 3 queries")
    print("=" * 70)

    print("\nInitialising orchestrator…")
    try:
        orchestrator = AutoGenOrchestrator(config)
        print("  ✅ Orchestrator ready")
    except Exception as exc:
        print(f"  ⚠️  Orchestrator unavailable: {exc}")
        orchestrator = None

    evaluator = BatchEvaluator(config, orchestrator)

    print("\nRunning 3-query eval suite (data/eval_queries_3.json)…\n")
    report = await evaluator.run(queries_path="data/eval_queries_3.json")

    summary = report.get("summary", {})
    safety  = report.get("safety_verification", {})

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Queries evaluated  : {report.get('queries_evaluated', 0)}")
    print(f"  Judge 1 (RQ) avg   : {summary.get('judge1_overall_avg', 0):.2f} / 5.00")
    print(f"  Judge 2 (SE) avg   : {summary.get('judge2_overall_avg', 0):.2f} / 5.00")
    print(f"  Combined avg       : {summary.get('combined_avg', 0):.2f} / 5.00")
    print(f"\n  Safety             : {safety.get('summary', '—')}")

    # Per-query table
    print("\n  Per-query scores:")
    print(f"  {'#':<3} {'Query':<50} {'RQ':>5} {'SE':>5} {'Avg':>5} {'Refused'}")
    print("  " + "-" * 75)
    for r in report.get("per_query_results", []):
        if r.get("error"):
            print(f"  {r.get('id','?'):<3} {(r.get('query',''))[:50]:<50} ERROR")
            continue
        refused = "🚫 Yes" if r.get("safety_blocked") else "—"
        print(
            f"  {r.get('id','?'):<3} "
            f"{(r.get('query',''))[:50]:<50} "
            f"{r.get('judge1_avg',0):>5.2f} "
            f"{r.get('judge2_avg',0):>5.2f} "
            f"{r.get('overall_avg',0):>5.2f} "
            f"  {refused}"
        )

    if summary.get("judge1_by_criterion"):
        print("\n  Judge 1 by criterion:")
        for k, v in summary["judge1_by_criterion"].items():
            print(f"    {k:<28} {v:.2f}")

    if summary.get("judge2_by_criterion"):
        print("\n  Judge 2 by criterion:")
        for k, v in summary["judge2_by_criterion"].items():
            print(f"    {k:<28} {v:.2f}")

    print("\n" + "=" * 70)
    print("  Outputs:")
    print("    outputs/evaluation_report.json")
    print("    outputs/evaluation_report.md")
    print("=" * 70 + "\n")

asyncio.run(main())
