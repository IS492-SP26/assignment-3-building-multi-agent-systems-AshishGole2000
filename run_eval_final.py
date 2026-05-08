"""
Run queries 1, 4, 7 from eval_queries.json fresh, combine with the
already-scored 5 results (ids 2,3,5,6,8) saved in outputs/eval_results_5.json,
and write the final 8-query evaluation_report.{json,md}.
"""

import asyncio
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.autogen_orchestrator import AutoGenOrchestrator
from src.evaluation.evaluator import BatchEvaluator

load_dotenv()


async def main():
    with open("config.yaml") as fh:
        config = yaml.safe_load(fh)

    # ── Step 1: load the 5 already-scored results ────────────────────────────
    five_path = Path("outputs/eval_results_5.json")
    with open(five_path) as fh:
        five_results = json.load(fh)
    print(f"Loaded {len(five_results)} pre-scored results "
          f"(IDs {sorted(r['id'] for r in five_results)})")

    # ── Step 2: run queries 1, 4, 7 fresh ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Running queries 1, 4, 7 (data/eval_queries_147.json)")
    print("=" * 70)

    print("\nInitialising orchestrator…")
    try:
        orchestrator = AutoGenOrchestrator(config)
        print("  ✅ Orchestrator ready")
    except Exception as exc:
        print(f"  ⚠️  Orchestrator unavailable: {exc}")
        orchestrator = None

    evaluator = BatchEvaluator(config, orchestrator)

    print("\nRunning…\n")
    new_report  = await evaluator.run(queries_path="data/eval_queries_147.json")
    new_results = new_report.get("per_query_results", [])

    # ── Step 3: merge all 8 and sort by id ───────────────────────────────────
    all_results = five_results + new_results
    all_results.sort(key=lambda r: r.get("id", 999))

    # Sanity-check: expected ids 1-8
    ids = [r.get("id") for r in all_results]
    print(f"\nMerged result IDs: {ids}")

    # ── Step 4: regenerate final reports ─────────────────────────────────────
    safety_check = evaluator._verify_safety(all_results)
    final_report = evaluator._build_json_report(all_results, safety_check)
    evaluator._save_json_report(final_report)
    evaluator._save_md_report(all_results, safety_check)

    # ── console summary ───────────────────────────────────────────────────────
    summary = final_report.get("summary", {})

    print("\n" + "=" * 70)
    print("FINAL EVALUATION COMPLETE — ALL 8 QUERIES")
    print("=" * 70)
    print(f"  Queries evaluated  : {final_report.get('queries_evaluated', 0)}")
    print(f"  Judge 1 (RQ) avg   : {summary.get('judge1_overall_avg', 0):.2f} / 5.00")
    print(f"  Judge 2 (SE) avg   : {summary.get('judge2_overall_avg', 0):.2f} / 5.00")
    print(f"  Combined avg       : {summary.get('combined_avg', 0):.2f} / 5.00")
    print(f"\n  Safety             : {safety_check.get('summary', '—')}")

    print("\n  Per-query scores:")
    print(f"  {'#':<3} {'Category':<24} {'Query':<44} {'RQ':>5} {'SE':>5} {'Avg':>5}  Refused")
    print("  " + "-" * 94)
    for r in all_results:
        if r.get("error"):
            print(f"  {r.get('id','?'):<3} {'ERROR':<24} {(r.get('query',''))[:44]:<44}  ERROR")
            continue
        refused = "🚫" if r.get("safety_blocked") else "—"
        print(
            f"  {r.get('id','?'):<3} "
            f"{r.get('category','—'):<24} "
            f"{(r.get('query',''))[:44]:<44} "
            f"{r.get('judge1_avg',0):>5.2f} "
            f"{r.get('judge2_avg',0):>5.2f} "
            f"{r.get('overall_avg',0):>5.2f}  "
            f"{refused}"
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
    print("  Final outputs:")
    print("    outputs/evaluation_report.json  (all 8 queries)")
    print("    outputs/evaluation_report.md    (complete summary table)")
    print("=" * 70 + "\n")


asyncio.run(main())
