import argparse
import os
import sqlite3
import time
import json
from pathlib import Path
from agents.baseline import run_b0, run_b1
from agents.llm import LLMError
from eval.score import fault_for, score_one
from run_baseline import load_tickets
from run_agent_loop import build_agent, load_tickets as load_agent_tickets

def _token_total(usage):
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if not isinstance(input_tokens, int):
        return None
    if not isinstance(output_tokens, int):
        return None
    return input_tokens + output_tokens

def _mean(values):
    values = [
        value
        for value in values
        if isinstance(value, (int, float))
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 3)

def evaluate_baseline(db_path,baseline,provider="mock",model=None,limit=None):
    tickets = load_tickets(db_path, limit=limit)
    connection = sqlite3.connect(db_path)
    rows = []
    try:
        for ticket in tickets:
            start = time.perf_counter()
            if baseline == "b0":
                diagnosis, usage = run_b0(ticket)

            elif baseline == "b1":
                diagnosis, usage = run_b1(
                    ticket,
                    provider=provider,
                    model=model,
                )
            else:
                raise ValueError(
                    "baseline must be 'b0' or 'b1'"
                )
            latency = time.perf_counter() - start
            fault = fault_for(connection,ticket["id"])
            if fault is None:
                continue
            result = score_one(connection,diagnosis,fault)
            result.update(
                system=baseline,
                latency_seconds=round(latency, 6),
                input_tokens=usage.get(
                    "input_tokens"
                ),
                output_tokens=usage.get(
                    "output_tokens"
                ),
                total_tokens=_token_total(usage),
                tool_calls=0,
                dispatcher_calls=0,
                verifier_status=None,
                termination_reason="baseline_complete",
            )
            rows.append(result)
    finally:
        connection.close()
    return rows

def evaluate_agent(
    db_path,
    provider="groq",
    model=None,
    max_tool_calls=8,
    limit=None,
):
    tickets = load_agent_tickets(
        db_path,
        limit=limit,
    )
    connection = sqlite3.connect(db_path)
    rows =[]
    agent = build_agent(
        db_path,
        model,
        max_tool_calls,
        provider=provider,
    )
    try:
        for ticket in tickets:
            result = agent.run_loop(ticket)

            fault = fault_for(
                connection,
                ticket["id"],
            )
            if fault is None:
                continue
            scored = score_one(
                connection,
                result,
                fault,
            )
            usage = result.get(
                "model_usage"
            ) or {}
            verification = result.get(
                "verification"
            ) or {}
            tool_trace = result.get(
                "tool_trace"
            ) or []
            scored["resolved"] = bool(
                scored["resolved"] and verification.get("status") == "passed"
                and not result.get("schema_problems") and not result.get("escalate")
            )
            dispatcher_calls = sum(
                1
                for item in tool_trace
                if (
                    item.get("tool_name") is not None
                    and not item.get("cache_hit", False)
                )
            )
            scored.update(
                system="agent",
                latency_seconds=round(
                    result.get("elapsed_ms", 0)
                    / 1000,
                    6,
                ),
                input_tokens=usage.get(
                    "input_tokens"
                ),
                output_tokens=usage.get(
                    "output_tokens"
                ),
                total_tokens=_token_total(
                    usage
                ),
                tool_calls=result.get(
                    "tool_attempts"
                ),
                dispatcher_calls=dispatcher_calls,
                verifier_status=verification.get(
                    "status"
                ),
                termination_reason=result.get(
                    "termination_reason"
                ),
            )
            rows.append(scored)
    finally:
        connection.close()
    return rows

def aggregate(rows):
    if not rows:
        return {}
    number_of_cases = len(rows)
    def tier_accuracy(tier):
        tier_rows = [
            row
            for row in rows
            if row.get("tier") == tier
        ]
        if not tier_rows:
            return None, 0
        correct = sum(
            1
            for row in tier_rows
            if row["code_ok"]
        )
        accuracy = round(
            (correct / len(tier_rows)) * 100,
            1,
        )
        return accuracy, len(tier_rows)
    def percentage(condition):
        count = sum(
            1
            for row in rows
            if condition(row)
        )
        return round(
            (count / number_of_cases) * 100,
            1,
        )
    tier1_accuracy, tier1_n = tier_accuracy(1)
    tier2_accuracy, tier2_n = tier_accuracy(2)
    tier3_accuracy, tier3_n = tier_accuracy(3)
    summary = {
        "total_cases": number_of_cases,
        "root_cause_accuracy_pct": percentage(
            lambda row: row["code_ok"]
        ),
        "tier1_accuracy_pct": tier1_accuracy,
        "tier1_n": tier1_n,
        "tier2_accuracy_pct": tier2_accuracy,
        "tier2_n": tier2_n,
        "tier3_accuracy_pct": tier3_accuracy,
        "tier3_n": tier3_n,
        "json_complete_pct": percentage(
            lambda row: not row["malformed"] and not row.get("schema_problems")
        ),
        "malformed_pct": percentage(
            lambda row: row["malformed"]
        ),
        "full_resolution_pct": percentage(
            lambda row: row["resolved"]
        ),
        "action_accuracy_pct": percentage(
            lambda row: row["action_ok"]
        ),
        "evidence_valid_pct": percentage(
            lambda row: row["evidence_valid"]
        ),
        "evidence_relevant_pct": percentage(
            lambda row: row["evidence_relevant"]
        ),
        "hallucinated_evidence_pct": percentage(
            lambda row: row["hallucinated_evidence"]
        ),
        "escalation_rate_pct": percentage(
            lambda row: bool(row["escalated"])
        ),
        "avg_latency_seconds": _mean(
            [
                row.get("latency_seconds")
                for row in rows
            ]
        ),
        "avg_total_tokens": _mean(
            [
                row.get("total_tokens")
                for row in rows
            ]
        ),
        "avg_tool_calls": _mean(
            [
                row.get("tool_calls")
                for row in rows
            ]
        ),

        "avg_dispatcher_calls": _mean(
            [
                row.get("dispatcher_calls")
                for row in rows
            ]
        ),
    }
    verifier_rows = [
        row
        for row in rows
        if row.get("verifier_status") is not None
    ]
    if verifier_rows:
        passed = sum(
            1
            for row in verifier_rows
            if row["verifier_status"] == "passed"
        )

        summary["verifier_pass_pct"] = round(
            passed / len(verifier_rows) * 100,
            1,
        )
    return summary

def print_summary(system, summary):
    print()
    print(f"{system.upper()} Evaluation Summary")
    print("-" * 50)

    for metric, value in summary.items():
        print(f"{metric:32} {value}")
        
def main():
    parser = argparse.ArgumentParser(description="Evaluate ERP ticket systems.")
    parser.add_argument("--db",default="data/erp.db")
    parser.add_argument("--systems",nargs="+",choices=["b0", "b1", "agent"],default=["b0"])
    parser.add_argument( "--provider",default="mock")
    parser.add_argument("--model",default=None)
    parser.add_argument("--limit",type=int,default=None,)
    parser.add_argument("--out",default=None,
    help="Optional JSON file for saving evaluation results.",
    )
    parser.add_argument("--max-tool-calls",type=int,default=8)
    parser.add_argument(
        "--agent-provider",
        choices=["groq", "voyager"],
        default="groq",
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(
            f"Database not found: {args.db}"
        )
    all_results = {}
    try:
        for system in args.systems:
            if system in {"b0", "b1"}:
                rows = evaluate_baseline(
                    args.db,
                    system,
                    provider=args.provider,
                    model=args.model,
                    limit=args.limit,
                )
            else:
                rows = evaluate_agent(
                args.db,
                provider=args.agent_provider,
                model=args.model,
                max_tool_calls=args.max_tool_calls,
                limit=args.limit,
            )
            print(
                f"\nEvaluated {len(rows)} tickets "
                f"using {system.upper()}"
            )

            for row in rows:
                print(
                    f"{row['ticket_id']}: "
                    f"expected={row['expected_code']} "
                    f"predicted={row['predicted_code']} "
                    f"correct={row['code_ok']} "
                    f"termination={row.get('termination_reason')}"
                )
            summary = aggregate(rows)
            print_summary(system,summary)
            all_results[system] = {"summary": summary,"rows": rows}

    except LLMError as error:
        raise SystemExit(
            f"Model provider failed: {error}"
        ) from error
    if args.out:
        output_path = Path(args.out)

        output_path.parent.mkdir(parents=True,exist_ok=True)
        output_path.write_text(
            json.dumps(
                all_results,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        print(
            f"\nResults written to {output_path}"
        )


if __name__ == "__main__":
    main()
