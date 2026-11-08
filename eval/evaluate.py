import argparse
import os
import sqlite3
import time
from agents.baseline import run_b0, run_b1
from agents.llm import LLMError
from eval.score import fault_for, score_one
from run_baseline import load_tickets

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

def main():
    parser = argparse.ArgumentParser(description="Evaluate ERP ticket baselines.")
    parser.add_argument("--db",default="data/erp.db")
    parser.add_argument("--systems",nargs="+",choices=["b0", "b1"],default=["b0"])
    parser.add_argument( "--provider",default="mock")
    parser.add_argument("--model",default=None)
    parser.add_argument("--limit",type=int,default=None,)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(
            f"Database not found: {args.db}"
        )

    try:
        for system in args.systems:
            rows = evaluate_baseline(args.db,system,provider=args.provider,model=args.model,limit=args.limit)

            print(
                f"\nEvaluated {len(rows)} tickets "
                f"using {system.upper()}"
            )

            for row in rows:
                print(
                    f"{row['ticket_id']}: "
                    f"expected={row['expected_code']} "
                    f"predicted={row['predicted_code']} "
                    f"correct={row['code_ok']}"
                )

    except LLMError as error:
        raise SystemExit(
            f"Model provider failed: {error}"
        ) from error


if __name__ == "__main__":
    main()