import argparse
import json
import os
import sqlite3
import sys
import time

from agents.baseline import run_b0, run_b1
from agents.llm import LLMError
from eval.score import fault_for, render, score_one, summarise

DB = "data/erp.db"
def parse_ticket_file(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    head, _, body = text.partition("\n\n")
    fields = {}
    for line in head.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip().upper()] = v.strip()
    return {
        "id": fields.get("TICKET", "TKT-LOCAL"),
        "subject": fields.get("SUBJECT", ""),
        "reported_by": fields.get("FROM", "unknown"),
        "created_at": fields.get("CREATED", ""),
        "referenced_doc": fields.get("REFERENCE") or None,
        "body": body.strip(),
    }


def load_tickets(db, ticket_id=None, limit=None):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    q = "SELECT * FROM tickets"
    args = ()
    if ticket_id:
        q += " WHERE id = ?"
        args = (ticket_id,)
    q += " ORDER BY id"
    if limit:
        q += f" LIMIT {int(limit)}"
    rows = [dict(r) for r in con.execute(q, args)]
    con.close()
    return rows


def main():
    ap = argparse.ArgumentParser(description="Run the ERP ticket diagnosis baseline.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="path to a ticket text file")
    src.add_argument("--ticket", help="ticket id from the database")
    src.add_argument("--all", action="store_true", help="every generated ticket")
    ap.add_argument("--baseline", choices=["b0", "b1"], default="b1")
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "gemini", "anthropic", "openai", "groq"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--score", action="store_true", help="grade against fault_log")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write results as JSONL")
    a = ap.parse_args()

    if not os.path.exists(a.db):
        sys.exit(f"no database at {a.db}\n"
                 f"run: python -m erpsim.build_world --seed 42 --out {a.db}")

    if a.input:
        tickets = [parse_ticket_file(a.input)]
    else:
        tickets = load_tickets(a.db, a.ticket, a.limit)
    if not tickets:
        sys.exit("no tickets matched")

    con = sqlite3.connect(a.db)
    rows, results = [], []
    label = "B0 keyword rules" if a.baseline == "b0" else f"B1 single call ({a.provider})"
    print(f"baseline: {label}")
    print(f"tickets : {len(tickets)}")
    print("=" * 68)

    for t in tickets:
        started = time.time()
        try:
            if a.baseline == "b0":
                out, usage = run_b0(t)
            else:
                out, usage = run_b1(t, provider=a.provider, model=a.model)
        except LLMError as e:
            sys.exit(f"provider error: {e}")
        elapsed = time.time() - started
        out["ticket_id"] = t["id"]
        results.append(out)

        print(f"\nTICKET {t['id']}  {t['subject']}")
        print("-" * 68)
        print(json.dumps({k: v for k, v in out.items()
                          if k not in ("schema_problems", "raw")}, indent=2))
        print(f"\n  latency {elapsed:.2f}s   tokens in/out "
              f"{usage.get('input_tokens')}/{usage.get('output_tokens')}")

        if a.score:
            fault = fault_for(con, t["id"])
            if not fault:
                print("  (no answer key for this ticket - not scored)")
            else:
                row = score_one(con, out, fault)
                rows.append(row)
                print()
                print(render(row))

    if a.score and len(rows) > 1:
        print("\n" + "=" * 68)
        print("SUMMARY")
        for k, v in summarise(rows).items():
            print(f"  {k:32} {v}")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            for out in results:
                fh.write(json.dumps(out) + "\n")
        print(f"\nwrote {a.out}")

    con.close()


if __name__ == "__main__":
    main()
