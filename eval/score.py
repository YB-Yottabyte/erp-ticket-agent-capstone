import sqlite3
from erpsim.taxonomy import EXPECTED_ACTION
ENTITY_TABLE = {
    "invoice": "invoices",
    "salesorder": "sales_orders",
    "sales_order": "sales_orders",
    "order": "sales_orders",
    "customer": "customers",
    "payment": "payments",
    "jobrun": "job_runs",
    "job_run": "job_runs",
    "job": "job_runs",
    "delivery": "deliveries",
    "pricelist": "price_lists",
    "price_list": "price_lists",
    "item": "items",
    "salesorderline": "sales_order_lines",
    "stockledger": "stock_ledger",
}


def fault_for(con, ticket_id):
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM fault_log WHERE ticket_id = ?",
                      (ticket_id,)).fetchone()
    return dict(row) if row else None

def record_exists(con, entity, rec_id):
    table = ENTITY_TABLE.get(str(entity).lower().replace(" ", ""))
    if not table:
        return False
    try:
        row = con.execute(f"SELECT 1 FROM {table} WHERE id = ?", (rec_id,)).fetchone()
    except sqlite3.Error:
        return False
    return row is not None

def score_one(con, output, fault):
    code_ok = output.get("root_cause_code") == fault["fault_code"]
    ev = output.get("evidence") or []
    bad = [e for e in ev if not record_exists(con, e.get("entity"), e.get("id"))]
    evidence_valid = bool(ev) and not bad
    hallucinated = bool(bad)
    path = fault["causal_path"]
    evidence_relevant = any(str(e.get("id")) in path for e in ev)
    expected = EXPECTED_ACTION.get(fault["fault_code"])
    action_ok = (output.get("proposed_fix") or {}).get("action") == expected
    resolved = bool(code_ok and evidence_valid and evidence_relevant and action_ok)
    return {
        "ticket_id": fault["ticket_id"],
        "tier": fault["tier"],
        "expected_code": fault["fault_code"],
        "predicted_code": output.get("root_cause_code"),
        "code_ok": code_ok,
        "evidence_cited": len(ev),
        "evidence_valid": evidence_valid,
        "evidence_relevant": evidence_relevant,
        "hallucinated_evidence": hallucinated,
        "hallucinated_ids": [e.get("id") for e in bad],
        "expected_action": expected,
        "predicted_action": (output.get("proposed_fix") or {}).get("action"),
        "action_ok": action_ok,
        "confidence": output.get("confidence"),
        "escalated": output.get("escalate"),
        "malformed": bool(output.get("malformed")),
        "schema_problems": output.get("schema_problems") or [],
        "resolved": resolved,
    }


def summarise(rows):
    n = len(rows)
    if not n:
        return {}
    def pct(pred):
        return round(100.0 * sum(1 for r in rows if pred(r)) / n, 1)
    out = {
        "n": n,
        "root_cause_accuracy_pct": pct(lambda r: r["code_ok"]),
        "full_resolution_pct": pct(lambda r: r["resolved"]),
        "hallucinated_evidence_pct": pct(lambda r: r["hallucinated_evidence"]),
        "no_evidence_cited_pct": pct(lambda r: r["evidence_cited"] == 0),
        "malformed_pct": pct(lambda r: r["malformed"]),
    }
    for tier in (1, 2, 3):
        sub = [r for r in rows if r["tier"] == tier]
        if sub:
            ok = sum(1 for r in sub if r["code_ok"])
            out[f"tier{tier}_accuracy_pct"] = round(100.0 * ok / len(sub), 1)
            out[f"tier{tier}_n"] = len(sub)
    return out

def render(row):
    mark = lambda b: "PASS" if b else "FAIL"
    lines = [
        f"  expected code : {row['expected_code']}  (tier {row['tier']})",
        f"  predicted code: {row['predicted_code']}",
        f"  [{mark(row['code_ok'])}] root cause correct",
        f"  [{mark(row['evidence_valid'])}] evidence real "
        f"({row['evidence_cited']} cited"
        + (f", fabricated: {', '.join(map(str, row['hallucinated_ids']))}"
           if row["hallucinated_ids"] else "") + ")",
        f"  [{mark(row['evidence_relevant'])}] evidence on causal path",
        f"  [{mark(row['action_ok'])}] fix action correct "
        f"(expected {row['expected_action']}, got {row['predicted_action']})",
        f"  confidence    : {row['confidence']}   escalated: {row['escalated']}",
        "",
        f"  RESOLVED: {'YES' if row['resolved'] else 'NO'}",
    ]
    if row["schema_problems"]:
        lines.insert(0, f"  schema problems: {row['schema_problems']}")
    return "\n".join(lines)
