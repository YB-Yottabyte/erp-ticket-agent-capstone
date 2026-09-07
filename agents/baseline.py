import json
import re

from agents.llm import complete
from erpsim.taxonomy import ACTIONS, codes, taxonomy_prompt_block

PROMPT = """You are an ERP support analyst diagnosing a stuck order-to-cash transaction.

Choose exactly one root cause from this closed list:
{taxonomy}

Allowed fix actions: {actions}

Reply with ONLY a JSON object, no prose and no code fences:
{{"root_cause_code": "<one code from the list>",
  "root_cause_explanation": "<one or two sentences>",
  "evidence": [{{"entity": "<Invoice|SalesOrder|Customer|Payment|JobRun|Delivery|PriceList>",
                 "id": "<record id>", "field": "<field>", "observed": "<value>"}}],
  "proposed_fix": {{"action": "<one action>", "target": "<what to act on>"}},
  "confidence": <0.0-1.0>,
  "escalate": <true|false>}}

Cite a record in `evidence` only if you are certain it exists. If you cannot
verify anything, return an empty evidence list and set escalate to true.

TICKET {ticket_id}
Subject: {subject}
From: {reported_by}
Created: {created_at}

{body}
"""

RULES = [
    (r"invoice", "INCOMPLETE_DELIVERY_BLOCKS_INVOICE"),
    (r"order", "CUSTOMER_ON_HOLD"),
]


def build_prompt(ticket):
    return PROMPT.format(
        taxonomy=taxonomy_prompt_block(),
        actions=", ".join(ACTIONS),
        ticket_id=ticket["id"], subject=ticket["subject"],
        reported_by=ticket["reported_by"], created_at=ticket["created_at"],
        body=ticket["body"],
    )


def run_b0(ticket):
    text = f"{ticket['subject']} {ticket['body']}".lower()
    code = "UNKNOWN"
    for pattern, candidate in RULES:
        if re.search(pattern, text):
            code = candidate
            break
    return {
        "ticket_id": ticket["id"],
        "root_cause_code": code,
        "root_cause_explanation": "Keyword rule match on ticket text.",
        "evidence": [],
        "proposed_fix": {"action": "ESCALATE", "target": "tier-2 support"},
        "confidence": 0.25,
        "escalate": True,
    }, {"input_tokens": 0, "output_tokens": 0}


def run_b1(ticket, provider="mock", model=None):
    raw, usage = complete(build_prompt(ticket), provider=provider, model=model)
    out = parse(raw)
    out["ticket_id"] = ticket["id"]
    return out, usage
    
FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

def parse(raw):
    text = FENCE.sub("", raw).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return _malformed(raw, "no JSON object found")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return _malformed(raw, f"invalid JSON: {e}")
    return validate(obj, raw)


def validate(obj, raw=""):
    problems = []
    if not isinstance(obj, dict):
        return _malformed(raw, "top level is not an object")

    code = obj.get("root_cause_code")
    if code not in codes():
        problems.append(f"root_cause_code {code!r} not in taxonomy")
        code = "UNKNOWN"

    ev = obj.get("evidence") or []
    clean_ev = []
    if not isinstance(ev, list):
        problems.append("evidence is not a list")
    else:
        for e in ev:
            if isinstance(e, dict) and "entity" in e and "id" in e:
                clean_ev.append({"entity": str(e.get("entity")), "id": str(e.get("id")),
                                 "field": str(e.get("field", "")),
                                 "observed": str(e.get("observed", ""))})
            else:
                problems.append(f"malformed evidence item: {e!r}")

    fix = obj.get("proposed_fix") or {}
    if not isinstance(fix, dict):
        fix, action = {}, None
    action = fix.get("action")
    if action not in ACTIONS:
        problems.append(f"action {action!r} not in allowed set")
        action = "ESCALATE"

    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        problems.append("confidence not a number")
        conf = 0.0

    return {
        "root_cause_code": code,
        "root_cause_explanation": str(obj.get("root_cause_explanation", ""))[:600],
        "evidence": clean_ev,
        "proposed_fix": {"action": action, "target": str(fix.get("target", ""))},
        "confidence": max(0.0, min(1.0, conf)),
        "escalate": bool(obj.get("escalate", False)),
        "schema_problems": problems,
    }


def _malformed(raw, why):
    return {
        "root_cause_code": "UNKNOWN",
        "root_cause_explanation": "",
        "evidence": [],
        "proposed_fix": {"action": "ESCALATE", "target": ""},
        "confidence": 0.0,
        "escalate": True,
        "schema_problems": [why],
        "malformed": True,
        "raw": raw[:600],
    }
