import json
import os
import re
import shutil
import sys
import textwrap


class AgentTerminal:
    CAUSES = {
        "CUSTOMER_ON_HOLD": "Customer account is on hold",
        "MISSING_TAX_ID": "Customer tax ID is missing",
        "STALE_PRICE_LIST": "Order uses an expired price list",
        "INCOMPLETE_DELIVERY_BLOCKS_INVOICE": "Delivery is incomplete",
        "CREDIT_LIMIT_STALE_SYNC": "Credit usage was not updated after payments",
        "PRICE_SYNC_JOB_FAILED": "Price update job failed",
        "UNKNOWN": "Not enough verified evidence",
    }
    TOOLS = {
        "find_document": "Find related documents",
        "get_invoice": "Read invoice",
        "get_sales_order": "Read sales order",
        "get_customer": "Check customer",
        "get_delivery": "Check delivery",
        "get_payments": "Check payments",
        "get_price_list": "Check price list",
        "get_job_runs": "Check background jobs",
    }
    ACTIONS = {
        "RELEASE_HOLD": "Review and release the customer hold",
        "UPDATE_FIELD": "Update the missing customer field",
        "REPRICE_ORDER": "Reprice the order",
        "COMPLETE_DELIVERY": "Complete the delivery",
        "RERUN_JOB": "Review and rerun the background job",
        "ESCALATE": "Ask a support analyst to investigate",
    }
    STOPS = {
        "final_model_response": "Model finished its investigation",
        "tool_budget_exhausted": "Lookup limit reached; human review required",
        "provider_error": "Model request failed; retry later",
        "invalid_tool_protocol": "Model did not send a valid API tool request",
        "repeated_unproductive_tool_call": "Repeated lookup did not make progress",
        "ambiguous_or_missing_ticket_reference": "Ticket needs one clear invoice or order reference",
    }

    # Use terminal colors only when supported, without extra package dependencies.
    def __init__(self, stream=None, color=True):
        self.stream = stream or sys.stdout
        self.color = color and self.stream.isatty() and "NO_COLOR" not in os.environ
        self.width = max(40, min(100, shutil.get_terminal_size((100, 24)).columns))

    # Remove terminal controls from model and ticket text before displaying it.
    @staticmethod
    def clean(value):
        text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(value))
        return " ".join(re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text).split())

    # Wrap output so summaries remain readable in narrow or redirected terminals.
    def line(self, text="", color=None):
        parts = textwrap.wrap(self.clean(text), self.width) or [""]
        for part in parts:
            if color and self.color:
                part = f"\033[{color}m{part}\033[0m"
            print(part, file=self.stream, flush=True)

    # Identify the actual provider, model, ticket, and lookup budget.
    def header(self, ticket, provider, model, budget):
        self.line()
        self.line("=" * self.width, "90")
        self.line(f"ERP Ticket Agent | {provider.title()} | {model}", "33")
        self.line(f"Ticket: {ticket['id']} | {ticket.get('subject', '')}")
        self.line(f"Read-only investigation | Up to {budget} lookups")
        self.line("-" * self.width, "90")
        self.line("Records checked", "36")

    # Show each real tool result without dumping its complete database observation.
    def tool_result(self, record, budget):
        name = self.TOOLS.get(
            record.get("tool_name"), record.get("tool_name", "Lookup")
        )
        status = record.get("observation", {}).get("status", "unknown")
        args = record.get("arguments")
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except ValueError:
            args = None
        reference = (
            next(
                (value for value in (args or {}).values() if isinstance(value, str)), ""
            )
            if isinstance(args, dict)
            else ""
        )
        cached = " (reused result)" if record.get("cache_hit") else ""
        self.line(
            f"  [{record['attempt']}/{budget}] {name} {reference}: {status}{cached}",
            "32" if status == "ok" else "33",
        )

    # Separate diagnosis, verification, recommendations, and evaluation in the summary.
    def result(self, result, score=None):
        verification = result.get("verification") or {}
        passed = verification.get("status") == "passed" and not result.get(
            "schema_problems"
        )
        needs_review = result.get("escalate", True) or not passed
        self.line()
        self.line(
            "Result: "
            + ("Human review required" if needs_review else "Verified diagnosis"),
            "33" if needs_review else "32",
        )
        self.line("-" * self.width, "90")
        code = result.get("root_cause_code", "UNKNOWN")
        self.line(
            f"{'Possible cause' if not passed else 'Cause'}: {self.CAUSES.get(code, code)}"
        )
        self.line(f"Diagnosis code: {code}")
        self.line(result.get("root_cause_explanation", ""))
        self.line()
        self.line(
            "Evidence: "
            + (
                "Verified against retrieved records and database"
                if passed
                else "Not verified / incomplete"
            ),
            "32" if passed else "33",
        )
        evidence = result.get("evidence", [])
        for item in evidence[:6]:
            self.line(
                f"  - {item['entity']} {item['id']} | {item['field']} = {item['observed']}"
            )
        if len(evidence) > 6:
            self.line(f"  + {len(evidence) - 6} more citations in the full JSON result")
        if not evidence:
            self.line(
                "No final citations accepted; retrieved records remain in the audit trace"
            )
        fix = result.get("proposed_fix", {})
        self.line()
        self.line(
            "Recommended action: " + self.ACTIONS.get(fix.get("action"), "Human review")
        )
        self.line(f"Target: {fix.get('target', 'support analyst')}")
        self.line(f"Confidence: {result.get('confidence', 0):.0%} (model estimate)")
        self.line(f"Escalation: {'Yes' if needs_review else 'No'}")
        self.line("Recommendations only. No ERP data was changed")
        errors = list(
            dict.fromkeys(
                result.get("schema_problems", []) + verification.get("errors", [])
            )
        )
        if errors:
            self.line("Why review is needed:", "33")
            for error in errors[:4]:
                self.line("  - " + self.clean(error))
        for error in result.get("provider_errors", [])[:1]:
            self.line("Provider error: " + error, "33")
        reason = result.get("termination_reason", "Unknown")
        self.line("Stop reason: " + self.STOPS.get(reason, reason.replace("_", " ")))
        usage = result.get("model_usage", {})
        self.line(
            f"Lookups: {result.get('tool_attempts', 0)}/{result.get('tool_budget', 8)} | Model calls: {result.get('model_calls', 0)} | Time: {result.get('elapsed_ms', 0) / 1000:.2f}s"
        )
        self.line(
            f"Reported tokens: {usage.get('input_tokens', 0)} input / {usage.get('output_tokens', 0)} output | Provider retries: {result.get('provider_retries', 0)}"
        )
        if not result.get("usage_complete", True):
            self.line(
                "Token usage is incomplete; do not treat the reported total as exact"
            )
        self.line(
            f"Recovery: {result.get('protocol_repair_attempts', 0)} tool-format correction(s), {result.get('final_repair_attempts', 0)} final-answer correction(s)"
        )
        if score is not None:
            self.line()
            self.line("Evaluation (answer key; not shown to the agent)", "36")
            self.line(
                f"Expected cause: {score['expected_code']} | Expected action: {score['expected_action']}"
            )
            self.line(
                f"Cause correct: {'Yes' if score['code_ok'] else 'No'} | Action correct: {'Yes' if score['action_ok'] else 'No'}"
            )
            self.line(
                "Fully resolved: " + ("YES" if score["resolved"] else "NO"),
                "32" if score["resolved"] else "33",
            )
        self.line("=" * self.width, "90")
