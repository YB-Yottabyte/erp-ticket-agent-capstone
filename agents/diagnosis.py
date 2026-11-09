from copy import deepcopy
import json
import math

from agents.evidence import (
    ALLOWED_FIELDS,
    ENTITY_TABLE,
    normalize_entity,
    validate_output,
    value_matches,
)
from erpsim.taxonomy import ACTIONS, codes


class DiagnosisValidator:
    # Define one answer contract for the model, controller, and evidence verifier.
    @staticmethod
    def schema():
        text = {"type": "string", "minLength": 1}
        return {
            "type": "object",
            "properties": {
                "root_cause_code": {"type": "string", "enum": codes()},
                "root_cause_explanation": text,
                "evidence": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": text,
                            "id": text,
                            "field": text,
                            "observed": {"type": "string"},
                        },
                        "required": ["entity", "id", "field", "observed"],
                        "additionalProperties": False,
                    },
                },
                "proposed_fix": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ACTIONS},
                        "target": text,
                    },
                    "required": ["action", "target"],
                    "additionalProperties": False,
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "escalate": {"type": "boolean"},
            },
            "required": [
                "root_cause_code",
                "root_cause_explanation",
                "evidence",
                "proposed_fix",
                "confidence",
                "escalate",
            ],
            "additionalProperties": False,
        }

    # Refuse ambiguous JSON rather than silently choosing duplicate field values.
    @staticmethod
    def _unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    # Accept complete JSON only and preserve scalar citation values without guessing.
    def parse(self, content):
        if not isinstance(content, str) or len(content) > 65536:
            return None, ["Final answer must be JSON text within 65536 characters"]
        try:
            result = json.loads(content, object_pairs_hook=self._unique_object)
        except (ValueError, RecursionError):
            return None, [
                "Final answer must be one complete JSON object, without prose or tool markup"
            ]
        if not isinstance(result, dict):
            return None, ["Final answer must be a JSON object"]
        result = deepcopy(result)
        evidence = result.get("evidence")
        if isinstance(evidence, list):
            if len(evidence) > 64:
                return None, ["Use at most 64 individual citations"]
            for item in evidence:
                if not isinstance(item, dict) or "observed" not in item:
                    continue
                value = item["observed"]
                if value is None:
                    item["observed"] = "null"
                elif type(value) is bool:
                    item["observed"] = "1" if value else "0"
                elif type(value) in (int, float) and math.isfinite(value):
                    item["observed"] = str(value)
        errors = validate_output(result)
        if errors:
            return None, errors
        if set(result) != set(self.schema()["required"]):
            errors.append("Use exactly the six required final-answer fields")
        for index, item in enumerate(result["evidence"], 1):
            table = ENTITY_TABLE.get(normalize_entity(item["entity"]))
            if table is None or item["field"] not in ALLOWED_FIELDS.get(table, set()):
                errors.append(
                    f"Evidence {index}: cite one supported entity and one exact database field; do not combine fields"
                )
            if set(item) != {"entity", "id", "field", "observed"}:
                errors.append(f"Evidence {index}: use only entity, id, field, observed")
        if set(result["proposed_fix"]) != {"action", "target"}:
            errors.append("proposed_fix needs only action and target")
        return (None, errors) if errors else (result, [])

    # Require every citation to match a field returned by an approved tool this run.
    @staticmethod
    def check_observations(diagnosis, trace):
        keys = {
            "get_invoice": ("invoices", "invoice"),
            "get_sales_order": ("sales_orders", "order"),
            "get_customer": ("customers", "customer"),
            "get_price_list": ("price_lists", "price_list"),
            "get_delivery": ("deliveries", "deliveries"),
            "get_payments": ("payments", "payments"),
            "get_job_runs": ("job_runs", "job_runs"),
        }
        records = {}
        for entry in trace:
            observation = entry.get("observation", {})
            tool = keys.get(entry.get("tool_name"))
            if tool is None or observation.get("status") != "ok":
                continue
            table, key = tool
            rows = observation.get(key)
            rows = [rows] if isinstance(rows, dict) else rows
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict):
                    records[(table, row.get("id"))] = row
        errors = []
        for index, item in enumerate(diagnosis["evidence"], 1):
            table = ENTITY_TABLE.get(normalize_entity(item["entity"]))
            row = records.get((table, item["id"]))
            if row is None or item["field"] not in row:
                errors.append(
                    f"Evidence {index}: record and field were not retrieved during this ticket"
                )
            elif not value_matches(row[item["field"]], item["observed"]):
                errors.append(
                    f"Evidence {index}: observed value differs from the retrieved result"
                )
        return errors
