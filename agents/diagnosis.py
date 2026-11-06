from copy import deepcopy
import json
import math

from erpsim.taxonomy import ACTIONS, codes


class FinalAnswerError(ValueError):
    # Keep field-level errors separate from the model's untrusted answer text.
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(errors))


class FinalAnswerValidator:
    ENTITIES = {
        "invoice": "Invoice",
        "salesorder": "SalesOrder",
        "customer": "Customer",
        "payment": "Payment",
        "jobrun": "JobRun",
        "delivery": "Delivery",
        "pricelist": "PriceList",
    }

    # Give the model a compact format template instead of repeating the full schema.
    def prompt_format(self):
        return json.dumps(
            {
                "root_cause_code": "UNKNOWN",
                "root_cause_explanation": "<explain using retrieved facts>",
                "evidence": [
                    {
                        "entity": "Invoice",
                        "id": "<retrieved ID>",
                        "field": "<exact database column>",
                        "observed": "<retrieved value; native numbers, booleans, and null are also allowed>",
                    }
                ],
                "proposed_fix": {
                    "action": "ESCALATE",
                    "target": "<retrieved target or support analyst>",
                },
                "confidence": 0.0,
                "escalate": True,
            }
        )

    # Describe the exact final format, including native JSON evidence values.
    def schema(self):
        return {
            "type": "object",
            "required": [
                "root_cause_code",
                "root_cause_explanation",
                "evidence",
                "proposed_fix",
                "confidence",
                "escalate",
            ],
            "properties": {
                "root_cause_code": {"type": "string", "enum": codes()},
                "root_cause_explanation": {"type": "string", "minLength": 1},
                "evidence": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "required": ["entity", "id", "field", "observed"],
                        "properties": {
                            "entity": {
                                "type": "string",
                                "enum": list(self.ENTITIES.values()),
                            },
                            "id": {"type": "string", "minLength": 1},
                            "field": {"type": "string", "minLength": 1},
                            "observed": {
                                "type": ["string", "number", "boolean", "null"]
                            },
                        },
                    },
                },
                "proposed_fix": {
                    "type": "object",
                    "required": ["action", "target"],
                    "properties": {
                        "action": {"type": "string", "enum": ACTIONS},
                        "target": {"type": "string", "minLength": 1},
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "escalate": {"type": "boolean"},
            },
        }

    # Reject duplicate keys instead of silently accepting a contradictory last value.
    @staticmethod
    def _unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise FinalAnswerError(["final_answer: duplicate JSON key"])
            result[key] = value
        return result

    # Parse the answer and report exact field paths without inventing missing values.
    def parse(self, content, ticket):
        if not isinstance(content, str) or len(content) > 65536:
            raise FinalAnswerError(
                ["final_answer: expected JSON text of at most 65536 characters"]
            )
        text = content.strip()
        if text.startswith("```json\n") and text.endswith("```"):
            text = text[8:-3].strip()
        elif text.startswith("```\n") and text.endswith("```"):
            text = text[4:-3].strip()
        try:
            result = json.loads(text, object_pairs_hook=self._unique_object)
        except FinalAnswerError:
            raise
        except (ValueError, RecursionError):
            raise FinalAnswerError(["final_answer: invalid JSON object"]) from None
        if not isinstance(result, dict):
            raise FinalAnswerError(["final_answer: must be an object"])
        errors = []
        if result.get("root_cause_code") not in codes():
            errors.append("root_cause_code: must use an allowed code")
        if (
            not isinstance(result.get("root_cause_explanation"), str)
            or not result["root_cause_explanation"].strip()
        ):
            errors.append("root_cause_explanation: must be nonempty text")
        fix = result.get("proposed_fix")
        if not isinstance(fix, dict):
            errors.append("proposed_fix: must contain action and target")
        else:
            if fix.get("action") not in ACTIONS:
                errors.append("proposed_fix.action: must use an allowed action")
            if not isinstance(fix.get("target"), str) or not fix["target"].strip():
                errors.append("proposed_fix.target: must be nonempty text")
        confidence = result.get("confidence")
        if (
            type(confidence) not in (int, float)
            or not 0 <= confidence <= 1
            or not math.isfinite(confidence)
        ):
            errors.append("confidence: must be a finite number from 0 to 1")
        if type(result.get("escalate")) is not bool:
            errors.append("escalate: must be a JSON boolean")
        evidence = result.get("evidence")
        normalized = []
        if not isinstance(evidence, list) or len(evidence) > 64:
            errors.append("evidence: must be a list of at most 64 citations")
        else:
            for index, item in enumerate(evidence):
                path = f"evidence[{index}]"
                if not isinstance(item, dict):
                    errors.append(f"{path}: must be an object")
                    continue
                entity = item.get("entity")
                canonical = (
                    self.ENTITIES.get(entity.lower().replace("_", "").replace(" ", ""))
                    if isinstance(entity, str)
                    else None
                )
                if canonical is None:
                    errors.append(f"{path}.entity: unsupported ERP entity")
                for key in ("id", "field"):
                    if not isinstance(item.get(key), str) or not item[key].strip():
                        errors.append(f"{path}.{key}: must be nonempty text")
                observed = item.get("observed")
                if "observed" not in item or type(observed) not in (
                    str,
                    int,
                    float,
                    bool,
                    type(None),
                ):
                    errors.append(
                        f"{path}.observed: must be a JSON scalar, not a list or object"
                    )
                elif type(observed) is float and not math.isfinite(observed):
                    errors.append(f"{path}.observed: number must be finite")
                normalized.append(
                    {
                        "entity": canonical,
                        "id": item.get("id"),
                        "field": item.get("field"),
                        "observed": observed,
                    }
                )
        if errors:
            raise FinalAnswerError(errors)
        diagnosis = {key: deepcopy(result[key]) for key in self.schema()["required"]}
        diagnosis.update(
            evidence=normalized,
            ticket_id=ticket.get("id", "TKT-LOCAL"),
            evidence_status="not_verified",
            escalate=True,
        )
        return diagnosis
