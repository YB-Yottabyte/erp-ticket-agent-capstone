from copy import deepcopy
import json
import unittest

from agents.diagnosis import FinalAnswerError, FinalAnswerValidator


class FinalAnswerTests(unittest.TestCase):
    # Define a valid answer whose evidence can exercise native ERP scalar types.
    def setUp(self):
        self.validator = FinalAnswerValidator()
        self.ticket = {"id": "TKT-LOCAL"}
        self.answer = {
            "root_cause_code": "UNKNOWN",
            "root_cause_explanation": "Further investigation needed",
            "evidence": [
                {
                    "entity": "customer",
                    "id": "C-100",
                    "field": "tax_id",
                    "observed": None,
                }
            ],
            "proposed_fix": {"action": "ESCALATE", "target": "support"},
            "confidence": 0.5,
            "escalate": True,
        }

    # Preserve strings, numbers, booleans, and null instead of forcing text conversion.
    def test_native_scalars(self):
        for value in ("BLOCKED", 25000, 25.5, True, False, None):
            with self.subTest(value=value):
                self.answer["evidence"][0]["observed"] = value
                result = self.validator.parse(json.dumps(self.answer), self.ticket)
                self.assertEqual(result["evidence"][0]["observed"], value)
                self.assertIs(type(result["evidence"][0]["observed"]), type(value))
                self.assertEqual(result["evidence"][0]["entity"], "Customer")
                self.assertEqual(result["evidence_status"], "not_verified")

    # Identify the exact citation field when nested values are not supported.
    def test_invalid_evidence(self):
        for value in ({"amount": 25}, [25], float("nan"), float("inf")):
            with self.subTest(value=value):
                self.answer["evidence"][0]["observed"] = value
                with self.assertRaises(FinalAnswerError) as caught:
                    self.validator.parse(json.dumps(self.answer), self.ticket)
                self.assertIn("evidence[0].observed", str(caught.exception))

    # Keep wrong actions, string booleans, and invalid confidence values rejected.
    def test_required_types(self):
        for key, value in (
            ("confidence", "0.5"),
            ("confidence", True),
            ("confidence", 2),
            ("confidence", float("nan")),
            ("escalate", "true"),
            ("evidence", {}),
            ("root_cause_code", "made_up"),
        ):
            with self.subTest(key=key, value=value):
                data = deepcopy(self.answer)
                data[key] = value
                with self.assertRaises(FinalAnswerError) as caught:
                    self.validator.parse(json.dumps(data), self.ticket)
                self.assertIn(key, str(caught.exception))

    # Reject duplicate JSON keys rather than silently using the last conflicting value.
    def test_duplicate_keys(self):
        with self.assertRaises(FinalAnswerError) as caught:
            self.validator.parse('{"confidence":0.2,"confidence":0.9}', self.ticket)
        self.assertIn("duplicate", str(caught.exception))

    # Accept a whole fenced JSON object without searching arbitrary prose for an answer.
    def test_fenced_json(self):
        content = "```json\n" + json.dumps(self.answer) + "\n```"
        self.assertEqual(
            self.validator.parse(content, self.ticket)["ticket_id"], "TKT-LOCAL"
        )
        with self.assertRaises(FinalAnswerError):
            self.validator.parse(
                "Some explanation " + json.dumps(self.answer), self.ticket
            )

    # Report missing citation values distinctly from an explicitly observed null.
    def test_missing_observed(self):
        del self.answer["evidence"][0]["observed"]
        with self.assertRaises(FinalAnswerError) as caught:
            self.validator.parse(json.dumps(self.answer), self.ticket)
        self.assertIn("evidence[0].observed", str(caught.exception))

    # Reject missing targets and unsupported entities without fabricating replacements.
    def test_invalid_target_and_entity(self):
        self.answer["proposed_fix"]["target"] = ""
        self.answer["evidence"][0]["entity"] = "fault_log"
        with self.assertRaises(FinalAnswerError) as caught:
            self.validator.parse(json.dumps(self.answer), self.ticket)
        self.assertIn("proposed_fix.target", str(caught.exception))
        self.assertIn("evidence[0].entity", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
