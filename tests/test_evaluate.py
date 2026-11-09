import unittest
from agents.fixed_workflow import FixedDatabaseBaseline
from agents.llm import LLMError
from erpsim.build_world import World, write
from unittest.mock import Mock
from eval.evaluate import _mean, _token_total, aggregate
from eval.experiment import AuditedDatabase, ExperimentRunner
import json
import sqlite3
import tempfile
from pathlib import Path

class TestEvaluation(unittest.TestCase):
    def test_helper_calculations(self):
        usage = {
            "input_tokens": 400,
            "output_tokens": 100,
        }
        self.assertEqual(_token_total(usage),500)
        self.assertEqual(_mean([1, 2, 3]),2.0)

    def test_aggregate_metrics(self):
        rows = [
            {
                "code_ok": True,
                "resolved": True,
                "action_ok": True,
                "evidence_valid": True,
                "evidence_relevant": True,
                "hallucinated_evidence": False,
                "escalated": False,
                "malformed": False,
                "latency_seconds": 1.0,
                "total_tokens": 100,
                "tool_calls": 2,
                "dispatcher_calls": 2,
                "verifier_status": "passed",
            },
            {
                "code_ok": False,
                "resolved": False,
                "action_ok": False,
                "evidence_valid": False,
                "evidence_relevant": False,
                "hallucinated_evidence": True,
                "escalated": True,
                "malformed": True,
                "latency_seconds": 3.0,
                "total_tokens": 300,
                "tool_calls": 4,
                "dispatcher_calls": 3,
                "verifier_status": "failed",
            },
        ]
        summary = aggregate(rows)
        self.assertEqual(summary["total_cases"],2)
        self.assertEqual(summary["root_cause_accuracy_pct"],50.0)
        self.assertEqual(summary["action_accuracy_pct"],50.0)
        self.assertEqual(summary["evidence_valid_pct"],50.0)
        self.assertEqual(summary["hallucinated_evidence_pct"],50.0)
        self.assertEqual(summary["avg_latency_seconds"],2.0)
        self.assertEqual(summary["avg_total_tokens"],200.0)
        self.assertEqual(summary["json_complete_pct"],50.0)
        self.assertEqual(summary["malformed_pct"],50.0)
        self.assertEqual(summary["avg_tool_calls"], 3.0)
        self.assertEqual(summary["avg_dispatcher_calls"], 2.5)
        self.assertEqual(summary["verifier_pass_pct"], 50.0)

    def test_empty_results(self):
        self.assertEqual(aggregate([]),{})

    def test_required_json_shape_is_separate_from_evidence(self):
        diagnosis = {
            "root_cause_code": "UNKNOWN",
            "root_cause_explanation": "Outside the supported causes.",
            "evidence": [],
            "proposed_fix": {"action": "ESCALATE", "target": "support analyst"},
            "confidence": 0.5,
            "escalate": True,
        }
        self.assertTrue(ExperimentRunner.valid_format(diagnosis, json.dumps(diagnosis)))
        self.assertFalse(ExperimentRunner.valid_format(diagnosis, "```json\n{}\n```"))
        for confidence in (float("nan"), float("inf"), True, -0.1, 1.1):
            self.assertFalse(ExperimentRunner.valid_format({**diagnosis, "confidence": confidence}))
        self.assertFalse(ExperimentRunner.valid_format({**diagnosis, "escalate": "true"}))
        self.assertFalse(ExperimentRunner.valid_format({**diagnosis, "schema_problems": ["missing field"]}))

    def test_schema_errors_do_not_count_as_complete_json(self):
        rows = [{
            "code_ok": False, "resolved": False, "action_ok": False,
            "evidence_valid": False, "evidence_relevant": False,
            "hallucinated_evidence": False, "escalated": True,
            "malformed": False, "schema_problems": ["invalid confidence"], "tier": 1,
        }]
        self.assertEqual(aggregate(rows)["json_complete_pct"], 0.0)

    def test_unknown_token_counts_are_not_zero(self):
        self.assertIsNone(_token_total({"input_tokens": 100, "output_tokens": None}))
        self.assertIsNone(_mean([None, None]))
        self.assertEqual(_mean([None, 100, 200]), 150.0)

    def test_agent_database_cannot_read_answer_key_or_write(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "erp.db"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE invoices (id TEXT)")
                connection.execute("CREATE TABLE fault_log (fault_code TEXT)")
                connection.execute("INSERT INTO invoices VALUES ('INV-0001')")
            database = AuditedDatabase(path)
            connection = database.connect()
            try:
                self.assertEqual(connection.execute("SELECT id FROM invoices").fetchone()[0], "INV-0001")
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute("SELECT fault_code FROM fault_log")
                with self.assertRaises(sqlite3.DatabaseError):
                    connection.execute("UPDATE invoices SET id='INV-0002'")
            finally:
                connection.close()
            self.assertEqual(database.denied_reads, 1)
            self.assertEqual(database.denied_writes, 1)



class TestFixedDatabaseBaseline(unittest.TestCase):
    # Build an isolated synthetic invoice case without contacting a model provider.
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "erp.db"
        world = World(42)
        world.build_base()
        world.inject("MISSING_TAX_ID")
        write(world, str(self.path))
        self.ticket = world.tickets[0]
        self.invoice = world.inv_for(world.faults[0]["so_id"])
        self.database = AuditedDatabase(self.path)

    # Changing fault values must not change a fixed lookup sequence or its arguments.
    def test_fixed_plan_ignores_fault_status(self):
        baseline = FixedDatabaseBaseline(self.database)
        trace, issue = baseline.collect(self.ticket)
        self.assertIsNone(issue)
        names = [row["tool_name"] for row in trace]
        self.assertEqual(names, ["get_invoice", "get_sales_order", "get_customer",
                                "get_delivery", "get_payments", "get_price_list",
                                "get_job_runs", "get_job_runs"])
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE invoices SET blocked_reason='OTHER_BLOCKER'")
            connection.execute("UPDATE customers SET on_hold=1")
        changed, issue = baseline.collect(self.ticket)
        self.assertIsNone(issue)
        self.assertEqual([(x["tool_name"], x["arguments"]) for x in trace],
                         [(x["tool_name"], x["arguments"]) for x in changed])
        self.assertEqual(self.database.denied_reads, 0)
        self.assertEqual(self.database.denied_writes, 0)

    # An order ticket uses seven predefined lookups and does not guess an invoice ID.
    def test_order_plan_uses_seven_calls(self):
        ticket = {**self.ticket, "referenced_doc": self.invoice["so_id"]}
        trace, issue = FixedDatabaseBaseline(self.database).collect(ticket)
        self.assertIsNone(issue)
        self.assertEqual(len(trace), 7)
        self.assertEqual(trace[0]["tool_name"], "get_sales_order")
        self.assertNotIn("get_invoice", [x["tool_name"] for x in trace])
        self.assertEqual(trace[-1]["arguments"]["date_range"],
                         {"start_date": "2026-08-21", "end_date": "2026-09-04"})

    # B1-DB makes one text-only model call with records but no hidden answer-key rows.
    def test_one_model_call_receives_observations(self):
        answer = FixedDatabaseBaseline.fallback("Needs a human review")
        completion = Mock(return_value=(json.dumps(answer), {"input_tokens": 100, "output_tokens": 20}))
        result, usage = FixedDatabaseBaseline(self.database, completion=completion).run(self.ticket)
        completion.assert_called_once()
        prompt = completion.call_args.args[0]
        self.assertIn("Retrieved observations:", prompt)
        self.assertIn(self.invoice["id"], prompt)
        self.assertNotIn('"fault_code"', prompt)
        self.assertNotIn('"causal_path"', prompt)
        self.assertEqual(completion.call_args.kwargs["temperature"], 0)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["tool_attempts"], 8)
        self.assertEqual(usage["input_tokens"], 100)

    # A citation with the wrong observed value must fail verification and escalate.
    def test_wrong_value_is_not_accepted_as_evidence(self):
        answer = {
            "root_cause_code": "MISSING_TAX_ID", "root_cause_explanation": "Missing tax ID",
            "evidence": [{"entity": "Customer", "id": self.invoice["customer_id"],
                          "field": "tax_id", "observed": "not-the-value"}],
            "proposed_fix": {"action": "UPDATE_FIELD", "target": self.invoice["customer_id"]},
            "confidence": 0.9, "escalate": False,
        }
        completion = Mock(return_value=(json.dumps(answer), {"input_tokens": 100, "output_tokens": 50}))
        result, _ = FixedDatabaseBaseline(self.database, completion=completion).run(self.ticket)
        self.assertEqual(result["verification"]["status"], "failed")
        self.assertTrue(result["escalate"])
        self.assertEqual(result["proposed_fix"]["action"], "ESCALATE")
        self.assertEqual(result["schema_problems"], [])
        completion.assert_called_once()

    # A supported diagnosis with retrieved, correct citations passes without a repair call.
    def test_valid_database_diagnosis_passes(self):
        answer = {
            "root_cause_code": "MISSING_TAX_ID", "root_cause_explanation": "The tax ID is missing",
            "evidence": [
                {"entity": "Invoice", "id": self.invoice["id"],
                 "field": "blocked_reason", "observed": "POSTING_VALIDATION_FAILED"},
                {"entity": "Customer", "id": self.invoice["customer_id"],
                 "field": "tax_id", "observed": "null"},
            ],
            "proposed_fix": {"action": "UPDATE_FIELD", "target": self.invoice["customer_id"]},
            "confidence": 0.9, "escalate": False,
        }
        completion = Mock(return_value=(json.dumps(answer), {"input_tokens": 100, "output_tokens": 50}))
        result, _ = FixedDatabaseBaseline(self.database, completion=completion).run(self.ticket)
        self.assertEqual(result["verification"]["status"], "passed")
        self.assertFalse(result["escalate"])
        self.assertEqual(result["proposed_fix"]["action"], "UPDATE_FIELD")
        self.assertTrue(ExperimentRunner.valid_format(result, result["raw_response"]))
        completion.assert_called_once()

    # Provider failures retain the completed lookup trace and do not trigger another model call.
    def test_provider_failure_preserves_audit(self):
        completion = Mock(side_effect=LLMError("Provider unavailable"))
        result, usage = FixedDatabaseBaseline(self.database, completion=completion).run(self.ticket)
        self.assertEqual(result["termination_reason"], "provider_error")
        self.assertEqual(result["tool_attempts"], 8)
        self.assertIsNone(usage["input_tokens"])
        self.assertFalse(result["usage_complete"])
        self.assertTrue(result["escalate"])
        completion.assert_called_once()

    # Missing or ambiguous references escalate before any database or model request.
    def test_ambiguous_reference_and_invalid_ticket(self):
        completion = Mock()
        baseline = FixedDatabaseBaseline(self.database, completion=completion)
        ticket = {**self.ticket, "referenced_doc": None, "subject": "Investigate",
                  "body": "Check INV-0001 and INV-0002"}
        result, _ = baseline.run(ticket)
        self.assertEqual(result["tool_attempts"], 0)
        self.assertTrue(result["escalate"])
        result, _ = baseline.run(None)
        self.assertEqual(result["tool_attempts"], 0)
        completion.assert_not_called()

if __name__ == "__main__":
    unittest.main()