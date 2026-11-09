from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from agents.agent_loop import ERPAgentLoop
from agents.diagnosis import DiagnosisValidator
from agents.llm import LLMError
from agents.voyager_client import VoyagerToolClient
from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry
from ui.terminal import AgentTerminal
import run_agent_loop


class ScriptedClient:
    # Keep all requests for checking tool permissions and recovery counts offline.
    def __init__(self, replies):
        self.model = "offline-test"
        self.replies = list(replies)
        self.requests = []

    # Return only scripted observations, never a model-generated substitute result.
    def chat(self, messages, tools=None):
        self.requests.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return deepcopy(reply)


class AgentRecoveryTests(unittest.TestCase):
    # Create a clean, single-fault database using the real schema and read-only tools.
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "erp.db"
        self.db_path = str(path)
        connection = sqlite3.connect(path)
        connection.executescript(
            (Path(__file__).resolve().parents[1] / "erpsim/schema.sql").read_text()
        )
        connection.executescript("""
            INSERT INTO customers (id,name,country,currency,tax_id,payment_terms,credit_limit,credit_used,on_hold)
                VALUES ('C-1','Customer','US','USD','TX1','NET30',1000,0,1);
            INSERT INTO price_lists (id,name,currency,valid_from,valid_to)
                VALUES ('PL-1','Prices','USD','2026-01-01','2027-01-01');
            INSERT INTO sales_orders (id,customer_id,status,order_date,currency,price_list_id,block_reason)
                VALUES ('SO-1','C-1','BLOCKED','2026-09-01','USD','PL-1','APPROVAL_CHECK_FAILED');
            INSERT INTO invoices (id,so_id,customer_id,status,amount,created_at)
                VALUES ('INV-1','SO-1','C-1','BLOCKED',500,'2026-09-01');
        """)
        connection.close()
        self.database = ERPDatabase(path)
        self.ticket = {
            "id": "TKT-1",
            "subject": "Invoice blocked",
            "body": "Please check INV-1",
            "reported_by": "test",
            "created_at": "2026-09-04",
            "referenced_doc": "INV-1",
        }

    # Build a real API-style tool request for the dispatcher to validate.
    def tool(self, name, arguments, call_id):
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    # Supply a complete supported diagnosis independent of any hidden answer key.
    def diagnosis(self):
        return {
            "root_cause_code": "CUSTOMER_ON_HOLD",
            "root_cause_explanation": "The customer account is on hold.",
            "evidence": [
                {"entity": "Customer", "id": "C-1", "field": "on_hold", "observed": "1"}
            ],
            "proposed_fix": {"action": "RELEASE_HOLD", "target": "C-1"},
            "confidence": 0.9,
            "escalate": False,
        }

    # Build a final response or deliberately malformed text for recovery tests.
    def final(self, diagnosis):
        return {
            "message": {
                "role": "assistant",
                "content": (
                    diagnosis if isinstance(diagnosis, str) else json.dumps(diagnosis)
                ),
            },
            "finish_reason": "stop",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    # Execute two useful lookups followed by scripted model responses.
    def run_replies(self, replies, budget=8, prefix=True):
        if prefix:
            replies = [
                self.tool("get_invoice", {"invoice_id": "INV-1"}, "invoice"),
                self.tool("get_customer", {"customer_id": "C-1"}, "customer"),
            ] + replies
        client = ScriptedClient(replies)
        agent = ERPAgentLoop(
            client, ToolDispatcher(ToolRegistry(self.database)), self.database, budget
        )
        return agent.run_loop(self.ticket), client

    # Accept a supported answer without spending unnecessary correction calls.
    def test_valid_diagnosis_passes(self):
        result, client = self.run_replies([self.final(self.diagnosis())])
        self.assertEqual(result["verification"]["status"], "passed")
        self.assertFalse(result["escalate"])
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(result["final_repair_attempts"], 0)

    # Correct missing nested action and nonnumeric confidence with tools disabled.
    def test_schema_repair_once(self):
        broken = self.diagnosis()
        broken["proposed_fix"] = "release the hold"
        broken["confidence"] = "high"
        result, client = self.run_replies(
            [self.final(broken), self.final(self.diagnosis())]
        )
        self.assertEqual(result["verification"]["status"], "passed")
        self.assertEqual(result["final_repair_attempts"], 1)
        self.assertEqual(result["tool_attempts"], 2)
        self.assertIsNone(client.requests[-1]["tools"])
        self.assertEqual(len(result["validation_history"]), 2)
        self.assertEqual(result["model_usage"]["input_tokens"], 40)
        for request in client.requests:
            self.assertEqual(
                [
                    i
                    for i, message in enumerate(request["messages"])
                    if message["role"] == "system"
                ],
                [0],
            )

    # Recheck corrected evidence rather than accepting a model's claim of verification.
    def test_wrong_value_repaired(self):
        broken = self.diagnosis()
        broken["evidence"][0]["observed"] = "0"
        result, _ = self.run_replies([self.final(broken), self.final(self.diagnosis())])
        self.assertEqual(result["verification"]["status"], "passed")
        self.assertEqual(
            result["validation_history"][0]["verification"]["status"], "failed"
        )

    # Stop after one unsuccessful correction while retaining the retrieved observations.
    def test_failed_repair_escalates(self):
        broken = self.diagnosis()
        broken["evidence"][0]["field"] = "on_hold / credit_used"
        result, client = self.run_replies([self.final(broken), self.final(broken)])
        self.assertTrue(result["escalate"])
        self.assertEqual(result["proposed_fix"]["action"], "ESCALATE")
        self.assertEqual(len(client.requests), 4)
        self.assertEqual(len(result["tool_trace"]), 2)
        self.assertEqual(result["verification"]["status"], "failed")

    # Request one native tool call after XML-like text without executing that text.
    def test_text_tool_protocol_recovered(self):
        replies = [
            self.final("<antml:parameter>get_invoice</antml:parameter>"),
            self.tool("get_invoice", {"invoice_id": "INV-1"}, "i"),
            self.tool("get_customer", {"customer_id": "C-1"}, "c"),
            self.final(self.diagnosis()),
        ]
        result, client = self.run_replies(replies, prefix=False)
        self.assertEqual(result["verification"]["status"], "passed")
        self.assertEqual(result["protocol_repair_attempts"], 1)
        self.assertEqual(result["tool_attempts"], 2)
        self.assertEqual(len(client.requests), 4)

    # A real database value is insufficient unless the agent actually retrieved it.
    def test_unretrieved_citation_rejected(self):
        unknown = self.diagnosis()
        unknown.update(
            root_cause_code="UNKNOWN",
            evidence=[],
            proposed_fix={"action": "ESCALATE", "target": "analyst"},
            escalate=True,
        )
        result, client = self.run_replies(
            [self.final(self.diagnosis()), self.final(unknown)], prefix=False
        )
        self.assertTrue(result["escalate"])
        self.assertIn("not retrieved", str(result["validation_history"][0]))
        self.assertEqual(result["tool_attempts"], 0)
        self.assertIsNone(client.requests[-1]["tools"])

    # Preserve exact numeric and null values when adapting to the verifier's strings.
    def test_native_scalar_values_normalized(self):
        diagnosis = self.diagnosis()
        diagnosis["evidence"][0]["observed"] = True
        parsed, errors = DiagnosisValidator().parse(json.dumps(diagnosis))
        self.assertFalse(errors)
        self.assertEqual(parsed["evidence"][0]["observed"], "1")

    # Do not allow duplicate JSON keys, blank targets, or confidence outside the range.
    def test_invalid_final_contract(self):
        examples = [
            self.final(self.diagnosis())["message"]["content"].replace(
                '"confidence": 0.9', '"confidence": 0.9, "confidence": 0.5'
            )
        ]
        for value in (-1, 2, "high", True):
            broken = self.diagnosis()
            broken["confidence"] = value
            examples.append(json.dumps(broken))
        for content in examples:
            with self.subTest(content=content):
                self.assertTrue(DiagnosisValidator().parse(content)[1])

    # Finish budget-limited requests without letting a repair request run another tool.
    def test_repair_cannot_extend_tool_budget(self):
        broken = self.diagnosis()
        broken["evidence"][0]["observed"] = "0"
        replies = [
            self.tool("get_customer", {"customer_id": "C-1"}, "c"),
            self.final(broken),
            self.final(self.diagnosis()),
        ]
        result, client = self.run_replies(replies, budget=1, prefix=False)
        self.assertEqual(result["tool_attempts"], 1)
        self.assertTrue(result["escalate"])
        self.assertIsNone(client.requests[1]["tools"])
        self.assertIsNone(client.requests[2]["tools"])

    # A failed correction request must retain its error and previous database trace.
    def test_provider_failure_during_repair(self):
        result, _ = self.run_replies(
            [self.final("not JSON"), LLMError("request failed")]
        )
        self.assertEqual(result["termination_reason"], "provider_error")
        self.assertTrue(result["escalate"])
        self.assertFalse(result["usage_complete"])
        self.assertEqual(len(result["tool_trace"]), 2)

    # Treat incomplete provider replies as failures even if partial text resembles JSON.
    def test_incomplete_finish_reason(self):
        reply = self.final(self.diagnosis())
        reply["finish_reason"] = "length"
        self.assertEqual(ERPAgentLoop._classify_model_response(reply)[0], "invalid")

    # Enforce the project's eight-attempt maximum at the controller boundary.
    def test_budget_cannot_exceed_eight(self):
        with self.assertRaises(ValueError):
            self.run_replies([], budget=9)

    # Submit the typed final answer without executing it as a database lookup.
    def test_native_final_submission(self):
        submission = self.tool("submit_diagnosis", self.diagnosis(), "submit")
        result, client = self.run_replies([submission])
        self.assertEqual(result["verification"]["status"], "passed")
        self.assertEqual(result["tool_attempts"], 2)
        self.assertEqual(result["model_calls"], 3)
        schema = next(
            tool["function"]["parameters"]
            for tool in client.requests[0]["tools"]
            if tool["function"]["name"] == "submit_diagnosis"
        )
        self.assertEqual(schema, DiagnosisValidator.schema())

    # Voyager can choose the typed submission or a database tool through native calling.
    def test_voyager_explicit_tool_choice(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(self.diagnosis()),
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        ).encode()
        with patch("agents.voyager_client._load_dotenv"), patch.object(
            VoyagerToolClient, "_ssl_context", return_value=object()
        ), patch("urllib.request.urlopen", return_value=response) as send:
            VoyagerToolClient(api_key="test-only").chat(
                [{"role": "user", "content": "Test"}], tools=[{"type": "function"}]
            )
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertNotIn("response_format", payload)

    # Keep terminal summaries compact, honest about verification, and free of controls.
    def test_readable_summary_preserves_result(self):
        result, _ = self.run_replies([self.final(self.diagnosis())])
        before = deepcopy(result)
        stream = StringIO()
        terminal = AgentTerminal(stream)
        result["root_cause_explanation"] += "\x1b[31m"
        terminal.result(result)
        text = stream.getvalue()
        self.assertIn("Verified diagnosis", text)
        self.assertIn("Recommendations only", text)
        self.assertNotIn("tool_trace", text)
        self.assertNotIn("\x1b", text)
        result["root_cause_explanation"] = before["root_cause_explanation"]
        self.assertEqual(result, before)

    # Keep JSON mode and the saved file machine-readable without terminal summaries.
    def test_cli_json_and_audit_file(self):
        result, _ = self.run_replies([self.final(self.diagnosis())])
        output = Path(self.temp.name) / "audit.jsonl"
        agent = MagicMock()
        agent.run_loop.return_value = result
        stream = StringIO()
        args = [
            "run_agent_loop.py",
            "--db",
            self.db_path,
            "--ticket",
            "TKT-1",
            "--provider",
            "voyager",
            "--json",
            "--out",
            str(output),
        ]
        with patch("sys.argv", args), patch(
            "run_agent_loop.load_tickets", return_value=[self.ticket]
        ), patch("run_agent_loop.build_agent", return_value=agent), redirect_stdout(
            stream
        ):
            run_agent_loop.main()
        self.assertEqual(json.loads(stream.getvalue()), result)
        self.assertEqual(json.loads(output.read_text()), result)
        agent.run_loop.assert_called_once_with(self.ticket, on_tool=None)

    # Never report a resolved ticket when the legacy scorer passes but verification fails.
    def test_cli_resolution_requires_verification(self):
        result, _ = self.run_replies([self.final(self.diagnosis())])
        result["verification"] = {
            "status": "failed",
            "errors": ["Unsupported evidence"],
        }
        agent = MagicMock()
        agent.run_loop.return_value = result
        stream = StringIO()
        args = [
            "run_agent_loop.py",
            "--db",
            self.db_path,
            "--ticket",
            "TKT-1",
            "--provider",
            "voyager",
            "--json",
            "--score",
        ]
        with patch("sys.argv", args), patch(
            "run_agent_loop.load_tickets", return_value=[self.ticket]
        ), patch("run_agent_loop.build_agent", return_value=agent), patch(
            "run_agent_loop.fault_for", return_value={"fault_code": "CUSTOMER_ON_HOLD"}
        ), patch(
            "run_agent_loop.score_one", return_value={"resolved": True}
        ), redirect_stdout(
            stream
        ):
            run_agent_loop.main()
        self.assertFalse(json.loads(stream.getvalue())["evaluation"]["resolved"])


if __name__ == "__main__":
    unittest.main()
