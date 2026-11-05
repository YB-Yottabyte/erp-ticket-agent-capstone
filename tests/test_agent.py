from copy import deepcopy
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents.erp_agent import ERPAgent
from agents.llm import LLMError
from erpsim.build_world import build
from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry
from ui.terminal import TerminalUI
from run_agent import AgentCLI


class ScriptedClient:
    # Supply deterministic responses without calling a live model or claiming accuracy.
    def __init__(self, replies):
        self.model = "scripted-test-only"
        self.replies = list(replies)
        self.requests = []

    # Capture message history before returning the next scripted response.
    def chat(self, messages, tools=None):
        self.requests.append({"messages": deepcopy(messages), "tools": deepcopy(tools)})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return deepcopy(reply)


class AgentTests(unittest.TestCase):
    # Generate a temporary ERP fixture and keep the answer key out of model messages.
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db_path = Path(self.directory.name) / "erp.db"
        build(seed=42, out=str(self.db_path))
        self.dispatcher = ToolDispatcher(ToolRegistry(ERPDatabase(self.db_path)))
        self.ticket = {
            "id": "TKT-LOCAL",
            "subject": "Invoice stuck",
            "body": "Please inspect INV-0009",
            "created_at": "2026-09-04",
        }

    # Make a realistic model tool request with explicit token usage.
    def _tool(self, call_id="call_1", name="get_invoice", arguments=None):
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(
                                arguments
                                if arguments is not None
                                else {"invoice_id": "INV-0009"}
                            ),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "retries": 0,
        }

    # Make a provisional final answer that still requires human evidence review.
    def _final(self):
        diagnosis = {
            "root_cause_code": "UNKNOWN",
            "root_cause_explanation": "Insufficient evidence",
            "evidence": [],
            "proposed_fix": {"action": "ESCALATE", "target": "support"},
            "confidence": 0.0,
            "escalate": True,
        }
        return {
            "message": {"role": "assistant", "content": json.dumps(diagnosis)},
            "finish_reason": "stop",
            "usage": {"input_tokens": 20, "output_tokens": 10},
            "retries": 0,
        }

    # Verify the complete model-tool-model cycle and correct call-ID association.
    def test_one_cycle(self):
        client = ScriptedClient([self._tool(), self._final()])
        result = ERPAgent(client, self.dispatcher).run(self.ticket)
        self.assertEqual(result.metrics["tool_calls"], 1)
        self.assertEqual(result.metrics["model_calls"], 2)
        self.assertEqual(
            result.metrics["reported_tokens"], {"input_tokens": 30, "output_tokens": 15}
        )
        self.assertIsNone(client.requests[1]["tools"])
        tool_message = client.requests[1]["messages"][-2]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_1")
        self.assertEqual(
            json.loads(tool_message["content"])["invoice"]["id"], "INV-0009"
        )
        self.assertEqual(result.diagnosis["evidence_status"], "not_verified")
        json.dumps(result.to_dict(), allow_nan=False)

    # Count failed requests against the budget while allowing a corrected next call.
    def test_invalid_arguments_recovery(self):
        client = ScriptedClient(
            [
                self._tool(arguments={"invoice_id": 9}),
                self._tool("call_2"),
                self._final(),
            ]
        )
        result = ERPAgent(client, self.dispatcher, 2).run(self.ticket)
        self.assertEqual(result.metrics["tool_calls"], 2)
        self.assertEqual(
            [item["observation"]["status"] for item in result.trace],
            ["invalid_input", "ok"],
        )

    # Return a prior observation for a repeated request without another database read.
    def test_repeated_lookup_is_cached(self):
        client = ScriptedClient([self._tool(), self._tool("call_2"), self._final()])
        result = ERPAgent(client, self.dispatcher, 2).run(self.ticket)
        self.assertEqual(result.metrics["tool_calls"], 2)
        self.assertEqual(result.metrics["dispatcher_calls"], 1)
        self.assertTrue(result.trace[1]["cache_hit"])

    # Stop when the model ignores the tool limit rather than executing extra calls.
    def test_budget_cannot_be_exceeded(self):
        client = ScriptedClient([self._tool(), self._tool("call_2")])
        result = ERPAgent(client, self.dispatcher).run(self.ticket)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.metrics["termination_reason"], "invalid_tool_response")
        self.assertTrue(result.diagnosis["escalate"])

    # Do not accept a text-only guess as a completed database investigation.
    def test_no_tool_response(self):
        result = ERPAgent(ScriptedClient([self._final()]), self.dispatcher).run(
            self.ticket
        )
        self.assertEqual(result.metrics["termination_reason"], "no_investigation")

    # Keep model transport failures separate from a wrong diagnosis.
    def test_provider_error(self):
        result = ERPAgent(
            ScriptedClient([LLMError("private error")]), self.dispatcher
        ).run(self.ticket)
        self.assertEqual(result.metrics["termination_reason"], "provider_error")
        self.assertFalse(result.metrics["usage_complete"])
        self.assertNotIn("private error", json.dumps(result.to_dict()))

    # Reject malformed tool calls and duplicate call IDs without crashing.
    def test_bad_tool_calls(self):
        bad = self._tool()
        bad["message"]["tool_calls"][0].pop("id")
        result = ERPAgent(ScriptedClient([bad]), self.dispatcher).run(self.ticket)
        self.assertEqual(result.metrics["termination_reason"], "invalid_tool_response")
        result = ERPAgent(
            ScriptedClient([self._tool(), self._tool()]), self.dispatcher, 2
        ).run(self.ticket)
        self.assertEqual(result.metrics["termination_reason"], "invalid_tool_response")

    # Reject parallel calls because this first version supports sequential execution.
    def test_parallel_response(self):
        reply = self._tool()
        reply["message"]["tool_calls"].append(
            self._tool("call_2")["message"]["tool_calls"][0]
        )
        result = ERPAgent(ScriptedClient([reply]), self.dispatcher).run(self.ticket)
        self.assertEqual(result.metrics["termination_reason"], "invalid_tool_response")
        self.assertEqual(result.metrics["tool_calls"], 0)
        self.assertEqual(result.metrics["rejected_tool_calls"], 2)

    # Treat truncated replies and malformed JSON as failures, not final diagnoses.
    def test_invalid_final(self):
        for content, finish in (("not json", "stop"), ("{}", "length")):
            with self.subTest(content=content):
                final = self._final()
                final["message"]["content"] = content
                final["finish_reason"] = finish
                result = ERPAgent(
                    ScriptedClient([self._tool(), final]), self.dispatcher
                ).run(self.ticket)
                self.assertTrue(result.diagnosis["escalate"])
                self.assertIn(
                    result.metrics["termination_reason"],
                    {"invalid_output", "incomplete_model_response"},
                )

    # Keep unknown token usage distinct from a real zero-token measurement.
    def test_missing_usage(self):
        final = self._final()
        final["usage"] = {}
        result = ERPAgent(ScriptedClient([self._tool(), final]), self.dispatcher).run(
            self.ticket
        )
        self.assertFalse(result.metrics["usage_complete"])

    # Never display a schema-valid model claim as verified evidence.
    def test_verification_is_pending(self):
        final = self._final()
        diagnosis = json.loads(final["message"]["content"])
        diagnosis.update(
            root_cause_code="MISSING_TAX_ID", escalate=False, confidence=0.9
        )
        final["message"]["content"] = json.dumps(diagnosis)
        result = ERPAgent(ScriptedClient([self._tool(), final]), self.dispatcher).run(
            self.ticket
        )
        self.assertTrue(result.diagnosis["escalate"])
        self.assertTrue(result.metrics["verification_pending"])

    # Avoid carrying conversation state or cached results into the next ticket.
    def test_state_is_per_run(self):
        client = ScriptedClient(
            [self._tool(), self._final(), self._tool(), self._final()]
        )
        agent = ERPAgent(client, self.dispatcher)
        first, second = agent.run(self.ticket), agent.run(self.ticket)
        self.assertFalse(first.trace[0]["cache_hit"])
        self.assertFalse(second.trace[0]["cache_hit"])
        self.assertEqual(len(client.requests[2]["messages"]), 2)

    # Pass only public ticket fields and suppress terminal control sequences.
    def test_untrusted_content(self):
        client = ScriptedClient([self._tool(), self._final()])
        ticket = dict(self.ticket, fault_log="hidden answer")
        result = ERPAgent(client, self.dispatcher).run(ticket)
        self.assertNotIn("hidden answer", json.dumps(client.requests))
        stream = StringIO()
        terminal = TerminalUI(stream=stream, color=False)
        terminal.show_header("\x1b[31m[bold]ticket", "model", 1)
        terminal.show_summary(result)
        self.assertNotIn("\x1b", stream.getvalue())
        self.assertIn("[bold]ticket", stream.getvalue())
        self.assertIn("Not verified", stream.getvalue())

    # Keep command-line JSON free of terminal output and persist the same result.
    def test_cli_json(self):
        client = ScriptedClient([self._tool(), self._final()])
        path = Path(self.directory.name) / "result.json"
        output, errors = StringIO(), StringIO()
        arguments = [
            "run_agent.py",
            "--ticket",
            "TKT-1001",
            "--db",
            str(self.db_path),
            "--json",
            "--out",
            str(path),
        ]
        with patch("sys.argv", arguments), patch(
            "run_agent.GroqToolClient", return_value=client
        ), redirect_stdout(output), redirect_stderr(errors):
            status = AgentCLI().run()
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(output.getvalue()), json.loads(path.read_text()))
        self.assertEqual(errors.getvalue(), "")

    # Preserve existing result files instead of overwriting research records.
    def test_cli_output_exists(self):
        path = Path(self.directory.name) / "result.json"
        path.write_text("previous result")
        arguments = [
            "run_agent.py",
            "--ticket",
            "TKT-1001",
            "--json",
            "--out",
            str(path),
        ]
        output = StringIO()
        with patch("sys.argv", arguments), redirect_stdout(output):
            status = AgentCLI().run()
        self.assertEqual(status, 1)
        self.assertEqual(path.read_text(), "previous result")
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
