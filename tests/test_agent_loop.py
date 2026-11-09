"""Deterministic controller tests for the ERP tool-using agent."""

import json
import tempfile
import unittest
from pathlib import Path

from agents.agent_loop import ERPAgentLoop
from erpsim.build_world import build
from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry


class ScriptedClient:
    """Return predefined model responses without network access."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools_enabled": tools is not None})
        if not self.responses:
            raise AssertionError("The agent made an unexpected model call")
        return self.responses.pop(0)


def tool_response(call_id, name, arguments):
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
        "usage": {"input_tokens": 2, "output_tokens": 1},
        "retries": 0,
    }


def final_response(code="UNKNOWN"):
    action = "ESCALATE" if code == "UNKNOWN" else "RELEASE_HOLD"
    return {
        "message": {
            "role": "assistant",
            "content": json.dumps(
                {
                    "root_cause_code": code,
                    "root_cause_explanation": "Scripted final answer.",
                    "evidence": [],
                    "proposed_fix": {"action": action, "target": "human support"},
                    "confidence": 0.2,
                    "escalate": True,
                }
            ),
        },
        "usage": {"input_tokens": 2, "output_tokens": 1},
        "retries": 0,
    }


class ERPAgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temporary_directory.name) / "erp.db")
        build(seed=42, out=self.db_path, per_code=1)
        self.database = ERPDatabase(self.db_path)
        self.ticket = {
            "id": "TKT-TEST",
            "subject": "Invoice is stuck",
            "body": "Please investigate this invoice.",
            "reported_by": "ops@example.test",
            "created_at": "2026-01-01",
            "referenced_doc": "INV-0001",
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_agent(self, responses, budget=8):
        client = ScriptedClient(responses)
        dispatcher = ToolDispatcher(ToolRegistry(self.database))
        return ERPAgentLoop(client, dispatcher, self.database, budget), client

    def test_one_tool_call_then_final_response(self):
        agent, client = self.make_agent(
            [
                tool_response("call-one", "get_invoice", {"invoice_id": "INV-0001"}),
                final_response(),
            ]
        )

        result = agent.run_loop(self.ticket)

        self.assertEqual(result["termination_reason"], "final_model_response")
        self.assertEqual(result["tool_attempts"], 1)
        self.assertEqual(len(result["tool_trace"]), 1)
        self.assertFalse(result["tool_trace"][0]["cache_hit"])
        self.assertTrue(client.calls[0]["tools_enabled"])
        self.assertTrue(result["escalate"])

    def test_second_duplicate_request_stops_the_loop(self):
        agent, client = self.make_agent(
            [
                tool_response("call-one", "get_invoice", {"invoice_id": "INV-0001"}),
                tool_response("call-two", "get_invoice", {"invoice_id": "INV-0001"}),
                tool_response("call-three", "get_invoice", {"invoice_id": "INV-0001"}),
                final_response(),
            ]
        )

        result = agent.run_loop(self.ticket)

        self.assertEqual(result["termination_reason"], "repeated_unproductive_tool_call")
        self.assertEqual(result["tool_attempts"], 3)
        self.assertEqual(
            [record["cache_hit"] for record in result["tool_trace"]],
            [False, True, True],
        )
        self.assertFalse(client.calls[-1]["tools_enabled"])

    def test_budget_exhaustion_requests_final_json_without_tools(self):
        agent, client = self.make_agent(
            [
                tool_response("call-one", "get_invoice", {"invoice_id": "INV-0001"}),
                final_response(),
            ],
            budget=1,
        )

        result = agent.run_loop(self.ticket)

        self.assertEqual(result["termination_reason"], "tool_budget_exhausted")
        self.assertEqual(result["tool_attempts"], 1)
        self.assertFalse(client.calls[-1]["tools_enabled"])
        self.assertTrue(result["escalate"])

    def test_invalid_ticket_returns_safe_fallback(self):
        agent, client = self.make_agent([])
        invalid_ticket = dict(self.ticket)
        invalid_ticket["id"] = "   "

        result = agent.run_loop(invalid_ticket)

        self.assertEqual(result["root_cause_code"], "UNKNOWN")
        self.assertEqual(result["termination_reason"], "invalid_ticket")
        self.assertTrue(result["escalate"])
        self.assertEqual(client.calls, [])

    def test_parallel_tool_calls_are_rejected(self):
        parallel = tool_response("call-one", "get_invoice", {"invoice_id": "INV-0001"})
        parallel["message"]["tool_calls"].append(
            {
                "id": "call-two",
                "type": "function",
                "function": {
                    "name": "get_sales_order",
                    "arguments": '{"order_id": "SO-0001"}',
                },
            }
        )
        agent, _ = self.make_agent([parallel, final_response()])

        result = agent.run_loop(self.ticket)

        self.assertEqual(result["termination_reason"], "invalid_model_response")
        self.assertTrue(result["escalate"])
        self.assertTrue(result["rejected_tool_calls"])


if __name__ == "__main__":
    unittest.main()
