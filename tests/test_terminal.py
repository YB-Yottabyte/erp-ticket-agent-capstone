from copy import deepcopy
from io import StringIO
import os
import unittest
from unittest.mock import patch

from agents.erp_agent import AgentResult
from ui.terminal import TerminalUI


class TerminalStream(StringIO):
    # Simulate an interactive terminal without opening a real window.
    def isatty(self):
        return True


class TerminalTests(unittest.TestCase):
    # Use explicit display fixtures rather than making model calls for UI tests.
    def setUp(self):
        self.result = AgentResult(
            diagnosis={
                "ticket_id": "TKT-1011",
                "root_cause_code": "CREDIT_LIMIT_STALE_SYNC",
                "root_cause_explanation": "The retrieved records suggest a credit-sync issue; further evidence review is needed.",
                "evidence": [
                    {
                        "entity": "Invoice",
                        "id": "INV-0009",
                        "field": "status",
                        "observed": "BLOCKED",
                    }
                ],
                "proposed_fix": {"action": "RERUN_JOB", "target": "credit_sync"},
                "escalate": True,
            },
            metrics={
                "tool_calls": 1,
                "tool_budget": 1,
                "elapsed_seconds": 2.16,
                "reported_tokens": {"input_tokens": 1800, "output_tokens": 646},
                "model_calls": 2,
                "retries": 0,
                "usage_complete": True,
                "termination_reason": "budget_limit",
            },
            trace=[],
        )
        self.record = {
            "tool_name": "get_invoice",
            "observation": {
                "status": "ok",
                "invoice": {
                    "id": "INV-0009",
                    "status": "BLOCKED",
                    "blocked_reason": "CREDIT_CHECK_FAILED",
                },
            },
            "cache_hit": False,
        }

    # Preserve the reference-style panel and evidence tree without Rich installed.
    def test_panel_and_tree(self):
        stream = StringIO()
        terminal = TerminalUI(stream, width=100)
        terminal.show_header("TKT-1011", "openai/gpt-oss-20b", 1)
        terminal.show_tool_result(self.record, 1, 1)
        terminal.show_summary(self.result)
        output = stream.getvalue()
        for text in (
            "╭",
            "╰",
            ">_ ERP Ticket Agent",
            "GPT-OSS 20B via Groq",
            "● get_invoice",
            "├─",
            "INV-0009",
            "Recommended action",
            "Not verified",
            "2446",
        ):
            self.assertIn(text, output)
        self.assertNotIn("\033", output)

    # Emit the gold, green, and teal palette only for an interactive terminal.
    def test_interactive_colors(self):
        stream = TerminalStream()
        with patch.dict(os.environ, {"TERM": "xterm-256color"}, clear=True):
            terminal = TerminalUI(stream, width=100)
        terminal.show_header("TKT-1011", "openai/gpt-oss-20b", 1)
        terminal.show_tool_result(self.record, 1, 1)
        for color in ("accent", "green", "teal"):
            self.assertIn(TerminalUI.COLORS[color], stream.getvalue())

    # Respect explicit no-color settings while keeping the visual structure.
    def test_color_opt_out(self):
        for environment, enabled in (
            ({"NO_COLOR": ""}, True),
            ({"TERM": "dumb"}, True),
            ({}, False),
        ):
            with self.subTest(environment=environment):
                stream = TerminalStream()
                with patch.dict(os.environ, environment, clear=True):
                    terminal = TerminalUI(stream, color=enabled)
                terminal.show_header("ticket", "model", 1)
                self.assertNotIn("\033", stream.getvalue())
                self.assertIn("╭", stream.getvalue())

    # Wrap output instead of breaking columns on narrow terminal windows.
    def test_narrow_terminal(self):
        for width in (34, 60, 80, 120):
            with self.subTest(width=width):
                stream = StringIO()
                terminal = TerminalUI(stream, width=width)
                terminal.show_header("TKT-1011", "openai/gpt-oss-20b", 1)
                terminal.show_tool_result(self.record, 1, 1)
                terminal.show_summary(self.result)
                self.assertTrue(
                    all(
                        len(line) <= terminal.width
                        for line in stream.getvalue().splitlines()
                    ),
                    stream.getvalue(),
                )

    # Display missing tax IDs because they are relevant ERP evidence.
    def test_null_tax_id(self):
        terminal = TerminalUI(StringIO())
        facts = terminal._record_facts({"customer": {"id": "C-100", "tax_id": None}})
        self.assertIn(("tax id: ", None), facts)

    # Distinguish failed lookups and cache hits from successful new database reads.
    def test_failure_and_cache(self):
        stream = StringIO()
        terminal = TerminalUI(stream)
        terminal.show_tool_result(
            {"tool_name": "get_invoice", "observation": {"status": "not_found"}}, 1, 2
        )
        cached = deepcopy(self.record)
        cached["cache_hit"] = True
        terminal.show_tool_result(cached, 2, 2)
        self.assertIn("not_found", stream.getvalue())
        self.assertIn("cached", stream.getvalue())

    # Sanitize terminal controls and avoid modifying the research result during rendering.
    def test_controls_and_result_preservation(self):
        before = self.result.to_dict()
        stream = StringIO()
        terminal = TerminalUI(stream, color=False)
        terminal.show_header("\033[31m[bold]ticket\n", "model", 1)
        terminal.show_summary(self.result)
        self.assertNotIn("\033", stream.getvalue())
        self.assertIn("[bold]ticket", stream.getvalue())
        self.assertEqual(before, self.result.to_dict())


if __name__ == "__main__":
    unittest.main()
