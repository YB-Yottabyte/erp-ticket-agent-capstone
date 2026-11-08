from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import Mock, patch

from agents.erp_agent import ERPAgent
from run_agent import AgentCLI


class AgentScaffoldTests(unittest.TestCase):
    # Keep the future loop interface without executing model or database requests.
    def test_loop_is_pending(self):
        client, dispatcher = Mock(), Mock()
        agent = ERPAgent(client, dispatcher, max_tool_calls=8)
        with self.assertRaisesRegex(NotImplementedError, "pending implementation"):
            agent.run({"id": "TKT-LOCAL"})
        client.chat.assert_not_called()
        dispatcher.dispatch_call.assert_not_called()

    # Explain the pending implementation before opening a model connection.
    def test_terminal_command_is_pending(self):
        output = StringIO()
        with patch("sys.argv", ["run_agent.py", "--ticket", "TKT-LOCAL"]), patch(
            "run_agent.GroqToolClient"
        ) as client, redirect_stderr(output):
            self.assertEqual(AgentCLI().run(), 2)
        self.assertIn("pending implementation", output.getvalue())
        client.assert_not_called()

    # Keep pending status machine-readable without creating an output file.
    def test_json_command_is_pending(self):
        output = StringIO()
        with patch(
            "sys.argv", ["run_agent.py", "--ticket", "TKT-LOCAL", "--json"]
        ), patch("run_agent.GroqToolClient") as client, redirect_stdout(output):
            self.assertEqual(AgentCLI().run(), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "not_implemented")
        client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
