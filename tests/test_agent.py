from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

from run_agent import AgentCLI


class AgentScaffoldTests(unittest.TestCase):
    # Explain the pending implementation before opening a model connection.
    def test_terminal_command_is_pending(self):
        output = StringIO()
        with patch("sys.argv", ["run_agent.py", "--ticket", "TKT-LOCAL"]), redirect_stderr(output):
            self.assertEqual(AgentCLI().run(), 2)
        self.assertIn("pending implementation", output.getvalue())

    # Keep pending status machine-readable without creating an output file.
    def test_json_command_is_pending(self):
        output = StringIO()
        with patch(
            "sys.argv", ["run_agent.py", "--ticket", "TKT-LOCAL", "--json"]
        ), redirect_stdout(output):
            self.assertEqual(AgentCLI().run(), 2)
        self.assertEqual(json.loads(output.getvalue())["status"], "not_implemented")


if __name__ == "__main__":
    unittest.main()
