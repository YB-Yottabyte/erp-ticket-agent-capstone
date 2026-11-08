import argparse
import json
import sys


class AgentCLI:
    # Define separate terminal and JSON modes without changing baseline commands.
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Investigate an ERP ticket using Groq and read-only tools."
        )
        source = self.parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--ticket", help="Ticket ID from the synthetic database")
        source.add_argument("--input", help="Path to a support-ticket text file")
        self.parser.add_argument("--db", default="data/erp.db")
        self.parser.add_argument("--model", default=None)
        self.parser.add_argument(
            "--max-tool-calls", type=int, choices=range(1, 9), default=1
        )
        self.parser.add_argument(
            "--json", action="store_true", help="Emit only structured JSON to stdout"
        )
        self.parser.add_argument("--no-color", action="store_true")
        self.parser.add_argument(
            "--no-animation",
            action="store_true",
            help="Disable the waiting spinner and sentence reveal",
        )
        self.parser.add_argument(
            "--out",
            help="Create a new JSON result file; existing files are not overwritten",
        )

    # Report the pending agent loop without opening the database or model client.
    def run(self):
        arguments = self.parser.parse_args()
        # Restore investigation startup when the assigned agent loop is implemented.
        message = "Agent loop is pending implementation; database tools and evidence verification remain available"
        if arguments.json:
            print(json.dumps({"status": "not_implemented", "message": message}))
        else:
            print(message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(AgentCLI().run())
