import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sqlite3
import sys

from agents.erp_agent import ERPAgent
from agents.groq_client import GroqToolClient
from agents.llm import LLMError
from run_baseline import parse_ticket_file
from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry
from ui.terminal import TerminalUI


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

    # Load only public ticket columns through a read-only database connection.
    def _load_ticket(self, arguments, database):
        if arguments.input:
            return parse_ticket_file(arguments.input)
        connection = database.connect()
        try:
            row = connection.execute(
                "SELECT id, subject, body, reported_by, created_at, referenced_doc FROM tickets WHERE id = ?",
                (arguments.ticket,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise ValueError("Ticket was not found")
        return dict(row)

    # Run one investigation and keep terminal rendering separate from structured data.
    def run(self):
        arguments = self.parser.parse_args()
        try:
            if arguments.out and Path(arguments.out).exists():
                raise ValueError("Output file already exists; choose a new path")
            database = ERPDatabase(arguments.db)
            ticket = self._load_ticket(arguments, database)
            client = GroqToolClient(model=arguments.model)
            dispatcher = ToolDispatcher(ToolRegistry(database))
            agent = ERPAgent(client, dispatcher, arguments.max_tool_calls)
            terminal = (
                None
                if arguments.json
                else TerminalUI(
                    color=not arguments.no_color, animate=not arguments.no_animation
                )
            )
            if terminal:
                terminal.show_header(
                    ticket["id"],
                    client.model,
                    arguments.max_tool_calls,
                    subject=ticket.get("subject"),
                )
            with terminal.waiting() if terminal else nullcontext():
                result = agent.run(
                    ticket, on_tool=terminal.show_tool_result if terminal else None
                )
            serialized = json.dumps(result.to_dict(), indent=2, allow_nan=False)
            if arguments.out:
                with open(arguments.out, "x", encoding="utf-8") as output:
                    output.write(serialized + "\n")
            if arguments.json:
                print(serialized)
            else:
                terminal.show_summary(result)
            return (
                1
                if result.metrics["termination_reason"]
                in {
                    "provider_error",
                    "invalid_tool_response",
                    "invalid_output",
                    "incomplete_model_response",
                }
                else 0
            )
        except (LLMError, OSError, sqlite3.Error, ValueError):
            if arguments.json:
                print(
                    json.dumps(
                        {
                            "status": "error",
                            "message": "Agent setup or output failed; check the database, ticket, API key, and output path",
                        }
                    )
                )
            else:
                print(
                    "Agent setup or output failed. Check the database, ticket, API key, and output path.",
                    file=sys.stderr,
                )
            return 1


if __name__ == "__main__":
    raise SystemExit(AgentCLI().run())
