"""Run the verified tool-using ERP ticket diagnosis agent."""

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from agents.agent_loop import ERPAgent
from agents.groq_client import GroqToolClient
from agents.voyager_client import VoyagerToolClient
from agents.llm import LLMError
from eval.score import fault_for, score_one, summarise
from run_baseline import parse_ticket_file
from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry
from ui.terminal import AgentTerminal

DEFAULT_DB = "data/erp.db"


def load_tickets(db_path, ticket_id=None, limit=None):
    """Load public ticket fields through a read-only SQLite connection."""
    database_uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM tickets"
        arguments = ()
        if ticket_id:
            query += " WHERE id = ?"
            arguments = (ticket_id,)
        query += " ORDER BY id"
        if limit is not None:
            query += " LIMIT ?"
            arguments += (limit,)
        return [dict(row) for row in connection.execute(query, arguments)]
    finally:
        connection.close()


def build_agent(db_path, model, max_tool_calls, provider="groq"):
    """Build the live agent from the repository's read-only integrations."""
    database = ERPDatabase(db_path)
    dispatcher = ToolDispatcher(ToolRegistry(database))
    clients = {"groq": GroqToolClient, "voyager": VoyagerToolClient}
    if provider not in clients:
        raise ValueError("Provider must be groq or voyager")
    client = clients[provider](model=model)
    return ERPAgent(client, dispatcher, database, max_tool_calls=max_tool_calls)


def main():
    parser = argparse.ArgumentParser(
        description="Run the verified tool-using ERP ticket diagnosis agent."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="path to a ticket text file")
    source.add_argument("--ticket", help="ticket ID from the database")
    source.add_argument("--all", action="store_true", help="every generated ticket")
    parser.add_argument("--db", default=DEFAULT_DB, help="read-only ERP database path")
    parser.add_argument(
        "--provider",
        choices=["groq", "voyager"],
        default="groq",
        help="model provider (default: groq)",
    )
    parser.add_argument(
        "--model", default=None, help="optional provider model override"
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        choices=range(1, 9),
        default=8,
        help="maximum logical tool attempts",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="limit database tickets"
    )
    parser.add_argument("--score", action="store_true", help="score against fault_log")
    parser.add_argument("--out", help="write JSONL results to this path")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print full JSONL audit records instead of the readable summary",
    )
    parser.add_argument(
        "--no-color", action="store_true", help="disable terminal colors"
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        sys.exit(
            f"no database at {args.db}\n"
            f"run: python -m erpsim.build_world --seed 42 --out {args.db}"
        )
    if args.limit is not None and args.limit < 1:
        sys.exit("--limit must be at least 1")

    tickets = (
        [parse_ticket_file(args.input)]
        if args.input
        else load_tickets(args.db, args.ticket, args.limit)
    )
    if not tickets:
        sys.exit("no tickets matched")

    try:
        agent = build_agent(args.db, args.model, args.max_tool_calls, args.provider)
    except (LLMError, TypeError, ValueError) as error:
        sys.exit(f"agent setup error: {error}")

    score_connection = None
    if args.score:
        score_uri = Path(args.db).resolve().as_uri() + "?mode=ro"
        score_connection = sqlite3.connect(score_uri, uri=True)
    rows = []
    results = []
    terminal = None if args.json else AgentTerminal(color=not args.no_color)

    try:
        for ticket in tickets:
            if terminal:
                terminal.header(
                    ticket, args.provider, agent.client.model, args.max_tool_calls
                )
            result = agent.run_loop(
                ticket, on_tool=terminal.tool_result if terminal else None
            )
            reasoning = getattr(agent.client, "reasoning_effort", None)
            result.update(
                provider=args.provider,
                model=str(agent.client.model),
                temperature=0.0,
                reasoning_effort=reasoning if isinstance(reasoning, str) else None,
            )
            results.append(result)
            row = None

            if args.score:
                fault = fault_for(score_connection, ticket["id"])
                if fault is None:
                    result["evaluation"] = {
                        "status": "not_scored",
                        "reason": "No answer key for this ticket",
                    }
                else:
                    row = score_one(score_connection, result, fault)
                    row["verification_passed"] = (result.get("verification") or {}).get(
                        "status"
                    ) == "passed"
                    row["resolved"] = bool(
                        row["resolved"]
                        and row["verification_passed"]
                        and not result.get("schema_problems")
                        and not result.get("escalate")
                    )
                    result["evaluation"] = row
                    rows.append(row)
            if args.json:
                print(json.dumps(result, allow_nan=False))
            else:
                terminal.result(result, row)

        if args.score and len(rows) > 1:
            summary = summarise(rows)
            summary["verification_pass_pct"] = round(
                100 * sum(row["verification_passed"] for row in rows) / len(rows), 1
            )
            if args.json:
                print(json.dumps({"type": "evaluation_summary", **summary}))
            else:
                terminal.line(f"Evaluation summary: {len(rows)} tickets", "36")
                terminal.line(
                    f"Root-cause accuracy: {summary['root_cause_accuracy_pct']}% | Verified full resolution: {summary['full_resolution_pct']}%"
                )
                terminal.line(
                    f"Evidence verification passed: {summary['verification_pass_pct']}% | Malformed answers: {summary['malformed_pct']}%"
                )
                terminal.line(
                    "Record-ID hallucination rate does not measure field/value accuracy"
                )
    finally:
        if score_connection is not None:
            score_connection.close()

    if args.out:
        parent = os.path.dirname(args.out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as output_file:
            for result in results:
                output_file.write(json.dumps(result, allow_nan=False) + "\n")
        if terminal:
            terminal.line(f"Full JSON results and audit traces saved to {args.out}")


if __name__ == "__main__":
    main()
