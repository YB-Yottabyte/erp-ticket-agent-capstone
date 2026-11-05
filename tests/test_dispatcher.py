import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock

from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry


class ToolDispatcherTests(unittest.TestCase):
    # Build a small deterministic ERP fixture independent of the developer database.
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db_path = Path(self.directory.name) / "erp.db"
        schema = Path(__file__).resolve().parents[1] / "erpsim" / "schema.sql"
        connection = sqlite3.connect(self.db_path)
        try:
            connection.executescript(schema.read_text())
            connection.executescript("""
                INSERT INTO customers VALUES
                ('C-100', 'Example Traders', 'US', 'USD', 'TX123', 'NET30', 1000, 100, 0, NULL);
                INSERT INTO price_lists VALUES
                ('PL-10', 'Example list', 'USD', '2026-01-01', '2026-12-31');
                INSERT INTO sales_orders VALUES
                ('SO-0001', 'C-100', 'BLOCKED', '2026-09-01', 'USD', 'PL-10', 'CHECK_FAILED');
                INSERT INTO invoices VALUES
                ('INV-0009', 'SO-0001', 'C-100', 'BLOCKED', 100, 'CHECK_FAILED', '2026-09-01');
                INSERT INTO deliveries VALUES
                ('DN-0008', 'SO-0001', 'PARTIAL', 10, 6, NULL);
                INSERT INTO payments VALUES
                ('PAY-0001', 'C-100', 50, '2026-09-01', 0);
                INSERT INTO job_runs VALUES
                ('JR-0001', 'credit_sync', '2026-09-01', 'FAILED', 'connection reset');
            """)
            connection.commit()
        finally:
            connection.close()
        self.database = ERPDatabase(self.db_path)
        self.registry = ToolRegistry(self.database)
        self.dispatcher = ToolDispatcher(self.registry)

    # Check registration and parameterized retrieval for all eight tools.
    def test_all_eight_tools(self):
        cases = {
            "find_document": {"reference": "INV-0009"},
            "get_invoice": {"invoice_id": "INV-0009"},
            "get_sales_order": {"order_id": "SO-0001"},
            "get_customer": {"customer_id": "C-100"},
            "get_delivery": {"order_id": "SO-0001"},
            "get_payments": {"customer_id": "C-100"},
            "get_price_list": {"price_list_id": "PL-10"},
            "get_job_runs": {
                "job_name": "credit_sync",
                "date_range": {"start_date": "2026-09-01", "end_date": "2026-09-01"},
            },
        }
        self.assertEqual(
            {d["function"]["name"] for d in self.registry.definitions()}, set(cases)
        )
        for name, arguments in cases.items():
            with self.subTest(tool=name):
                result = self.dispatcher.dispatch(name, arguments, "call_test")
                self.assertEqual(result.observation["status"], "ok")
                self.assertGreaterEqual(result.elapsed_ms, 0)
                json.dumps(result.to_record(), allow_nan=False)

    # Reject helper access, answer-key access, and arbitrary SQL names.
    def test_unapproved_tools(self):
        for name in (
            "connect",
            "fault_log",
            "execute",
            "__getattribute__",
            "missing",
            None,
            [],
        ):
            with self.subTest(name=name):
                result = self.dispatcher.dispatch(name, {})
                self.assertEqual(result.observation["status"], "invalid_tool")
                json.dumps(result.to_record(), allow_nan=False)

    # Prove invalid arguments never reach a database handler.
    def test_reject_arguments_before_execution(self):
        self.database.get_invoice = Mock()
        dispatcher = ToolDispatcher(ToolRegistry(self.database))
        for arguments in (
            {},
            {"invoice_id": 9},
            {"invoice_id": None},
            {"invoice_id": "INV-0009", "db_path": "other.db"},
            {"invoice_id": "INV-0009' OR 1=1 --"},
            {"invoice_id": "INV-0009\n"},
            [],
            None,
            "[]",
            "null",
            "{bad json}",
            '{"invoice_id":"INV-0009","invoice_id":"INV-0010"}',
            {"invoice_id": {1, 2}},
            "x" * 8193,
        ):
            with self.subTest(arguments=arguments):
                result = dispatcher.dispatch("get_invoice", arguments)
                self.assertEqual(result.observation["status"], "invalid_input")
                json.dumps(result.to_record(), allow_nan=False)
        self.database.get_invoice.assert_not_called()

    # Validate nested date-range fields and preserve database date-order checks.
    def test_date_validation(self):
        for dates in (
            {},
            {"start_date": "2026-09-01", "end_date": None},
            {"start_date": "2026-02-30", "end_date": "2026-09-01"},
            {"start_date": "20260901", "end_date": "2026-09-01"},
            {"start_date": "2026-09-02", "end_date": "2026-09-01"},
            {"start_date": "2026-09-01", "end_date": "2026-09-01", "extra": True},
        ):
            with self.subTest(dates=dates):
                result = self.dispatcher.dispatch(
                    "get_job_runs", {"job_name": "credit_sync", "date_range": dates}
                )
                self.assertEqual(result.observation["status"], "invalid_input")
        result = self.dispatcher.dispatch(
            "get_job_runs", {"job_name": "unknown", "date_range": {}}
        )
        self.assertEqual(result.observation["status"], "invalid_input")

    # Keep missing records distinct from unavailable-database errors.
    def test_database_outcomes(self):
        result = self.dispatcher.dispatch("get_invoice", {"invoice_id": "INV-9999"})
        self.assertEqual(result.observation["status"], "not_found")
        dispatcher = ToolDispatcher(
            ToolRegistry(ERPDatabase(self.db_path.parent / "missing.db"))
        )
        result = dispatcher.dispatch("get_invoice", {"invoice_id": "INV-0009"})
        self.assertEqual(result.observation["status"], "database_error")

    # Preserve model call IDs and JSON-encode the tool observation for the next turn.
    def test_groq_call_round_trip(self):
        result = self.dispatcher.dispatch_call(
            {
                "id": "call_invoice",
                "type": "function",
                "function": {
                    "name": "get_invoice",
                    "arguments": '{"invoice_id":"INV-0009"}',
                },
            }
        )
        message = result.to_message()
        self.assertEqual(message["role"], "tool")
        self.assertEqual(message["tool_call_id"], "call_invoice")
        self.assertEqual(json.loads(message["content"])["invoice"]["id"], "INV-0009")

    # Reject malformed model-call envelopes rather than emitting unlinked messages.
    def test_invalid_call_envelopes(self):
        for call in (
            None,
            [],
            {},
            {"id": "", "type": "function", "function": {}},
            {"id": "x", "type": "other", "function": {}},
            {"id": "x", "type": "function", "function": []},
        ):
            with self.subTest(call=call), self.assertRaises(ValueError):
                self.dispatcher.dispatch_call(call)
        with self.assertRaises(ValueError):
            self.dispatcher.dispatch(
                "get_invoice", {"invoice_id": "INV-0009"}
            ).to_message()
        for call_id in ("", "  ", 123):
            with self.subTest(call_id=call_id), self.assertRaises(ValueError):
                self.dispatcher.dispatch(
                    "get_invoice", {"invoice_id": "INV-0009"}, call_id
                )

    # Keep tool exceptions contained and retain their type without leaking details.
    def test_handler_failure(self):
        self.database.get_invoice = Mock(side_effect=RuntimeError("private diagnostic"))
        dispatcher = ToolDispatcher(ToolRegistry(self.database))
        result = dispatcher.dispatch("get_invoice", {"invoice_id": "INV-0009"})
        self.assertEqual(result.observation["status"], "tool_error")
        self.assertEqual(result.error_type, "RuntimeError")
        self.assertNotIn("private diagnostic", json.dumps(result.to_record()))

    # Reject malformed handler results without passing broken JSON to the model.
    def test_bad_handler_result(self):
        for observation in (None, {"status": "ok", "value": float("nan")}):
            with self.subTest(observation=observation):
                self.database.get_invoice = Mock(return_value=observation)
                result = ToolDispatcher(ToolRegistry(self.database)).dispatch(
                    "get_invoice", {"invoice_id": "INV-0009"}
                )
                self.assertEqual(result.observation["status"], "tool_error")

    # Protect registry schemas from edits to the definitions sent to the model.
    def test_definitions_are_copies(self):
        definitions = self.registry.definitions()
        definitions[0]["function"]["name"] = "fault_log"
        self.assertEqual(
            self.registry.definitions()[0]["function"]["name"], "find_document"
        )
        _, parameters = self.registry.resolve("get_invoice")
        parameters["properties"]["invoice_id"]["type"] = "object"
        self.assertEqual(
            self.dispatcher.dispatch(
                "get_invoice", {"invoice_id": "INV-0009"}
            ).observation["status"],
            "ok",
        )

    # Ensure tool execution cannot change the fixture database.
    def test_database_remains_unchanged(self):
        before = self.db_path.read_bytes()
        self.dispatcher.dispatch("get_invoice", {"invoice_id": "INV-0009"})
        self.dispatcher.dispatch("find_document", {"reference": "Example"})
        self.assertEqual(self.db_path.read_bytes(), before)
        connection = self.database.connect()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("UPDATE invoices SET amount = 0")
        finally:
            connection.close()


if __name__ == "__main__":
    unittest.main()
