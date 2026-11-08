import sqlite3
import tempfile
import unittest
from pathlib import Path

from agents.evidence import verify_diagnosis
from tools.database import ERPDatabase


class TestDiagnosisVerifier(unittest.TestCase):
    def setUp(self):
        """Create a temporary database using the real project schema."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        db_path = Path(self.temp_dir.name) / "erp.db"
        project_dir = Path(__file__).resolve().parents[1]
        schema = (project_dir / "erpsim" / "schema.sql").read_text()

        connection = sqlite3.connect(db_path)

        try:
            connection.executescript(schema)

            connection.executescript("""
                INSERT INTO customers (
                    id, name, country, currency, tax_id,
                    payment_terms, credit_limit, credit_used, on_hold
                ) VALUES (
                    'C-1', 'Example Customer', 'US', 'USD', NULL,
                    'NET30', 1000, 1000, 1
                );

                INSERT INTO customers (
                    id, name, country, currency, tax_id,
                    payment_terms, credit_limit, credit_used, on_hold
                ) VALUES (
                    'C-2', 'Other Customer', 'US', 'USD', 'TX2',
                    'NET30', 1000, 100, 1
                );

                INSERT INTO price_lists (
                    id, name, currency, valid_from, valid_to
                ) VALUES (
                    'PL-1', 'Old Prices', 'USD',
                    '2026-01-01', '2026-08-31'
                );

                INSERT INTO sales_orders (
                    id, customer_id, status, order_date,
                    currency, price_list_id
                ) VALUES (
                    'SO-1', 'C-1', 'BLOCKED', '2026-09-01',
                    'USD', 'PL-1'
                );

                INSERT INTO invoices (
                    id, so_id, customer_id, status, amount, created_at
                ) VALUES (
                    'INV-1', 'SO-1', 'C-1', 'BLOCKED',
                    500, '2026-09-02'
                );

                INSERT INTO deliveries (
                    id, so_id, status, ordered_qty, shipped_qty
                ) VALUES (
                    'DEL-1', 'SO-1', 'PARTIAL', 100, 60
                );

                INSERT INTO payments (
                    id, customer_id, amount,
                    received_at, applied_to_limit
                ) VALUES (
                    'PAY-1', 'C-1', 500, '2026-09-02', 0
                );

                INSERT INTO job_runs (
                    id, job_name, started_at, status, error
                ) VALUES (
                    'JR-1', 'credit_sync', '2026-09-03',
                    'FAILED', 'Connection failed'
                );

                INSERT INTO job_runs (
                    id, job_name, started_at, status, error
                ) VALUES (
                    'JR-2', 'price_sync', '2026-09-03',
                    'FAILED', 'Feed unavailable'
                );
            """)

            connection.commit()
        finally:
            connection.close()

        self.database = ERPDatabase(db_path)
        self.ticket = {
            "id": "TKT-1",
            "referenced_doc": "INV-1",
            "created_at": "2026-09-04",
        }

    def claim(self, entity, record_id, field, observed):
        """Build one citation in the agent's output format."""
        return {
            "entity": entity,
            "id": record_id,
            "field": field,
            "observed": observed,
        }

    def diagnosis(self, code, action, target, evidence):
        """Build a diagnosis without calling an AI model."""
        return {
            "root_cause_code": code,
            "root_cause_explanation": "Example explanation",
            "proposed_fix": {
                "action": action,
                "target": target,
            },
            "evidence": evidence,
            "confidence": 0.8,
            "escalate": False,
        }

    def check(self, diagnosis, expected_status, ticket=None):
        """Run the verifier and show its errors if the assertion fails."""
        result = verify_diagnosis(
            ticket if ticket is not None else self.ticket,
            diagnosis,
            self.database,
        )
        self.assertEqual(
            result["status"],
            expected_status,
            result,
        )

    def hold_diagnosis(self):
        """Return a supported customer-hold diagnosis."""
        return self.diagnosis(
            "CUSTOMER_ON_HOLD",
            "RELEASE_HOLD",
            "C-1",
            [self.claim("Customer", "C-1", "on_hold", "1")],
        )

    def price_evidence(self):
        """Return citations linking this order to an expired price list."""
        return [
            self.claim("SalesOrder", "SO-1", "price_list_id", "PL-1"),
            self.claim("PriceList", "PL-1", "valid_to", "2026-08-31"),
        ]

    def job_evidence(self, record_id, job_name):
        """Return the required citations for a failed sync run."""
        return [
            self.claim("JobRun", record_id, "job_name", job_name),
            self.claim("JobRun", record_id, "status", "FAILED"),
            self.claim("JobRun", record_id, "started_at", "2026-09-03"),
        ]

    def test_customer_hold_passes(self):
        self.check(self.hold_diagnosis(), "passed")

    def test_missing_tax_id_passes(self):
        diagnosis = self.diagnosis(
            "MISSING_TAX_ID",
            "UPDATE_FIELD",
            "C-1",
            [self.claim("Customer", "C-1", "tax_id", "null")],
        )
        self.check(diagnosis, "passed")

    def test_expired_price_list_passes(self):
        diagnosis = self.diagnosis(
            "STALE_PRICE_LIST",
            "REPRICE_ORDER",
            "SO-1",
            self.price_evidence(),
        )
        self.check(diagnosis, "passed")

    def test_incomplete_delivery_passes(self):
        diagnosis = self.diagnosis(
            "INCOMPLETE_DELIVERY_BLOCKS_INVOICE",
            "COMPLETE_DELIVERY",
            "DEL-1",
            [
                self.claim("Delivery", "DEL-1", "ordered_qty", "100"),
                self.claim("Delivery", "DEL-1", "shipped_qty", "60"),
            ],
        )
        self.check(diagnosis, "passed")

    def test_failed_credit_sync_passes(self):
        evidence = [
            self.claim("Customer", "C-1", "credit_limit", "1000"),
            self.claim("Customer", "C-1", "credit_used", "1000"),
            self.claim("Payment", "PAY-1", "applied_to_limit", "0"),
            self.claim("Payment", "PAY-1", "amount", "500"),
            self.claim("Payment", "PAY-1", "received_at", "2026-09-02"),
        ]
        evidence += self.job_evidence("JR-1", "credit_sync")

        diagnosis = self.diagnosis(
            "CREDIT_LIMIT_STALE_SYNC",
            "RERUN_JOB",
            "credit_sync",
            evidence,
        )
        self.check(diagnosis, "passed")

    def test_failed_price_sync_passes(self):
        evidence = self.price_evidence()
        evidence += self.job_evidence("JR-2", "price_sync")

        diagnosis = self.diagnosis(
            "PRICE_SYNC_JOB_FAILED",
            "RERUN_JOB",
            "price_sync",
            evidence,
        )
        self.check(diagnosis, "passed")

    def test_true_but_unrelated_customer_fails(self):
        diagnosis = self.hold_diagnosis()
        diagnosis["evidence"] = [
            self.claim("Customer", "C-2", "on_hold", "1")
        ]
        diagnosis["proposed_fix"]["target"] = "C-2"

        self.check(diagnosis, "failed")

    def test_wrong_database_value_fails(self):
        diagnosis = self.hold_diagnosis()
        diagnosis["evidence"][0]["observed"] = "0"

        self.check(diagnosis, "failed")

    def test_matching_value_without_issue_support_fails(self):
        diagnosis = self.hold_diagnosis()

        # The name is correct, but it does not establish an account hold.
        diagnosis["evidence"] = [
            self.claim("Customer", "C-1", "name", "Example Customer")
        ]
        self.check(diagnosis, "failed")

    def test_wrong_action_fails(self):
        diagnosis = self.hold_diagnosis()
        diagnosis["proposed_fix"]["action"] = "REPRICE_ORDER"

        self.check(diagnosis, "failed")

    def test_wrong_target_fails(self):
        diagnosis = self.hold_diagnosis()
        diagnosis["proposed_fix"]["target"] = "C-2"

        self.check(diagnosis, "failed")

    def test_reference_in_ticket_text_passes(self):
        ticket = {
            "created_at": "2026-09-04",
            "referenced_doc": None,
            "body": "Please investigate invoice INV-1.",
        }
        self.check(self.hold_diagnosis(), "passed", ticket)

    def test_missing_reference_fails(self):
        ticket = {
            "created_at": "2026-09-04",
            "referenced_doc": None,
            "body": "Please investigate.",
        }
        self.check(self.hold_diagnosis(), "failed", ticket)

    def test_unknown_requires_review(self):
        diagnosis = self.diagnosis(
            "UNKNOWN", "ESCALATE", "support analyst", []
        )
        self.check(diagnosis, "insufficient")

    def test_failed_job_alone_is_not_enough(self):
        diagnosis = self.diagnosis(
            "PRICE_SYNC_JOB_FAILED",
            "RERUN_JOB",
            "price_sync",
            self.job_evidence("JR-2", "price_sync"),
        )
        self.check(diagnosis, "failed")


if __name__ == "__main__":
    unittest.main()