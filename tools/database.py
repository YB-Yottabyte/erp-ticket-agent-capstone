from pathlib import Path
from datetime import date
import sqlite3

class ERPDatabase:
    # Build address: "data/erp.db" -> "file:///project/data/erp.db?mode=ro"
    def __init__(self, db_path="data/erp.db"):
        self.database_uri = Path(db_path).resolve().as_uri() + "?mode=ro"

    # Open read-only connection; access results by name: row["status"]
    def connect(self):
        connection = sqlite3.connect(self.database_uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    # Find invoice: "INV-0342" -> matching invoice details, or not_found if absent
    def get_invoice(self, invoice_id):
        if not isinstance(invoice_id, str):
            return {
                "status": "invalid_input",
                "message": "Invoice ID must be text",
            }
        invoice_id = invoice_id.strip()

        if not invoice_id.startswith("INV-") or not invoice_id[4:].isdigit():
            return {
                "status": "invalid_input",
                "message": "Invalid invoice ID",
            }
        try:
            connection = self.connect()

            try:
                row = connection.execute(
                    "SELECT id, so_id, customer_id, status, amount, blocked_reason, created_at FROM invoices WHERE id = ?", (invoice_id,)
                ).fetchone()
            finally:
                connection.close()

        except sqlite3.Error:
            return {
                "status": "database_error",
                "message": "Could not read the invoice",
            }

        if row is None:
            return {
                "status": "not_found",
                "message": invoice_id
            }

        return {
            "status": "ok",
            "invoice": dict(row),
        }

    # Find sales order: "SO-0001" -> order details, or not_found if absent
    def get_sales_order(self, order_id):
        if not isinstance(order_id, str):
            return {
                "status": "invalid_input",
                "message": "Order ID must be text",
            }
        order_id = order_id.strip()

        if not order_id.startswith("SO-") or not order_id[3:].isdigit():
            return {
                "status": "invalid_input",
                "message": "Invalid order ID",
            }
        try:
            connection = self.connect()

            try:
                row = connection.execute(
                    "SELECT id, customer_id, status, order_date, currency, price_list_id, block_reason FROM sales_orders WHERE id = ?",
                    (order_id,),
                ).fetchone()
            finally:
                connection.close()

        except sqlite3.Error:
            return {
                "status": "database_error",
                "message": "Could not read the sales order",
            }
        if row is None:
            return {
                "status": "not_found",
                "message": order_id
            }
        return {
            "status": "ok",
            "order": dict(row),
        }

    # Find customer: "C-128" -> account details, or not_found if absent
    def get_customer(self, customer_id):
        if not isinstance(customer_id, str):
            return {
                "status": "invalid_input",
                "message": "Customer ID must be text",
            }
        customer_id = customer_id.strip()

        if not customer_id.startswith("C-") or not customer_id[2:].isdigit():
            return {
                "status": "invalid_input",
                "message": "Invalid customer ID",
            }
        try:
            connection = self.connect()

            try:
                row = connection.execute(
                    "SELECT id, name, country, currency, tax_id, payment_terms, credit_limit, credit_used, on_hold, hold_reason FROM customers WHERE id = ?",
                    (customer_id,),
                ).fetchone()
            finally:
                connection.close()

        except sqlite3.Error:
            return {
                "status": "database_error",
                "message": "Could not read the customer",
            }
        if row is None:
            return {
                "status": "not_found",
                "message": customer_id
            }
        return {
            "status": "ok",
            "customer": dict(row),
        }

    # Find deliveries: "SO-0001" -> shipment records, or not_found if absent.
    def get_delivery(self, order_id):
        if not isinstance(order_id, str):
            return {
                "status": "invalid_input",
                "message": "Order ID must be text",
            }

        order_id = order_id.strip()

        if not order_id.startswith("SO-") or not order_id[3:].isdigit():
            return {
                "status": "invalid_input",
                "message": "Invalid order ID",
            }

        try:
            connection = self.connect()

            try:
                rows = connection.execute(
                    """
                    SELECT id, so_id, status, ordered_qty,
                           shipped_qty, delivered_at
                    FROM deliveries
                    WHERE so_id = ?
                    ORDER BY id
                    """,
                    (order_id,),
                ).fetchall()
            finally:
                connection.close()

        except sqlite3.Error:
            return {
                "status": "database_error",
                "message": "Could not read the deliveries",
            }

        if not rows:
            return {
                "status": "not_found",
                "message": "No delivery records found for this order",
                "order_id": order_id,
            }

        return {
            "status": "ok",
            "deliveries": [dict(row) for row in rows],
        }

    # Find payments: "C-128" -> payment records, or not_found if absent.
    def get_payments(self, customer_id):
        if not isinstance(customer_id, str):
            return {
                "status": "invalid_input",
                "message": "Customer ID must be text",
            }

        customer_id = customer_id.strip()

        if not customer_id.startswith("C-") or not customer_id[2:].isdigit():
            return {
                "status": "invalid_input",
                "message": "Invalid customer ID",
            }

        try:
            connection = self.connect()

            try:
                rows = connection.execute(
                    """
                    SELECT id, customer_id, amount,
                           received_at, applied_to_limit
                    FROM payments
                    WHERE customer_id = ?
                    ORDER BY received_at DESC, id
                    """,
                    (customer_id,),
                ).fetchall()
            finally:
                connection.close()

        except sqlite3.Error:
            return {
                "status": "database_error",
                "message": "Could not read the payments",
            }

        if not rows:
            return {
                "status": "not_found",
                "message": "No payment records found for this customer",
                "customer_id": customer_id,
            }

        return {
            "status": "ok",
            "payments": [dict(row) for row in rows],
        }

    # Find price list: "PL-13" -> validity dates, or not_found if absent.
    def get_price_list(self, price_list_id):
        if not isinstance(price_list_id, str):
            return {
                "status": "invalid_input",
                "message": "Price list ID must be text",
            }

        price_list_id = price_list_id.strip()

        if not price_list_id.startswith("PL-") or not price_list_id[3:].isdigit():
            return {
                "status": "invalid_input",
                "message": "Invalid price list ID",
            }

        try:
            connection = self.connect()

            try:
                row = connection.execute(
                    """
                    SELECT id, name, currency, valid_from, valid_to
                    FROM price_lists
                    WHERE id = ?
                    """,
                    (price_list_id,),
                ).fetchone()
            finally:
                connection.close()

        except sqlite3.Error:
            return {
                "status": "database_error",
                "message": "Could not read the price list",
            }

        if row is None:
            return {
                "status": "not_found",
                "price_list_id": price_list_id,
            }

        return {
            "status": "ok",
            "price_list": dict(row),
        }

    # Find job runs: "credit_sync" + date range -> job history, or not_found.
    def get_job_runs(self, job_name, date_range):
        if not isinstance(job_name, str):
            return {
                "status": "invalid_input",
                "message": "Job name must be text",
            }

        job_name = job_name.strip()

        if job_name not in {"credit_sync", "price_sync", "ledger_post"}:
            return {
                "status": "invalid_input",
                "message": "Unsupported job name",
            }

        if not isinstance(date_range, dict):
            return {
                "status": "invalid_input",
                "message": "Date range must contain start_date and end_date",
            }

        start_date = date_range.get("start_date")
        end_date = date_range.get("end_date")

        try:
            if not isinstance(start_date, str) or not isinstance(end_date, str):
                raise ValueError

            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)

            if start.isoformat() != start_date or end.isoformat() != end_date:
                raise ValueError
            if start > end:
                raise ValueError

        except ValueError:
            return {
                "status": "invalid_input",
                "message": "Use valid YYYY-MM-DD dates with start_date <= end_date",
            }

        try:
            connection = self.connect()

            try:
                rows = connection.execute(
                    """
                    SELECT id, job_name, started_at, status, error
                    FROM job_runs
                    WHERE job_name = ? AND started_at BETWEEN ? AND ?
                    ORDER BY started_at DESC, id
                    """,
                    (job_name, start_date, end_date),
                ).fetchall()
            finally:
                connection.close()

        except sqlite3.Error:
            return {
                "status": "database_error",
                "message": "Could not read the job history",
            }

        if not rows:
            return {
                "status": "not_found",
                "message": "No job runs found in this date range",
                "job_name": job_name,
            }

        return {
            "status": "ok",
            "job_runs": [dict(row) for row in rows],
        }
