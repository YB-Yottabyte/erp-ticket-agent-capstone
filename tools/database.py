from pathlib import Path
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
