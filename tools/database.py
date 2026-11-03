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
