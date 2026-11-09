import sqlite3
import math
import re
from datetime import date, timedelta

from erpsim.taxonomy import ACTIONS, EXPECTED_ACTION, codes


# Only these tables and fields can be checked.
# Never accept arbitrary table or field names from the agent.
ALLOWED_FIELDS = {
    "customers": {
        "id", "name", "country", "currency", "tax_id",
        "payment_terms", "credit_limit", "credit_used",
        "on_hold", "hold_reason",
    },
    "invoices": {
        "id", "so_id", "customer_id", "status", "amount",
        "blocked_reason", "created_at",
    },
    "sales_orders": {
        "id", "customer_id", "status", "order_date",
        "currency", "price_list_id", "block_reason",
    },
    "deliveries": {
        "id", "so_id", "status", "ordered_qty",
        "shipped_qty", "delivered_at",
    },
    "payments": {
        "id", "customer_id", "amount",
        "received_at", "applied_to_limit",
    },
    "price_lists": {
        "id", "name", "currency", "valid_from", "valid_to",
    },
    "job_runs": {
        "id", "job_name", "started_at", "status", "error",
    },
    "items": {
        "id", "name", "uom", "discontinued",
    },
    "sales_order_lines": {
        "id", "so_id", "item_id", "qty", "unit_price",
    },
    "stock_ledger": {
        "id", "item_id", "warehouse",
        "qty_delta", "ref_doc", "posted_at",
    },
}

# Accept the entity names already used by eval/score.py.
ENTITY_TABLE = {
    "customer": "customers",
    "invoice": "invoices",
    "salesorder": "sales_orders",
    "order": "sales_orders",
    "delivery": "deliveries",
    "payment": "payments",
    "pricelist": "price_lists",
    "jobrun": "job_runs",
    "job": "job_runs",
    "item": "items",
    "salesorderline": "sales_order_lines",
    "stockledger": "stock_ledger",
}

def normalize_entity(entity):
    """Example: 'Sales_Order' becomes 'salesorder'."""
    return entity.lower().replace("_", "").replace(" ", "")

def value_matches(actual, observed):
    """Compare a database value with the agent's string value."""
    if actual is None:
        # Agree on this convention with the agent teammate.
        return observed == "null"

    if isinstance(actual, (int, float)):
        # Allow equivalent numbers such as "1" and "1.0".
        from decimal import Decimal, InvalidOperation

        try:
            claimed = Decimal(observed)
            return claimed.is_finite() and claimed == Decimal(str(actual))
        except InvalidOperation:
            return False

    return observed == str(actual)

def verify_evidence(diagnosis, database):
    """
    diagnosis: result.diagnosis from ERPAgent.run()
    database: an ERPDatabase instance
    
    Returns a result for each citation plus an overall result.
    """
    if not isinstance(diagnosis, dict):
        return {
            "citations_valid": False,
            "checks": [],
            "errors": ["Diagnosis must be a dictionary"],
        }

    evidence = diagnosis.get("evidence")

    if not isinstance(evidence, list) or not evidence:
        return {
            "citations_valid": False,
            "checks": [],
            "errors": ["No evidence list was provided, or it is empty"],
        }

    checks = []
    errors = []

    try:
        # ERPDatabase.connect() already opens the database read-only.
        connection = database.connect()
    except sqlite3.Error:
        return {
            "citations_valid": False,
            "checks": [],
            "errors": ["Could not open the database"],
        }

    try:
        for index, item in enumerate(evidence):
            check = {"index": index, "valid": False}

            required = ("entity", "id", "field", "observed")

            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str)
                for key in required
            ):
                reason = "Evidence needs entity, id, field, and observed strings"
            else:
                table = ENTITY_TABLE.get(normalize_entity(item["entity"]))
                field = item["field"]

                if table is None:
                    reason = "Unsupported entity"

                elif field not in ALLOWED_FIELDS[table]:
                    reason = "Field is not allowed for this entity"

                else:
                    try:
                        # Table and field names come from the allowlists above.
                        # Record IDs are passed as SQL parameters.
                        row = connection.execute(
                            f'SELECT "{field}" FROM "{table}" WHERE id = ?',
                            (item["id"],),
                        ).fetchone()

                        if row is None:
                            reason = "Record not found"
                        elif not value_matches(row[0], item["observed"]):
                            reason = "Cited value does not match the database"
                        else:
                            reason = "Record, field, and value match"
                            check["valid"] = True

                    except sqlite3.Error:
                        reason = "Could not read the cited record"

            check["reason"] = reason
            checks.append(check)

            if not check["valid"]:
                errors.append(f"Evidence {index + 1}: {reason}")

    finally:
        connection.close()

    return {
        "citations_valid": not errors,
        "checks": checks,
        "errors": errors,
    }


# Each entry defines which field connects two entities.
RELATIONSHIPS = {
    ("invoice", "customer"): ("invoices", "customer_id", "customers"),
    ("salesorder", "customer"): (
        "sales_orders", "customer_id", "customers"
    ),
    ("salesorder", "pricelist"): (
        "sales_orders", "price_list_id", "price_lists"
    ),
    ("delivery", "salesorder"): (
        "deliveries", "so_id", "sales_orders"
    ),
    ("payment", "customer"): (
        "payments", "customer_id", "customers"
    ),
    ("invoice", "salesorder"): (
        "invoices", "so_id", "sales_orders"
    ),
}

def verify_relationship(
    database,
    source_entity,
    source_id,
    target_entity,
    target_id,
):
    """
    Check a direct relationship between two existing records.
    
    Returns status: passed, failed, or not_checked.
    """
    values = (source_entity, source_id, target_entity, target_id)

    if any(not isinstance(value, str) or not value.strip()
           for value in values):
        return {
            "status": "failed",
            "reason": "Entity names and IDs must be nonempty strings",
        }

    # Uses the normalize_entity() already in this file.
    source = normalize_entity(source_entity)
    target = normalize_entity(target_entity)

    # Accept the existing Order alias.
    source = "salesorder" if source == "order" else source
    target = "salesorder" if target == "order" else target

    relationship = RELATIONSHIPS.get((source, target))

    # Also allow callers to supply the records in reverse order.
    if relationship is None:
        relationship = RELATIONSHIPS.get((target, source))
        if relationship is not None:
            source_id, target_id = target_id, source_id

    if relationship is None:
        return {
            "status": "not_checked",
            "reason": "No supported direct relationship",
        }

    source_table, linking_field, target_table = relationship
    connection = None

    try:
        connection = database.connect()

        # SQL identifiers come only from RELATIONSHIPS.
        # Record IDs are SQL parameters.
        source_row = connection.execute(
            f'SELECT "{linking_field}" '
            f'FROM "{source_table}" WHERE id = ?',
            (source_id,),
        ).fetchone()

        if source_row is None:
            return {
                "status": "failed",
                "reason": "Source record not found",
            }

        target_row = connection.execute(
            f'SELECT id FROM "{target_table}" WHERE id = ?',
            (target_id,),
        ).fetchone()

        if target_row is None:
            return {
                "status": "failed",
                "reason": "Target record not found",
            }

        if source_row[0] != target_id:
            return {
                "status": "failed",
                "reason": "Records are not connected by this relationship",
            }

        return {
            "status": "passed",
            "reason": "Relationship matches",
        }

    except sqlite3.Error:
        return {
            "status": "failed",
            "reason": "Could not check the database relationship",
        }

    finally:
        if connection is not None:
            connection.close()

def find_ticket_reference(ticket):
    """Get an invoice/order reference from the ticket or its text."""
    reference = ticket.get("referenced_doc")

    if isinstance(reference, str) and reference.strip():
        reference = reference.strip()

        if re.fullmatch(r"(INV|SO)-[0-9]+", reference):
            return reference

        raise ValueError("Unsupported ticket reference")

    text = (
        str(ticket.get("subject", ""))
        + " "
        + str(ticket.get("body", ""))
    )

    references = set(re.findall(r"\b(?:INV|SO)-[0-9]+\b", text))

    if len(references) != 1:
        raise ValueError("Ticket needs one clear invoice or order reference")

    return references.pop()

def get_related_records(connection, ticket):
    """Find the order, customer, and price list for this ticket."""
    reference = find_ticket_reference(ticket)
    invoice = None

    if reference.startswith("INV-"):
        invoice = connection.execute(
            "SELECT * FROM invoices WHERE id = ?",
            (reference,),
        ).fetchone()

        if invoice is None:
            raise ValueError("Ticket invoice not found")

        order_id = invoice["so_id"]
    else:
        order_id = reference

    order = connection.execute(
        "SELECT * FROM sales_orders WHERE id = ?",
        (order_id,),
    ).fetchone()

    if order is None:
        raise ValueError("Ticket order not found")

    customer = connection.execute(
        "SELECT * FROM customers WHERE id = ?",
        (order["customer_id"],),
    ).fetchone()

    if customer is None:
        raise ValueError("Order customer not found")

    if invoice is not None:
        if invoice["customer_id"] != customer["id"]:
            raise ValueError("Invoice and order customers do not match")

    price_list = connection.execute(
        "SELECT * FROM price_lists WHERE id = ?",
        (order["price_list_id"],),
    ).fetchone()

    return invoice, order, customer, price_list

def has_citation(evidence, table, record, *fields):
    """Check that the agent cited every required field for this record."""
    if record is None:
        return False

    for field in fields:
        found = any(
            ENTITY_TABLE.get(normalize_entity(item["entity"])) == table
            and item["id"] == record["id"]
            and item["field"] == field
            for item in evidence
        )

        if not found:
            return False

    return True

def check_issue(connection, ticket, diagnosis):
    """Check record relevance, issue support, action, and target."""
    invoice, order, customer, price_list = get_related_records(
        connection, ticket
    )

    evidence = diagnosis["evidence"]
    code = diagnosis["root_cause_code"]
    errors = []

    # Check that this transaction is blocked for the claimed type of problem.
    expected_blockers = {
        "CUSTOMER_ON_HOLD": (
            order, "block_reason", "APPROVAL_CHECK_FAILED"
        ),
        "MISSING_TAX_ID": (
            invoice, "blocked_reason", "POSTING_VALIDATION_FAILED"
        ),
        "STALE_PRICE_LIST": (
            order, "block_reason", "PRICING_CHECK_FAILED"
        ),
        "INCOMPLETE_DELIVERY_BLOCKS_INVOICE": (
            invoice, "blocked_reason", "DELIVERY_CHECK_FAILED"
        ),
        "CREDIT_LIMIT_STALE_SYNC": (
            invoice, "blocked_reason", "CREDIT_CHECK_FAILED"
        ),
        "PRICE_SYNC_JOB_FAILED": (
            order, "block_reason", "PRICING_CHECK_FAILED"
        ),
    }

    blocker = expected_blockers.get(code)

    if blocker is not None:
        record, field, expected_reason = blocker

        if record is None:
            errors.append(
                "Required transaction record is unavailable"
            )
        else:
            if record["status"] != "BLOCKED":
                errors.append(
                    "Transaction is not marked BLOCKED"
                )

            if record[field] != expected_reason:
                errors.append(
                    f"Claimed issue expects {expected_reason}, "
                    f"but transaction records {record[field]}"
                )

    # Use the ticket date because this project contains historical data.
    ticket_date = date.fromisoformat(ticket["created_at"][:10])

    deliveries = connection.execute(
        "SELECT * FROM deliveries WHERE so_id = ?",
        (order["id"],),
    ).fetchall()

    payments = connection.execute(
        "SELECT * FROM payments WHERE customer_id = ?",
        (customer["id"],),
    ).fetchall()

    lines = connection.execute(
        "SELECT * FROM sales_order_lines WHERE so_id = ?",
        (order["id"],),
    ).fetchall()

    ledger = connection.execute(
        "SELECT id FROM stock_ledger WHERE ref_doc = ?",
        (order["id"],),
    ).fetchall()

    job_name = {
        "CREDIT_LIMIT_STALE_SYNC": "credit_sync",
        "PRICE_SYNC_JOB_FAILED": "price_sync",
    }.get(code)

    # Policy: consider sync runs within nine days before the ticket,
    # including the ticket date. This is not a database relationship.
    jobs = connection.execute(
        """
        SELECT * FROM job_runs
        WHERE job_name = ?
          AND started_at >= ?
          AND started_at < ?
        """,
        (
            job_name,
            (ticket_date - timedelta(days=9)).isoformat(),
            (ticket_date + timedelta(days=1)).isoformat(),
        ),
    ).fetchall()

    # These are the records that belong to this ticket's transaction.
    related = {
        "invoices": {invoice["id"]} if invoice is not None else set(),
        "sales_orders": {order["id"]},
        "customers": {customer["id"]},
        "price_lists": (
            {price_list["id"]} if price_list is not None else set()
        ),
        "deliveries": {row["id"] for row in deliveries},
        "payments": {row["id"] for row in payments},
        "sales_order_lines": {row["id"] for row in lines},
        "items": {row["item_id"] for row in lines},
        "stock_ledger": {row["id"] for row in ledger},
        "job_runs": {row["id"] for row in jobs},
    }

    for item in evidence:
        table = ENTITY_TABLE.get(normalize_entity(item["entity"]))

        if item["id"] not in related.get(table, set()):
            errors.append(
                f"Unrelated evidence: {item['entity']} {item['id']}"
            )

    # Small helper so the issue rules are easier to read.
    def cited(table, record, *fields):
        return has_citation(evidence, table, record, *fields)

    supported = False
    targets = set()

    if code == "CUSTOMER_ON_HOLD":
        supported = (
            customer["on_hold"] == 1
            and cited("customers", customer, "on_hold")
        )
        targets = {customer["id"]}

    elif code == "MISSING_TAX_ID":
        tax_id = customer["tax_id"]

        supported = (
            (tax_id is None or not tax_id.strip())
            and cited("customers", customer, "tax_id")
        )
        targets = {customer["id"]}

    elif code == "STALE_PRICE_LIST":
        # The order must use an expired price list.
        supported = (
            price_list is not None
            and date.fromisoformat(price_list["valid_to"]) < ticket_date
            and cited("sales_orders", order, "price_list_id")
            and cited("price_lists", price_list, "valid_to")
        )
        targets = {order["id"]}

    elif code == "PRICE_SYNC_JOB_FAILED":
        # Failed price refresh does not require an expired price list.
        # The failed-job check below supplies the remaining condition.
        supported = (
            price_list is not None
            and cited("sales_orders", order, "price_list_id")
        )

    elif code == "INCOMPLETE_DELIVERY_BLOCKS_INVOICE":
        partial = [
            row for row in deliveries
            if row["shipped_qty"] < row["ordered_qty"]
            and cited("deliveries", row, "shipped_qty", "ordered_qty")
        ]

        # The generator creates one delivery per order.
        # Multiple deliveries need an agreed aggregation rule.
        supported = len(deliveries) == 1 and bool(partial)
        targets = {row["id"] for row in partial}

    elif code == "CREDIT_LIMIT_STALE_SYNC":
        unapplied = [
            row for row in payments
            if row["applied_to_limit"] == 0
            and row["amount"] > 0
            and date.fromisoformat(row["received_at"][:10]) <= ticket_date
            and cited("payments", row, "applied_to_limit")
        ]

        supported = (
            customer["credit_used"] >= customer["credit_limit"]
            and cited("customers", customer, "credit_used", "credit_limit")
            and bool(unapplied)
        )

    # Both sync issues also require a cited failed run of the right job.
    if job_name is not None:
        failed_jobs = [
            row for row in jobs
            if row["status"] == "FAILED"
            and cited("job_runs", row, "status")
        ]

        supported = supported and bool(failed_jobs)
        targets = {row["id"] for row in failed_jobs} | {job_name}

    if not supported:
        errors.append("Required evidence does not support this issue")

    fix = diagnosis["proposed_fix"]

    if fix["action"] != EXPECTED_ACTION.get(code):
        errors.append("Proposed action does not match the issue")

    if fix["target"] not in targets:
        errors.append("Proposed target is not the relevant record or job")

    return errors

def validate_output(diagnosis):
    """Check required fields and types before reading the database."""
    errors = []
    if not isinstance(diagnosis, dict):
        return ["Diagnosis must be a dictionary"]

    required = (
        "root_cause_code", "root_cause_explanation", "evidence",
        "proposed_fix", "confidence", "escalate",
    )
    for field in required:
        if field not in diagnosis:
            errors.append(f"Missing field: {field}")
    if errors:
        return errors

    code = diagnosis["root_cause_code"]
    if not isinstance(code, str) or code not in codes():
        errors.append("Invalid root cause code")

    explanation = diagnosis["root_cause_explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        errors.append("Explanation must be a nonempty string")

    confidence = diagnosis["confidence"]
    if type(confidence) not in (int, float):
        errors.append("Confidence must be a number")
    elif not 0 <= confidence <= 1 or not math.isfinite(confidence):
        errors.append("Confidence must be finite and between 0 and 1")

    if type(diagnosis["escalate"]) is not bool:
        errors.append("Escalate must be true or false")

    fix = diagnosis["proposed_fix"]
    if not isinstance(fix, dict):
        errors.append("Proposed fix must be a dictionary")
    else:
        action = fix.get("action")
        if not isinstance(action, str) or action not in ACTIONS:
            errors.append("Invalid fix action")
        target = fix.get("target")
        if not isinstance(target, str) or not target.strip():
            errors.append("Fix target must be a nonempty string")

    evidence = diagnosis["evidence"]
    if not isinstance(evidence, list):
        errors.append("Evidence must be a list")
    else:
        if code != "UNKNOWN" and not evidence:
            errors.append("A known issue needs evidence")
        for index, item in enumerate(evidence, start=1):
            if not isinstance(item, dict):
                errors.append(f"Evidence {index} must be a dictionary")
                continue
            for field in ("entity", "id", "field"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(
                        f"Evidence {index}: {field} must be a nonempty string"
                    )
            # Empty observed text can match a real empty database field.
            # Numeric/null normalization is handled separately, not here.
            if not isinstance(item.get("observed"), str):
                errors.append(f"Evidence {index}: observed must be a string")
    return errors


def verify_diagnosis(ticket, diagnosis, database):
    """Run citation and issue checks, returning one verification result."""
    errors = validate_output(diagnosis)
    if errors:
        return {"status": "failed", "errors": errors}

    if diagnosis.get("root_cause_code") == "UNKNOWN":
        return {
            "status": "insufficient",
            "errors": ["Unknown issue requires review"],
        }

    citations = verify_evidence(diagnosis, database)

    if not citations["citations_valid"]:
        return {
            "status": "failed",
            "errors": citations["errors"],
        }

    connection = None

    try:
        connection = database.connect()
        errors = check_issue(connection, ticket, diagnosis)

        return {
            "status": "failed" if errors else "passed",
            "errors": errors,
        }

    except (sqlite3.Error, ValueError, TypeError, KeyError) as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }

    finally:
        if connection is not None:
            connection.close()
