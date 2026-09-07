TAXONOMY = {
    "CUSTOMER_ON_HOLD": (
        1, "Customer account is flagged on hold, so orders and invoices are blocked."),
    "MISSING_TAX_ID": (
        1, "Customer has no tax ID on file, so the invoice cannot be posted."),
    "STALE_PRICE_LIST": (
        2, "The order was priced from a price list whose validity window has expired."),
    "INCOMPLETE_DELIVERY_BLOCKS_INVOICE": (
        2, "Only part of the order shipped, so the invoice cannot be released."),
    "CREDIT_LIMIT_STALE_SYNC": (
        3, "Credit limit looks exhausted because a failed sync job left payments unapplied."),
    "PRICE_SYNC_JOB_FAILED": (
        3, "Prices are wrong because the price refresh job has been failing silently."),
    "UNKNOWN": (
        0, "Cause is outside the taxonomy; the ticket should be escalated."),
}

ACTIONS = [
    "RERUN_JOB",
    "UPDATE_FIELD",
    "RELEASE_HOLD",
    "REPRICE_ORDER",
    "COMPLETE_DELIVERY",
    "ESCALATE",
]

EXPECTED_ACTION = {
    "CUSTOMER_ON_HOLD": "RELEASE_HOLD",
    "MISSING_TAX_ID": "UPDATE_FIELD",
    "STALE_PRICE_LIST": "REPRICE_ORDER",
    "INCOMPLETE_DELIVERY_BLOCKS_INVOICE": "COMPLETE_DELIVERY",
    "CREDIT_LIMIT_STALE_SYNC": "RERUN_JOB",
    "PRICE_SYNC_JOB_FAILED": "RERUN_JOB",
    "UNKNOWN": "ESCALATE",
}


def codes():
    return [c for c in TAXONOMY if c != "UNKNOWN"] + ["UNKNOWN"]


def taxonomy_prompt_block():
    lines = []
    for code in codes():
        tier, desc = TAXONOMY[code]
        lines.append(f"- {code}: {desc}")
    return "\n".join(lines)
