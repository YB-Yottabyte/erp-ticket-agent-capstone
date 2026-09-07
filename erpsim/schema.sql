-- Order-to-cash slice of a synthetic ERP.
-- Two tables at the bottom are NOT part of the ERP: `tickets` and `fault_log`.
-- `fault_log` is the answer key and must never be exposed through the tool layer.

DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS price_lists;
DROP TABLE IF EXISTS price_list_items;
DROP TABLE IF EXISTS stock;
DROP TABLE IF EXISTS stock_ledger;
DROP TABLE IF EXISTS sales_orders;
DROP TABLE IF EXISTS sales_order_lines;
DROP TABLE IF EXISTS deliveries;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS job_runs;
DROP TABLE IF EXISTS tickets;
DROP TABLE IF EXISTS fault_log;

CREATE TABLE customers (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    country       TEXT NOT NULL,
    currency      TEXT NOT NULL,
    tax_id        TEXT,
    payment_terms TEXT NOT NULL,
    credit_limit  REAL NOT NULL,
    credit_used   REAL NOT NULL,
    on_hold       INTEGER NOT NULL DEFAULT 0,
    hold_reason   TEXT
);

CREATE TABLE items (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    uom          TEXT NOT NULL,
    discontinued INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE price_lists (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    currency    TEXT NOT NULL,
    valid_from  TEXT NOT NULL,
    valid_to    TEXT NOT NULL
);

CREATE TABLE price_list_items (
    price_list_id TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    unit_price    REAL NOT NULL,
    PRIMARY KEY (price_list_id, item_id)
);

CREATE TABLE stock (
    item_id      TEXT NOT NULL,
    warehouse    TEXT NOT NULL,
    qty_on_hand  REAL NOT NULL,
    qty_reserved REAL NOT NULL,
    PRIMARY KEY (item_id, warehouse)
);

CREATE TABLE stock_ledger (
    id        TEXT PRIMARY KEY,
    item_id   TEXT NOT NULL,
    warehouse TEXT NOT NULL,
    qty_delta REAL NOT NULL,
    ref_doc   TEXT,
    posted_at TEXT NOT NULL
);

CREATE TABLE sales_orders (
    id            TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    status        TEXT NOT NULL,
    order_date    TEXT NOT NULL,
    currency      TEXT NOT NULL,
    price_list_id TEXT,
    block_reason  TEXT
);

CREATE TABLE sales_order_lines (
    id         TEXT PRIMARY KEY,
    so_id      TEXT NOT NULL,
    item_id    TEXT NOT NULL,
    qty        REAL NOT NULL,
    unit_price REAL NOT NULL
);

CREATE TABLE deliveries (
    id           TEXT PRIMARY KEY,
    so_id        TEXT NOT NULL,
    status       TEXT NOT NULL,
    ordered_qty  REAL NOT NULL,
    shipped_qty  REAL NOT NULL,
    delivered_at TEXT
);

CREATE TABLE invoices (
    id             TEXT PRIMARY KEY,
    so_id          TEXT NOT NULL,
    customer_id    TEXT NOT NULL,
    status         TEXT NOT NULL,
    amount         REAL NOT NULL,
    blocked_reason TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE payments (
    id                TEXT PRIMARY KEY,
    customer_id       TEXT NOT NULL,
    amount            REAL NOT NULL,
    received_at       TEXT NOT NULL,
    applied_to_limit  INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE job_runs (
    id         TEXT PRIMARY KEY,
    job_name   TEXT NOT NULL,
    started_at TEXT NOT NULL,
    status     TEXT NOT NULL,
    error      TEXT
);

-- Not part of the ERP -------------------------------------------------------

CREATE TABLE tickets (
    id             TEXT PRIMARY KEY,
    subject        TEXT NOT NULL,
    body           TEXT NOT NULL,
    reported_by    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    referenced_doc TEXT
);

-- ANSWER KEY. Never expose through the tool layer.
CREATE TABLE fault_log (
    ticket_id       TEXT PRIMARY KEY,
    fault_code      TEXT NOT NULL,
    tier            INTEGER NOT NULL,
    affected_entity TEXT NOT NULL,
    affected_id     TEXT NOT NULL,
    causal_path     TEXT NOT NULL
);
