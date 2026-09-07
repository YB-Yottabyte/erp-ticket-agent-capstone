import argparse
import os
import random
import sqlite3
from datetime import date, timedelta

from erpsim.taxonomy import TAXONOMY

TODAY = date(2026, 9, 5)

FIRST = ["Northwind", "Blue Ridge", "Corvus", "Meridian", "Halcyon", "Ardent",
         "Kestrel", "Lakeshore", "Ironwood", "Silverbrook", "Tessellate", "Vantage",
         "Windrow", "Pinnacle", "Cobalt", "Dovetail", "Everline", "Fairmount"]
LAST = ["Traders", "Industrial", "Supply Co", "Logistics", "Distribution",
        "Partners", "Manufacturing", "Wholesale"]
PRODUCTS = ["bearing", "gasket", "valve", "coupling", "flange", "actuator",
            "bracket", "seal kit", "drive belt", "sensor module"]
JOBS = ["credit_sync", "price_sync", "ledger_post"]


def d(offset):
    return (TODAY + timedelta(days=offset)).isoformat()

class World:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.customers, self.items, self.price_lists = [], [], []
        self.pl_items, self.stock, self.ledger = [], [], []
        self.orders, self.lines, self.deliveries = [], [], []
        self.invoices, self.payments, self.jobs = [], [], []
        self.tickets, self.faults = [], []
        self._n = 0

    def nid(self, prefix, width=4):
        self._n += 1
        return f"{prefix}-{self._n:0{width}d}"

    def build_base(self):
        rng = self.rng
        for i in range(40):
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            limit = rng.choice([25000, 50000, 75000, 100000])
            self.customers.append({
                "id": f"C-{100 + i}", "name": name,
                "country": rng.choice(["US", "DE", "IN", "UK"]),
                "currency": "USD", "tax_id": f"TX{rng.randint(10**7, 10**8 - 1)}",
                "payment_terms": rng.choice(["NET30", "NET45", "NET60"]),
                "credit_limit": limit,
                "credit_used": round(limit * rng.uniform(0.05, 0.55), 2),
                "on_hold": 0, "hold_reason": None,
            })

        for i in range(60):
            self.items.append({
                "id": f"SKU-{2000 + i}",
                "name": f"{rng.choice(PRODUCTS)} {rng.choice('ABCDEFGH')}{rng.randint(10, 99)}",
                "uom": rng.choice(["EA", "BOX", "SET"]), "discontinued": 0,
            })

        for i in range(4):
            self.price_lists.append({
                "id": f"PL-{10 + i}", "name": f"standard {2026}-Q{i + 1}",
                "currency": "USD",
                "valid_from": d(-365 + i * 90), "valid_to": d(-275 + i * 90 + 200),
            })
        self.price_lists[-1]["valid_from"] = d(-60)
        self.price_lists[-1]["valid_to"] = d(120)

        for pl in self.price_lists:
            for it in self.items:
                self.pl_items.append({
                    "price_list_id": pl["id"], "item_id": it["id"],
                    "unit_price": round(rng.uniform(12, 480), 2),
                })

        for it in self.items:
            for wh in ("WH-MAIN", "WH-EAST"):
                on_hand = rng.randint(0, 400)
                self.stock.append({
                    "item_id": it["id"], "warehouse": wh,
                    "qty_on_hand": on_hand,
                    "qty_reserved": rng.randint(0, max(1, on_hand // 3)),
                })

        current_pl = self.price_lists[-1]["id"]
        for _ in range(120):
            cust = rng.choice(self.customers)
            so_id = self.nid("SO", 4)
            odate = d(-rng.randint(3, 45))
            self.orders.append({
                "id": so_id, "customer_id": cust["id"], "status": "COMPLETED",
                "order_date": odate, "currency": "USD",
                "price_list_id": current_pl, "block_reason": None,
            })
            total = 0.0
            for _ in range(rng.randint(1, 3)):
                it = rng.choice(self.items)
                qty = rng.randint(1, 40)
                price = round(rng.uniform(12, 480), 2)
                total += qty * price
                self.lines.append({
                    "id": self.nid("SOL", 5), "so_id": so_id,
                    "item_id": it["id"], "qty": qty, "unit_price": price,
                })
                self.ledger.append({
                    "id": self.nid("LG", 5), "item_id": it["id"],
                    "warehouse": "WH-MAIN", "qty_delta": -qty,
                    "ref_doc": so_id, "posted_at": odate,
                })
            self.deliveries.append({
                "id": self.nid("DN", 4), "so_id": so_id, "status": "DELIVERED",
                "ordered_qty": 1, "shipped_qty": 1, "delivered_at": odate,
            })
            self.invoices.append({
                "id": self.nid("INV", 4), "so_id": so_id,
                "customer_id": cust["id"], "status": "SENT",
                "amount": round(total, 2), "blocked_reason": None,
                "created_at": odate,
            })

        for cust in self.customers:
            for _ in range(rng.randint(0, 3)):
                self.payments.append({
                    "id": self.nid("PAY", 4), "customer_id": cust["id"],
                    "amount": round(rng.uniform(500, 18000), 2),
                    "received_at": d(-rng.randint(1, 60)), "applied_to_limit": 1,
                })

        for job in JOBS:
            for day in range(14, 0, -1):
                self.jobs.append({
                    "id": self.nid("JR", 4), "job_name": job,
                    "started_at": d(-day), "status": "SUCCESS", "error": None,
                })

    def free_order(self):
        """An order not already used by an injected fault."""
        used = {f["so_id"] for f in self.faults}
        pool = [o for o in self.orders if o["id"] not in used]
        return self.rng.choice(pool)

    def inv_for(self, so_id):
        return next(i for i in self.invoices if i["so_id"] == so_id)

    def cust(self, cid):
        return next(c for c in self.customers if c["id"] == cid)

    def inject(self, code):
        so = self.free_order()
        cust = self.cust(so["customer_id"])
        inv = self.inv_for(so["id"])
        tier = TAXONOMY[code][0]
        ref, path = None, ""

        if code == "CUSTOMER_ON_HOLD":
            cust["on_hold"] = 1
            cust["hold_reason"] = "compliance review"
            so["status"] = "BLOCKED"
            so["block_reason"] = "APPROVAL_CHECK_FAILED"
            inv["status"] = "DRAFT"
            ref, entity, aid = so["id"], "SalesOrder", so["id"]
            path = f"Customer {cust['id']} on_hold -> SalesOrder {so['id']} blocked"

        elif code == "MISSING_TAX_ID":
            cust["tax_id"] = None
            inv["status"] = "BLOCKED"
            inv["blocked_reason"] = "POSTING_VALIDATION_FAILED"
            ref, entity, aid = inv["id"], "Invoice", inv["id"]
            path = f"Customer {cust['id']} tax_id NULL -> Invoice {inv['id']} blocked"

        elif code == "STALE_PRICE_LIST":
            expired = self.price_lists[0]["id"]
            so["price_list_id"] = expired
            so["status"] = "BLOCKED"
            so["block_reason"] = "PRICING_CHECK_FAILED"
            inv["status"] = "DRAFT"
            ref, entity, aid = so["id"], "SalesOrder", so["id"]
            path = (f"SalesOrder {so['id']} -> PriceList {expired} "
                    f"valid_to {self.price_lists[0]['valid_to']} expired")

        elif code == "INCOMPLETE_DELIVERY_BLOCKS_INVOICE":
            dn = next(x for x in self.deliveries if x["so_id"] == so["id"])
            dn["status"] = "PARTIAL"
            dn["ordered_qty"] = 100
            dn["shipped_qty"] = 60
            dn["delivered_at"] = None
            inv["status"] = "BLOCKED"
            inv["blocked_reason"] = "DELIVERY_CHECK_FAILED"
            ref, entity, aid = inv["id"], "Invoice", inv["id"]
            path = (f"Delivery {dn['id']} shipped 60/100 -> "
                    f"Invoice {inv['id']} blocked")

        elif code == "CREDIT_LIMIT_STALE_SYNC":
            cust["credit_used"] = cust["credit_limit"]
            unapplied = []
            for k in (2, 5):
                p = {"id": self.nid("PAY", 4), "customer_id": cust["id"],
                     "amount": round(self.rng.uniform(4000, 12000), 2),
                     "received_at": d(-k), "applied_to_limit": 0}
                self.payments.append(p)
                unapplied.append(p["id"])
            failed = []
            for jr in self.jobs:
                if jr["job_name"] == "credit_sync" and jr["started_at"] >= d(-7):
                    jr["status"] = "FAILED"
                    jr["error"] = "connection reset applying payment batch"
                    failed.append(jr["id"])
            inv["status"] = "BLOCKED"
            inv["blocked_reason"] = "CREDIT_CHECK_FAILED"
            ref, entity, aid = inv["id"], "Invoice", inv["id"]
            path = (f"JobRun {failed[0]} credit_sync FAILED -> Payment "
                    f"{unapplied[0]} unapplied -> Customer {cust['id']} "
                    f"credit_used at limit -> Invoice {inv['id']} blocked")

        elif code == "PRICE_SYNC_JOB_FAILED":
            expired = self.price_lists[1]["id"]
            so["price_list_id"] = expired
            so["status"] = "BLOCKED"
            so["block_reason"] = "PRICING_CHECK_FAILED"
            inv["status"] = "DRAFT"
            failed = []
            for jr in self.jobs:
                if jr["job_name"] == "price_sync" and jr["started_at"] >= d(-9):
                    jr["status"] = "FAILED"
                    jr["error"] = "price feed unavailable"
                    failed.append(jr["id"])
            ref, entity, aid = so["id"], "SalesOrder", so["id"]
            path = (f"JobRun {failed[0]} price_sync FAILED -> PriceList "
                    f"{expired} never renewed -> SalesOrder {so['id']} blocked")

        else:
            raise ValueError(code)

        tid = f"TKT-{1000 + len(self.tickets) + 1}"
        subject, body = render_ticket(self.rng, code, cust, so, inv, ref)
        if self.rng.random() < 0.5:
            ref = None
        self.tickets.append({
            "id": tid, "subject": subject, "body": body,
            "reported_by": f"ops@{cust['name'].split()[0].lower()}.example",
            "created_at": d(-1), "referenced_doc": ref,
        })
        self.faults.append({
            "ticket_id": tid, "fault_code": code, "tier": tier,
            "affected_entity": entity, "affected_id": aid,
            "causal_path": path, "so_id": so["id"],
        })

SYMPTOM_FAMILY = {
    "CUSTOMER_ON_HOLD": "order_stuck",
    "STALE_PRICE_LIST": "order_stuck",
    "PRICE_SYNC_JOB_FAILED": "order_stuck",
    "MISSING_TAX_ID": "invoice_stuck",
    "INCOMPLETE_DELIVERY_BLOCKS_INVOICE": "invoice_stuck",
    "CREDIT_LIMIT_STALE_SYNC": "invoice_stuck",
}

TEMPLATES = {
    "order_stuck": [
        ("Order for {n} not progressing",
         "We raised an order for {n} earlier this week and it still hasn't "
         "progressed. Customer is asking when it will ship. Nothing looks wrong "
         "on it from where I'm sitting."),
        ("{n} order won't go through",
         "The {n} order won't move past confirmation. Same parts they always "
         "order and the last one went through fine. Can someone take a look?"),
        ("{n} order stuck",
         "Raised an order for {n} and it's just sitting there. Warehouse says "
         "nothing has come through to them yet. No idea what's different this time."),
        ("Cannot get {n} order through",
         "Can't get the {n} order to go through - it keeps getting kicked back "
         "and I don't know what it's unhappy about. Customer is chasing."),
    ],
    "invoice_stuck": [
        ("Invoice for {n} still not sent",
         "We closed out the {n} order over a week ago and delivery went out "
         "fine, but the invoice still hasn't gone to the customer. They're "
         "chasing us for it. Their account looked fine last time I checked."),
        ("{n} invoice held up",
         "The {n} invoice is being held and I can't work out why. Everything "
         "upstream looks complete to me. Please advise."),
        ("No invoice raised for {n}",
         "No invoice has gone to {n} yet and accounts are asking about it. "
         "Nothing on the order side looks wrong."),
        ("{n} invoice won't go out",
         "Trying to get the {n} invoice out and it won't go. Order itself looks "
         "complete. Customer has been waiting a while now."),
    ],
}


def render_ticket(rng, code, cust, so, inv, ref):
    n = cust["name"]
    subject, body = rng.choice(TEMPLATES[SYMPTOM_FAMILY[code]])
    subject, body = subject.format(n=n), body.format(n=n)
    if ref:
        body += f" Reference {ref} if that helps."
    return subject, body

def write(world, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if os.path.exists(out):
        os.remove(out)
    con = sqlite3.connect(out)
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "schema.sql"), encoding="utf-8") as fh:
        con.executescript(fh.read())

    def ins(table, rows, cols):
        if not rows:
            return
        q = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        con.executemany(q, [[r[c] for c in cols] for r in rows])

    ins("customers", world.customers, ["id", "name", "country", "currency", "tax_id",
                                       "payment_terms", "credit_limit", "credit_used",
                                       "on_hold", "hold_reason"])
    ins("items", world.items, ["id", "name", "uom", "discontinued"])
    ins("price_lists", world.price_lists, ["id", "name", "currency", "valid_from", "valid_to"])
    ins("price_list_items", world.pl_items, ["price_list_id", "item_id", "unit_price"])
    ins("stock", world.stock, ["item_id", "warehouse", "qty_on_hand", "qty_reserved"])
    ins("stock_ledger", world.ledger, ["id", "item_id", "warehouse", "qty_delta",
                                       "ref_doc", "posted_at"])
    ins("sales_orders", world.orders, ["id", "customer_id", "status", "order_date",
                                       "currency", "price_list_id", "block_reason"])
    ins("sales_order_lines", world.lines, ["id", "so_id", "item_id", "qty", "unit_price"])
    ins("deliveries", world.deliveries, ["id", "so_id", "status", "ordered_qty",
                                         "shipped_qty", "delivered_at"])
    ins("invoices", world.invoices, ["id", "so_id", "customer_id", "status", "amount",
                                     "blocked_reason", "created_at"])
    ins("payments", world.payments, ["id", "customer_id", "amount", "received_at",
                                     "applied_to_limit"])
    ins("job_runs", world.jobs, ["id", "job_name", "started_at", "status", "error"])
    ins("tickets", world.tickets, ["id", "subject", "body", "reported_by",
                                   "created_at", "referenced_doc"])
    ins("fault_log", world.faults, ["ticket_id", "fault_code", "tier",
                                    "affected_entity", "affected_id", "causal_path"])
    con.commit()
    con.close()


def build(seed=42, out="data/erp.db", per_code=3):
    w = World(seed)
    w.build_base()
    codes = [c for c in TAXONOMY if c != "UNKNOWN"]
    for _ in range(per_code):
        for code in codes:
            w.inject(code)
    write(w, out)
    return w


DEFAULT_EXAMPLE = "TKT-1011"


def export_example(world, path="examples/test1.txt", ticket_id=DEFAULT_EXAMPLE,
                   code="CREDIT_LIMIT_STALE_SYNC"):
    fault = next((f for f in world.faults if f["ticket_id"] == ticket_id), None)
    if fault is None:
        fault = next(f for f in world.faults if f["fault_code"] == code)
    t = next(x for x in world.tickets if x["id"] == fault["ticket_id"])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"TICKET: {t['id']}\n")
        fh.write(f"FROM: {t['reported_by']}\n")
        fh.write(f"CREATED: {t['created_at']}\n")
        fh.write(f"SUBJECT: {t['subject']}\n\n")
        fh.write(t["body"] + "\n")
    return path, fault


def main():
    ap = argparse.ArgumentParser(description="Build the synthetic ERP world.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/erp.db")
    ap.add_argument("--per-code", type=int, default=3,
                    help="tickets generated per fault code")
    ap.add_argument("--example", default=DEFAULT_EXAMPLE,
                    help="ticket id to export as examples/test1.txt")
    a = ap.parse_args()
    w = build(a.seed, a.out, a.per_code)
    print(f"built {a.out}  seed={a.seed}")
    print(f"  customers={len(w.customers)} items={len(w.items)} "
          f"orders={len(w.orders)} invoices={len(w.invoices)}")
    print(f"  ledger={len(w.ledger)} payments={len(w.payments)} jobs={len(w.jobs)}")
    print(f"  tickets={len(w.tickets)}  faults logged={len(w.faults)}")
    path, fault = export_example(w, ticket_id=a.example)
    print(f"  wrote {path}  ({fault['ticket_id']}, {fault['fault_code']}, "
          f"tier {fault['tier']})")


if __name__ == "__main__":
    main()
