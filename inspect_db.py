import sqlite3
import sys
con = sqlite3.connect("data/erp.db")
con.row_factory = sqlite3.Row
def counts():
    print(f"{'TABLE':24} ROWS")
    q = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    for (t,) in con.execute(q):
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"{t:24} {n}")

    print("\nANSWER KEY (fault_log)")
    print(f"  {'ticket':10} {'tier':5} {'fault_code':38} affected")
    for r in con.execute("SELECT * FROM fault_log ORDER BY tier, ticket_id"):
        print(f"  {r['ticket_id']:10} {r['tier']:<5} {r['fault_code']:38} "
              f"{r['affected_entity']} {r['affected_id']}")
def trace(tid):
    f = con.execute("SELECT * FROM fault_log WHERE ticket_id=?", (tid,)).fetchone()
    t = con.execute("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone()
    if not f:
        sys.exit(f"no such ticket: {tid}")

    print(f"TICKET {tid}  (what the analyst sees)")
    print(f"  subject: {t['subject']}")
    print(f"  body   : {t['body']}")
    print(f"\nANSWER KEY  (hidden from the agent)")
    print(f"  fault  : {f['fault_code']}  tier {f['tier']}")
    print(f"  path   : {f['causal_path']}")
    inv = con.execute("SELECT * FROM invoices WHERE id=?", (f["affected_id"],)).fetchone()
    if not inv:
        return
    cust = con.execute("SELECT * FROM customers WHERE id=?",
                       (inv["customer_id"],)).fetchone()
    print(f"\nWHAT THE AGENT WOULD HAVE TO FIND")
    print(f"  1. Invoice {inv['id']:10} status={inv['status']} "
          f"blocked_reason={inv['blocked_reason']}")
    print(f"  2. Customer {cust['id']:9} credit_limit={cust['credit_limit']} "
          f"credit_used={cust['credit_used']} on_hold={cust['on_hold']}")
    pays = con.execute("SELECT * FROM payments WHERE customer_id=? "
                       "ORDER BY received_at DESC LIMIT 4", (cust["id"],)).fetchall()
    for p in pays:
        print(f"  3. Payment {p['id']:10} {p['amount']:>10} on {p['received_at']} "
              f"applied_to_limit={p['applied_to_limit']}")
    jobs = con.execute("SELECT * FROM job_runs WHERE job_name='credit_sync' "
                       "ORDER BY started_at DESC LIMIT 4").fetchall()
    for j in jobs:
        print(f"  4. JobRun  {j['id']:10} {j['started_at']} {j['status']:8} "
              f"{j['error'] or ''}")

if len(sys.argv) > 1:
    trace(sys.argv[1])
else:
    counts()
