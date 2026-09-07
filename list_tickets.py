import sqlite3
import sys

con = sqlite3.connect("data/erp.db")
con.row_factory = sqlite3.Row

tier = int(sys.argv[1]) if len(sys.argv) > 1 else None
q = ("SELECT t.id, f.tier, f.fault_code, f.causal_path, t.subject, t.body "
     "FROM tickets t JOIN fault_log f ON f.ticket_id = t.id ")
if tier:
    q += f"WHERE f.tier = {tier} "
q += "ORDER BY f.tier, t.id"

for r in con.execute(q):
    print(f"[tier {r['tier']}] {r['id']}  ->  {r['fault_code']}")
    print(f"  subject: {r['subject']}")
    print(f"  body   : {r['body']}")
    print(f"  chain  : {r['causal_path']}")
    print()
