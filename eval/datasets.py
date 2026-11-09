import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from erpsim.build_world import World, build, d, write
from erpsim.taxonomy import TAXONOMY


class DatasetBuilder:
    # Keep generated evaluation data separate from the user's working ERP database.
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.cases = []

    # Give development and held-out records disjoint IDs without changing their facts.
    def namespace(self, world, offset):
        groups = [value for value in vars(world).values() if isinstance(value, list)]
        mapping = {}
        for group in groups:
            for record in group:
                record_id = record.get("id")
                match = re.fullmatch(r"([A-Z]+-)(\d+)", record_id or "")
                if match:
                    mapping[record_id] = match[1] + str(offset + int(match[2]))
        for group in groups:
            for record in group:
                for key, value in record.items():
                    if isinstance(value, str):
                        record[key] = re.sub(
                            r"\b[A-Z]+-\d+\b", lambda m: mapping.get(m[0], m[0]), value
                        )

    # Check actual links because the starter schema does not declare foreign keys.
    def check_database(self, path, expected_faults):
        links = [
            ("sales_orders", "customer_id", "customers"),
            ("sales_orders", "price_list_id", "price_lists"),
            ("invoices", "so_id", "sales_orders"),
            ("invoices", "customer_id", "customers"),
            ("deliveries", "so_id", "sales_orders"),
            ("payments", "customer_id", "customers"),
            ("sales_order_lines", "so_id", "sales_orders"),
            ("sales_order_lines", "item_id", "items"),
            ("price_list_items", "price_list_id", "price_lists"),
            ("price_list_items", "item_id", "items"),
            ("stock", "item_id", "items"),
            ("stock_ledger", "item_id", "items"),
        ]
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert (
                connection.execute("SELECT COUNT(*) FROM fault_log").fetchone()[0]
                == expected_faults
            )
            for table, field, target in links:
                missing = connection.execute(
                    f"SELECT COUNT(*) FROM {table} s LEFT JOIN {target} t ON s.{field}=t.id WHERE s.{field} IS NOT NULL AND t.id IS NULL"
                ).fetchone()[0]
                if missing:
                    raise ValueError(f"Broken references in {table}.{field}")
            for subject, body in connection.execute("SELECT subject,body FROM tickets"):
                if any(
                    code in subject + body for code in TAXONOMY if code != "UNKNOWN"
                ):
                    raise ValueError("Ticket exposes its fault label")

    # Register public inputs and private evaluation labels without sending labels to models.
    def register(self, world, name, split, seed, path):
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        for ticket, fault in zip(world.tickets, world.faults):
            self.cases.append(
                {
                    "case_id": f"{name}-{split}-{seed}-{ticket['id']}",
                    "dataset": name,
                    "split": split,
                    "seed": seed,
                    "db_path": str(path.relative_to(self.output)),
                    "ticket_id": ticket["id"],
                    "expected_code": fault["fault_code"],
                    "tier": fault["tier"],
                    "database_sha256": fingerprint,
                }
            )

    # Create an unsupported blocker while keeping the six supported causes absent.
    def unsupported_world(self, seed, index):
        world = World(seed)
        world.build_base()
        order = world.orders[0]
        invoice = world.inv_for(order["id"])
        order["price_list_id"] = world.price_lists[-1]["id"]
        blocker = [
            "EXPORT_LICENSE_REQUIRED",
            "DUPLICATE_DOCUMENT_REVIEW",
            "BANK_CONFIRMATION_PENDING",
        ][index % 3]
        order.update(status="BLOCKED", block_reason=blocker)
        invoice.update(status="BLOCKED", blocked_reason=blocker)
        ticket_id = "TKT-1001"
        world.tickets = [
            {
                "id": ticket_id,
                "subject": "Invoice cannot proceed",
                "body": f"Please investigate invoice {invoice['id']}. The transaction is blocked and the cause is unclear.",
                "reported_by": "ops@synthetic.example",
                "created_at": d(-1),
                "referenced_doc": invoice["id"],
            }
        ]
        world.faults = [
            {
                "ticket_id": ticket_id,
                "fault_code": "UNKNOWN",
                "tier": 0,
                "affected_entity": "Invoice",
                "affected_id": invoice["id"],
                "so_id": order["id"],
                "causal_path": f"Invoice {invoice['id']} has unsupported blocker {blocker}",
            }
        ]
        return world

    # Freeze the promised 60/30/12 split and a separate 18-ticket mixed-fault reference set.
    def generate(self):
        self.output.mkdir(parents=True, exist_ok=False)
        codes = [code for code in TAXONOMY if code != "UNKNOWN"]
        for split, count, seed_start, offset in [
            ("dev", 10, 10000, 1000000),
            ("test", 5, 20000, 10000000),
        ]:
            for index in range(count * len(codes)):
                code = codes[index % len(codes)]
                seed = seed_start + index
                world = World(seed)
                world.build_base()
                world.inject(code)
                self.namespace(world, offset + index * 10000)
                path = self.output / split / f"case-{index:03}.db"
                write(world, str(path))
                self.check_database(path, 1)
                self.register(world, "isolated", split, seed, path)
        for index in range(12):
            world = self.unsupported_world(30000 + index, index)
            self.namespace(world, 20000000 + index * 10000)
            path = self.output / "unsupported" / f"case-{index:03}.db"
            write(world, str(path))
            self.check_database(path, 1)
            self.register(world, "unsupported", "test", 30000 + index, path)
        path = self.output / "mixed" / "seed-42.db"
        world = build(seed=42, out=str(path))
        self.check_database(path, 18)
        self.register(world, "mixed", "reference", 42, path)
        manifest = {
            "generator": "erpsim.build_world.World",
            "license": "MIT",
            "synthetic_only": True,
            "counts": dict(
                Counter(case["dataset"] + "/" + case["split"] for case in self.cases)
            ),
            "cases": self.cases,
        }
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = DatasetBuilder(args.out).generate()
    print(json.dumps(manifest["counts"], indent=2))
