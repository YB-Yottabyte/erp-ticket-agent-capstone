import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from time import perf_counter

from agents.agent_loop import ERPAgentLoop
from agents.baseline import build_prompt, parse, run_b0
from agents.evidence import verify_diagnosis, verify_evidence
from agents.groq_client import GroqToolClient
from agents.fixed_workflow import FixedDatabaseBaseline
from agents.llm import LLMError, _load_dotenv, complete
from agents.voyager_client import VoyagerToolClient
from erpsim.taxonomy import ACTIONS, codes
from eval.evaluate import aggregate
from eval.score import fault_for, score_one
from run_agent_loop import load_tickets
from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry


class AuditedDatabase(ERPDatabase):
    # Add an evaluation-only SQL barrier without changing the agent or its observations.
    def __init__(self, path):
        super().__init__(path)
        self.denied_reads = 0
        self.denied_writes = 0

    # Deny answer-key reads and writes on tool/verifier connections, including attempted access.
    def authorize(self, action, first, second, database, trigger):
        if action == sqlite3.SQLITE_READ and first == "fault_log":
            self.denied_reads += 1
            return sqlite3.SQLITE_DENY
        if action in {
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_UPDATE,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_ATTACH,
            sqlite3.SQLITE_DETACH,
            sqlite3.SQLITE_ALTER_TABLE,
        }:
            self.denied_writes += 1
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    # Preserve the normal read-only connection and record forbidden SQL requests.
    def connect(self):
        connection = super().connect()
        connection.set_authorizer(self.authorize)
        return connection


class ExperimentRunner:
    # Freeze settings before any held-out call and save each case as soon as it completes.
    def __init__(
        self, manifest, output, provider="voyager", model="qwen38-27b", workers=2, systems=None
    ):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.provider, self.model, self.workers = provider, model, workers
        self.systems = tuple(systems or ("b0", "b1", "agent"))
        if len(set(self.systems)) != len(self.systems) or any(
            system not in {"b0", "b1", "fixed", "agent"} for system in self.systems
        ):
            raise ValueError("Choose distinct systems from b0, b1, fixed, agent")
        self.code_hash = self.implementation_hash()
        self.settings = {
            "provider": provider,
            "model": model,
            "temperature": 0,
            "reasoning_effort": (
                os.environ.get("VOYAGER_REASONING_EFFORT", "none")
                if provider == "voyager"
                else None
            ),
            "max_tool_calls": 8,
            "workers": workers,
            "systems": list(self.systems),
            "implementation_sha256": self.code_hash,
            "evaluation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
                + Path(__file__).with_name("evaluate.py").read_bytes()
                + Path(__file__).with_name("score.py").read_bytes()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                self.manifest_path.read_bytes()
            ).hexdigest(),
        }

    # Fingerprint the actual model, controller, tools, and verification code used in this run.
    def implementation_hash(self):
        root = Path(__file__).resolve().parents[1]
        digest = hashlib.sha256()
        for folder in ("agents", "tools", "erpsim"):
            for path in sorted((root / folder).glob("*.py")):
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
        return digest.hexdigest()

    # Measure required JSON fields and types separately from factual evidence support.
    @staticmethod
    def valid_format(diagnosis, raw=None):
        if diagnosis.get("malformed") or diagnosis.get("schema_problems"):
            return False
        if raw is not None:
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                return False
        else:
            value = diagnosis
        if not isinstance(value, dict):
            return False
        required = {
            "root_cause_code",
            "root_cause_explanation",
            "evidence",
            "proposed_fix",
            "confidence",
            "escalate",
        }
        if not required.issubset(value):
            return False
        confidence = value["confidence"]
        fix = value["proposed_fix"]
        if (
            not isinstance(value["root_cause_code"], str)
            or value["root_cause_code"] not in codes()
            or not isinstance(value["root_cause_explanation"], str)
            or not value["root_cause_explanation"].strip()
            or type(confidence) not in (int, float)
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or type(value["escalate"]) is not bool
            or not isinstance(fix, dict)
            or fix.get("action") not in ACTIONS
            or not isinstance(fix.get("target"), str)
            or not fix["target"].strip()
            or not isinstance(value["evidence"], list)
        ):
            return False
        return all(
            isinstance(item, dict)
            and all(
                isinstance(item.get(field), str)
                for field in ("entity", "id", "field", "observed")
            )
            for item in value["evidence"]
        )

    # Run exactly the existing B0/B1 or adaptive system, with labels used only afterward.
    def evaluate(self, case, system):
        path = (self.manifest_path.parent / case["db_path"]).resolve()
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != case["database_sha256"]:
            raise ValueError("Evaluation database changed after dataset freeze")
        ticket = load_tickets(str(path), ticket_id=case["ticket_id"])[0]
        database = AuditedDatabase(path)
        raw = None
        error = None
        started = perf_counter()
        try:
            if system == "b0":
                diagnosis, usage = run_b0(ticket)
            elif system == "b1":
                raw, usage = complete(
                    build_prompt(ticket),
                    provider=self.provider,
                    model=self.model,
                    temperature=0,
                )
                diagnosis = parse(raw)
                diagnosis["ticket_id"] = ticket["id"]
            elif system == "fixed":
                diagnosis, usage = FixedDatabaseBaseline(
                    database, provider=self.provider, model=self.model
                ).run(ticket)
                raw = diagnosis.get("raw_response")
            elif system == "agent":
                clients = {"voyager": VoyagerToolClient, "groq": GroqToolClient}
                client = clients[self.provider](model=self.model)
                agent = ERPAgentLoop(
                    client,
                    ToolDispatcher(ToolRegistry(database)),
                    database,
                    max_tool_calls=8,
                )
                diagnosis = agent.run_loop(ticket)
                usage = diagnosis.get("model_usage", {})
                history = diagnosis.get("validation_history", [])
                raw = history[-1].get("raw_response") if history else None
            else:
                raise ValueError("Unknown evaluation system")
        except LLMError as failure:
            error = str(failure)
            usage = {"input_tokens": None, "output_tokens": None}
            diagnosis = {
                "root_cause_code": "UNKNOWN",
                "root_cause_explanation": "Provider request failed",
                "evidence": [],
                "proposed_fix": {"action": "ESCALATE", "target": "human support"},
                "confidence": 0,
                "escalate": True,
                "malformed": True,
                "schema_problems": [error],
            }
        elapsed = perf_counter() - started
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
            fault = fault_for(connection, ticket["id"])
            scored = score_one(connection, diagnosis, fault)
        citation_checks = verify_evidence(diagnosis, database)
        verification = (
            diagnosis.get("verification")
            if system in {"agent", "fixed"}
            else verify_diagnosis(ticket, diagnosis, database)
        )
        verification = verification or {
            "status": "failed",
            "errors": ["No verification result"],
        }
        valid_json = self.valid_format(diagnosis, raw)
        operational = (
            error is None and diagnosis.get("termination_reason")
            not in {"provider_error", "fixed_reference_unavailable"}
        )
        supported = case["expected_code"] != "UNKNOWN"
        if supported:
            resolved = bool(
                scored["resolved"]
                and verification["status"] == "passed"
                and valid_json
                and not diagnosis.get("escalate")
                and operational
            )
        else:
            resolved = bool(
                diagnosis.get("root_cause_code") == "UNKNOWN"
                and diagnosis.get("escalate") is True
                and diagnosis.get("proposed_fix", {}).get("action") == "ESCALATE"
                and valid_json
                and operational
            )
        trace = diagnosis.get("tool_trace", [])
        attempts = diagnosis.get("tool_attempts", 0)
        tokens_known = diagnosis.get("usage_complete", True) and all(
            type(usage.get(k)) is int for k in ("input_tokens", "output_tokens")
        )
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        scored.update(
            system=system,
            case_id=case["case_id"],
            dataset=case["dataset"],
            split=case["split"],
            seed=case["seed"],
            resolved=resolved,
            code_ok=bool(scored["code_ok"] and operational),
            supported=supported,
            valid_json=valid_json,
            schema_problems=diagnosis.get("schema_problems", []),
            malformed=not valid_json,
            latency_seconds=round(elapsed, 6),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            total_tokens=(
                sum(usage[k] for k in ("input_tokens", "output_tokens"))
                if tokens_known
                else None
            ),
            usage_complete=tokens_known,
            tool_calls=attempts,
            dispatcher_calls=sum(not item.get("cache_hit", False) for item in trace),
            successful_tool_calls=sum(
                item.get("observation", {}).get("status") == "ok" for item in trace
            ),
            not_found_calls=sum(
                item.get("observation", {}).get("status") == "not_found"
                for item in trace
            ),
            repeated_calls=sum(bool(item.get("cache_hit")) for item in trace),
            citations_checked=len(citation_checks["checks"]),
            citations_correct=sum(
                check["valid"] for check in citation_checks["checks"]
            ),
            verifier_status=verification["status"],
            termination_reason=diagnosis.get("termination_reason", "baseline_complete"),
            protocol_repairs=diagnosis.get("protocol_repair_attempts", 0),
            final_repairs=diagnosis.get("final_repair_attempts", 0),
            denied_answer_key_reads=database.denied_reads,
            denied_writes=database.denied_writes,
            database_unchanged=before == after,
            provider_error=error or diagnosis.get("provider_errors", []),
            api_cost_usd=0.0 if self.provider == "voyager" or system == "b0" else None,
        )
        return {
            "settings": self.settings,
            "case": case,
            "score": scored,
            "diagnosis": diagnosis,
            "raw_baseline_response": raw if system == "b1" else None,
            "citation_verification": citation_checks,
            "verification": verification,
        }

    # Aggregate both original evaluator measures and the missing Milestone 1 checks.
    @staticmethod
    def summarize(records):
        groups = defaultdict(list)
        for record in records:
            row = record["score"]
            groups[(row["dataset"], row["system"])].append(row)
        output = {}
        for (dataset, system), rows in sorted(groups.items()):
            summary = aggregate(rows)
            total = len(rows)
            cited = sum(row["citations_checked"] for row in rows)
            correct = sum(row["citations_correct"] for row in rows)
            calls = sum(row["tool_calls"] for row in rows)
            summary.update(
                evidence_field_value_accuracy_pct=(
                    round(100 * correct / cited, 2) if cited else None
                ),
                invalid_citation_pct=(
                    round(100 * (cited - correct) / cited, 2) if cited else None
                ),
                citations_checked=cited,
                citations_correct=correct,
                no_evidence_cited_pct=round(
                    100 * sum(row["evidence_cited"] == 0 for row in rows) / total, 2
                ),
                successful_tool_call_pct=(
                    round(
                        100 * sum(row["successful_tool_calls"] for row in rows) / calls,
                        2,
                    )
                    if calls
                    else None
                ),
                total_tool_calls=calls,
                max_tool_calls=max(row["tool_calls"] for row in rows),
                repeated_tool_calls=sum(row["repeated_calls"] for row in rows),
                not_found_calls=sum(row["not_found_calls"] for row in rows),
                total_final_repairs=sum(row["final_repairs"] for row in rows),
                total_protocol_repairs=sum(row["protocol_repairs"] for row in rows),
                correct_escalation_pct=(
                    round(100 * sum(row["resolved"] for row in rows) / total, 2)
                    if dataset == "unsupported"
                    else None
                ),
                provider_failure_cases=sum(bool(row["provider_error"]) for row in rows),
                token_usage_complete_cases=sum(row["usage_complete"] for row in rows),
                unchanged_database_cases=sum(row["database_unchanged"] for row in rows),
                denied_answer_key_reads=sum(
                    row["denied_answer_key_reads"] for row in rows
                ),
                denied_writes=sum(row["denied_writes"] for row in rows),
                api_cost_usd=(
                    sum(row["api_cost_usd"] for row in rows)
                    if all(row["api_cost_usd"] is not None for row in rows)
                    else None
                ),
            )
            output[f"{dataset}/{system}"] = summary
        return output

    # Resume only identical frozen experiments and checkpoint every completed system/case pair.
    def run(self):
        checkpoint = self.output / "results.jsonl"
        records = (
            [json.loads(line) for line in checkpoint.read_text().splitlines()]
            if checkpoint.exists()
            else []
        )
        if any(record["settings"] != self.settings for record in records):
            raise ValueError("Resume settings differ; choose a new output directory")
        completed = {
            (record["case"]["case_id"], record["score"]["system"]) for record in records
        }
        cases = [case for case in self.manifest["cases"] if case["split"] != "dev"]
        tasks = [
            (case, system)
            for case in cases
            for system in self.systems
            if (case["case_id"], system) not in completed
        ]
        print(
            f"Frozen evaluation: {len(cases)} cases x {len(self.systems)} systems; {len(tasks)} tasks remaining",
            flush=True,
        )
        (self.output / "settings.json").write_text(
            json.dumps(self.settings, indent=2) + "\n"
        )
        with checkpoint.open("a", encoding="utf-8") as saved, ThreadPoolExecutor(
            max_workers=self.workers
        ) as pool:
            futures = {
                pool.submit(self.evaluate, case, system): (case, system)
                for case, system in tasks
            }
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                saved.write(json.dumps(record, allow_nan=False) + "\n")
                saved.flush()
                row = record["score"]
                print(
                    f"[{len(records)}/{len(cases)*len(self.systems)}] {row['dataset']} {row['system']} {row['case_id']} cause={row['code_ok']} success={row['resolved']} verify={row['verifier_status']} tools={row['tool_calls']} time={row['latency_seconds']:.1f}s",
                    flush=True,
                )
        if self.implementation_hash() != self.code_hash:
            raise ValueError(
                "Implementation changed during the experiment; results require review"
            )
        summary = self.summarize(records)
        (self.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--provider", choices=["voyager", "groq"], default="voyager")
    parser.add_argument("--model", default="qwen38-27b")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--systems", nargs="+", choices=["b0", "b1", "fixed", "agent"],
                        default=["b0", "b1", "agent"])
    args = parser.parse_args()
    _load_dotenv()
    os.environ["VOYAGER_REASONING_EFFORT"] = "none"
    ExperimentRunner(
        args.manifest, args.out, args.provider, args.model, args.workers, args.systems
    ).run()
