# Tool-using agent usage

`run_agent_loop.py` runs the bounded ERP investigation controller. It uses the
Groq or Voyager tool client, read-only `ERPDatabase`, allowlisted `ToolDispatcher`,
and final `verify_diagnosis` evidence check.

Configure `GROQ_API_KEY` in a private `.env`, build a synthetic database, then
run a ticket:

```bash
python -m erpsim.build_world --seed 42 --out data/erp.db
python run_agent_loop.py --ticket TKT-1005 --score
```

The runner also accepts an exported ticket file or every database ticket:

```bash
python run_agent_loop.py --input examples/test1.txt
python run_agent_loop.py --all --limit 5 --score --out outputs/agent.jsonl
python3 run_agent_loop.py --ticket TKT-1011 --provider voyager --score
python3 run_agent_loop.py --ticket TKT-1011 --provider voyager --json
```

## Controller policy

- The agent accepts one tool call per model response and at most eight logical
  tool attempts by default. `--max-tool-calls` selects a limit from one to eight.
- All tool calls go through the dispatcher allowlist. The agent cannot issue
  arbitrary SQL, write database records, or call `fault_log`.
- Exact repeated requests use a per-ticket cache. The first duplicate receives
  the cached result; a second duplicate terminates with escalation.
- A `database_error` or `tool_error` may be retried once if the model requests
  the same lookup again. Invalid input, invalid tools, not-found results, and
  successful lookups are not re-executed.
- The model can call `submit_diagnosis` to finish with the exact final-answer
  fields. This control action is not a database call and does not consume a lookup.
- A plain-text tool request can trigger one protocol correction; it is never
  parsed into an executable tool call. The normal dispatcher still validates
  all native database tool calls.
- The final JSON is checked for exact required fields, types, finite confidence,
  and individual citations. Numeric/null citation values are normalized to the
  verifier's text convention without guessing or splitting combined fields.
- Every cited field must match an observation retrieved in the current ticket,
  then pass `verify_diagnosis` for database values, relationships, cause, and action.
  Failed output/evidence checks trigger at most one final correction, with database
  tools disabled. Failed corrections escalate. Audit history preserves both answers.
- System updates are appended to the initial system message, not inserted between
  tool results, because Voyager's selected backend requires that ordering.

The default output is a readable terminal summary with live lookup progress and
separate diagnosis, verification, recommendation, review decision, and metrics.
`--no-color` disables colors. `--json` prints full JSONL records with no UI text.
`--out` saves full records regardless of display mode.

The full result includes the diagnosis at top level plus `verification`,
`tool_trace`, `rejected_tool_calls`, tool/model counts, timing, and a
`termination_reason`. `--score` reads `fault_log` separately after the agent
finishes; it is never exposed through agent tools. Reported full resolution requires
the legacy score checks plus passed verification, valid output, and no escalation.
The model's recommendation before escalation and the validation history remain in
the audit record. JSON-mode summary lines carry `type: evaluation_summary`.

## Tests

Run deterministic controller tests without a model key:

```bash
python -m unittest discover -s tests -v
```

The tests use scripted model responses and a temporary synthetic database. They
cover a normal tool turn, budget exhaustion, duplicate-call handling, malformed
parallel calls, and invalid-ticket fallback.
