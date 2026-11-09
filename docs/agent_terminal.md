# Tool-using agent usage

`run_agent_loop.py` runs the bounded ERP investigation controller. It uses the
existing Groq tool client, read-only `ERPDatabase`, allowlisted `ToolDispatcher`,
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
```

## Controller policy

- The agent accepts one tool call per model response and at most eight logical
  tool attempts by default. `--max-tool-calls` changes that limit.
- All tool calls go through the dispatcher allowlist. The agent cannot issue
  arbitrary SQL, write database records, or call `fault_log`.
- Exact repeated requests use a per-ticket cache. The first duplicate receives
  the cached result; a second duplicate terminates with escalation.
- A `database_error` or `tool_error` may be retried once if the model requests
  the same lookup again. Invalid input, invalid tools, not-found results, and
  successful lookups are not re-executed.
- The final JSON is parsed with the baseline output contract and then checked by
  `verify_diagnosis`. Missing, fabricated, unrelated, or insufficient evidence
  forces `escalate: true` and an `ESCALATE` action.

The result is JSON with the diagnosis at top level plus `verification`,
`tool_trace`, `rejected_tool_calls`, tool/model counts, timing, and a
`termination_reason`. `--score` reads `fault_log` separately after the agent
finishes; it is never exposed through agent tools.

## Tests

Run deterministic controller tests without a model key:

```bash
python -m unittest discover -s tests -v
```

The tests use scripted model responses and a temporary synthetic database. They
cover a normal tool turn, budget exhaustion, duplicate-call handling, malformed
parallel calls, and invalid-ticket fallback.
