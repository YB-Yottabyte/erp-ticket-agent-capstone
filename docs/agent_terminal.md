# Agent cycle and terminal summary

The first version defaults to **one tool call**, followed by a final model reply.
This makes the initial model-tool-model cycle easy to inspect. A budget of up to
eight calls is also supported for sequential investigations. It is not a complete
evaluated diagnosis system yet: factual evidence verification remains a separate
task, and all current diagnoses require human review.

## Run

Configure `GROQ_API_KEY` in your private `.env`. The existing `GROQ_MODEL` setting
is reused. Neither keys nor `.env` belong in version control.

```bash
python3 -B run_agent.py --ticket TKT-1011
python3 -B run_agent.py --input examples/test1.txt --max-tool-calls 8
python3 -B run_agent.py --ticket TKT-1011 --json --out outputs/agent-first-run.json
```

The output directory must already exist, and existing output files are never
overwritten. The JSON mode writes a machine-readable object to stdout with no
terminal UI. Interactive output goes to stderr. The terminal shows actual tool
results in a bordered header, green progress markers, teal record values, evidence
trees, and an aligned final table. `--no-color` disables color without changing
the layout. Narrow terminals stack the summary fields and wrap long text.

Interactive terminals show a small waiting spinner and reveal the completed
explanation at 0.08 seconds per word, with a 0.4-second pause between sentences. This is a display
effect after generation, not token streaming from Groq. It does not change the
recorded agent latency or token usage. Use `--no-animation` to disable both effects.
Animations are disabled automatically for redirected output and JSON mode.

The display follows four numbered sections: records checked, preliminary finding,
evidence cited by the model, and recommended action. It shows the ticket subject,
labels the one-call default as a quick check, and translates tool names and common
ERP status codes into readable labels. Original codes and values remain unchanged
in JSON output. Retrieved facts and unverified model citations stay visibly separate.

The UI uses standard Python and ANSI colors, so Rich is not required. Colors are
enabled only in a real terminal and respect `NO_COLOR` and `TERM=dumb`. Redirected
output has no escape codes but retains the panel and tree layout. The optional
dependency below provides Certifi for Python without configured CA roots:

```bash
python3 -m pip install -r requirements-ui.txt
```

## Classes

- `GroqToolClient`: sends message history and definitions, disables parallel tool
  calls, requests final JSON without tools after the budget, and retries transient
  failures at most twice. Provider error bodies and reasoning are not displayed.
- `ERPAgent`: keeps per-ticket state, counts invalid tool requests against the
  budget, reuses byte-identical repeated requests, and records termination reasons.
- `TerminalUI`: displays progress and the summary separately from agent logic.
- `AgentCLI`: loads public ticket fields and selects terminal or JSON output.

Only one tool request per response is supported. A parallel-call response or a
malformed call envelope ends the investigation with escalation. It is not silently
truncated. A budget of N permits at most N tool attempts and N+1 logical model
calls; transport retries are separately bounded. Repeated calls still consume the
tool budget even when their observations come from cache. No cross-ticket cache
or conversation memory is retained.

The system prompt identifies ticket and database text as untrusted data. The
dispatcher remains the tool allowlist. These measures reduce exposure; they are
not proof of prompt-injection resistance.

## Research limitations

The output includes tool observations and call IDs, cache flags, local tool time,
wall time, model time including retries, reported tokens, and a termination reason.
Missing usage is marked incomplete, not presented as a known zero. Failed provider
requests may have unreported token consumption. Raw model reasoning is not logged.
`dispatcher_calls` counts uncached requests routed to validation and dispatch, not
SQL executions: a rejected argument may never reach a database query. Calls from
unsupported parallel responses or requests after the budget are recorded separately
as `rejected_tool_calls`; they are not executed.
The configured model is recorded, but reproducibility also requires recording the
database seed, code revision, environment, and repeated runs in the later evaluation
harness. Results from scripted clients test control flow, not diagnosis accuracy.

Every diagnosis is currently provisional. The parser checks field types and allowed
codes/actions; it does not establish that cited records prove the cause. The UI
therefore displays **Not verified**, and the controller forces `escalate: true`
pending the evidence-verification implementation. Do not interpret these outputs
as full-resolution results or compare them as final research scores yet.

The existing B0/B1 baseline code is unchanged. No scoring or answer-key accessor is
added to the agent. The `--db` path is read-only for ticket loading and tool queries.

## Offline checks

```bash
python3 -B -m unittest discover -s tests -v
```

Protocol reference: [Groq local tool calling](https://console.groq.com/docs/tool-use/local-tool-calling).

## macOS HTTPS certificates

If a live call fails because Python has no trusted certificate bundle, configure
your Python installation's CA certificates (the Python.org macOS installer includes
an Install Certificates command). Do not disable HTTPS certificate verification.
The client uses Python's trusted roots, falling back to Certifi only if no roots
are configured. The standard `SSL_CERT_FILE` setting can select a trusted bundle.
An initial live check on this development machine required the already-installed
Certifi bundle. With that trusted bundle available, a one-tool Groq cycle
completed: model request, invoice lookup, and final JSON reply. This smoke test
establishes connectivity and protocol handling, not diagnosis correctness.
