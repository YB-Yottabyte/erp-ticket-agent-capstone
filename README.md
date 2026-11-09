# Order-to-Cash Ticket Diagnosis Agent

Diagnose why an ERP transaction is stuck, given only a vague support ticket.

Every ticket is generated from a fault deliberately injected into a synthetic
ERP, so the correct answer is known before anything runs and scoring is fully
automatic. No real data is used anywhere.

## Baselines

Two baselines are included: **B0** (regex rules, no model, no database) and
**B1** (a single LLM call, ticket text only, no database access). Neither can
query the ERP. The verified tool-using agent below is evaluated against both.

## Requirements

Python 3.9+ (developed and tested on 3.13). **No third-party packages** :
standard library only, so there is nothing to `pip install`.

## Run it

```bash
python -m erpsim.build_world --seed 42 --out data/erp.db
python run_baseline.py --input examples/test1.txt --score
```

That works offline with no API key: the default `--provider mock` is a
deterministic stub that exercises the whole pipeline.

## Run it against a real model

```bash
cp .env.example .env      # then paste in one key
python run_baseline.py --input examples/test1.txt --provider gemini --score
```

Set one of `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENAI_API_KEY`. The
`mock` provider is a pipeline stub, never a system under test, every
measured number must come from a real provider.

## Voyager provider

Copy `.env.example` to `.env` if you do not already have a local `.env`.
Add the following settings to `.env`, replacing the placeholder locally:

```dotenv
VOYAGER_API_KEY=your_voyager_key
VOYAGER_MODEL=qwen38-27b
VOYAGER_REASONING_EFFORT=none
```

Build the database, then run the text-only baseline:

```bash
python3 -m erpsim.build_world --seed 42 --out data/erp.db
python3 run_baseline.py --all --baseline b1 --provider voyager --score
```

Voyager uses `https://openai.rc.asu.edu/v1/chat/completions`. The default reasoning
setting is `none` to leave output tokens available for the JSON answer. Keep this
setting and the model consistent when comparing the baseline and agent.
Higher reasoning settings may require a larger output budget.

`VoyagerToolClient` in `agents/voyager_client.py` also exposes
`chat(messages, tools=None)`, returning the same message, finish reason, token usage,
retry count, and model fields as `GroqToolClient`. The agent passes the
approved dispatcher definitions through this interface. This adds a provider,
not the agent loop. Tool choice and call limits remain the controller's responsibility.
Requests use a 120-second timeout and at most two retries for transient errors.
Busy models can queue longer; a timeout is reported rather than treated as a diagnosis.

See the [Voyager API documentation](https://docs.rc.asu.edu/ai/api/) and
[chat and tool-calling reference](https://docs.rc.asu.edu/ai/api/chat/).
Keep `.env` private. No real API key belongs in source control.

## Tool-using agent

The agent supports Groq or Voyager tool calling with the eight approved, read-only ERP
lookups. It enforces a configurable tool-call budget (eight by default), caches
duplicate requests within a ticket, verifies final evidence, and escalates unsafe
or insufficient diagnoses.

Set `GROQ_API_KEY` in your private `.env`, then run one generated ticket:

```bash
python run_agent_loop.py --ticket TKT-1005 --score
```

To use Voyager instead, configure `VOYAGER_API_KEY` and run:

```bash
python3 run_agent_loop.py --ticket TKT-1011 --provider voyager --score
```

The default terminal view shows live lookup progress, the diagnosis, exact evidence,
verification status, recommended action, and whether human review is required.
For the complete audit record instead, add `--json`. Save full results with `--out`:

```bash
python3 run_agent_loop.py --all --limit 3 --provider voyager --score --out outputs/voyager-agent.jsonl
python3 run_agent_loop.py --ticket TKT-1011 --provider voyager --json
```

The agent offers a typed `submit_diagnosis` action alongside the eight database
tools. It submits an answer without reading or changing data, and does not consume
a database lookup. Every citation must match a record and field actually retrieved
in that ticket and pass the database/relationship verifier. A tool request emitted
as plain-text markup can be corrected once; it is never executed as text. A final
answer with schema or evidence errors can be corrected once using existing facts,
with database tools disabled. Both attempts and their verification results are
retained in the audit record. Failed corrections escalate.

Voyager's `qwen38-27b` supports native tool calling but does not advertise strict
response-schema support. The agent uses the submission function's parameter schema
and local validation rather than requesting an unsupported JSON-schema API mode.
Controller updates remain in the initial system message for model compatibility.
The `--score` full-resolution result also requires successful verification, valid
output, and no escalation. Record-ID validity alone does not prove field/value
accuracy. Small sample runs are smoke tests, not final held-out evaluation results.

Voyager defaults to `qwen38-27b`; `--model` overrides the selected provider's model.
The default provider remains Groq when `--provider` is omitted.

Other useful commands:

```bash
python run_agent_loop.py --input examples/test1.txt
python run_agent_loop.py --all --limit 5 --score --out outputs/agent.jsonl
python run_agent_loop.py --ticket TKT-1005 --max-tool-calls 4
```

The live agent never accesses `fault_log`; `--score` reads it separately after
the investigation. B0/B1 remain unchanged.

## Implemented components

| Area | Files | Responsibility |
| --- | --- | --- |
| Synthetic ERP and taxonomy | `erpsim/schema.sql`, `erpsim/build_world.py`, `erpsim/taxonomy.py` | Defines ERP tables, generates fault-isolated tickets, and defines the six root-cause codes and approved actions. |
| Safe ERP tools | `tools/database.py`, `tools/dispatcher.py` | Opens SQLite read-only, exposes eight allowlisted lookups, validates arguments, rejects arbitrary SQL, and never exposes `fault_log`. |
| Agent controller | `agents/agent_loop.py` | Runs the bounded model-tool loop, maintains per-ticket state, prevents repeated calls, tracks up to eight lookup attempts, validates final output, and makes the final escalation decision. |
| Model clients | `agents/groq_client.py`, `agents/voyager_client.py`, `agents/llm.py` | Calls Groq or ASU Voyager with tool definitions; `llm.py` also supports text-only baseline providers. |
| Diagnosis contract | `agents/diagnosis.py`, `agents/baseline.py` | Defines and validates the structured diagnosis schema: cause, explanation, citations, action, confidence, and escalation. |
| Evidence verification | `agents/evidence.py` | Verifies cited entity/ID/field/value pairs, ticket-record relationships, fault-specific evidence requirements, and action/target correctness. |
| Terminal and CLI | `run_agent_loop.py`, `ui/terminal.py`, `run_baseline.py` | Loads tickets, runs the agent or baselines, renders live lookup progress, prints JSON audit records, and supports optional scoring/output files. |
| Evaluation | `eval/score.py`, `eval/evaluate.py`, `eval/case_generation.py` | Scores predictions after runs complete, computes aggregate metrics, and supports repeatable evaluation/case-generation work. |
| Tests | `tests/test_agent_loop.py`, `tests/test_evidence.py`, `tests/test_evaluate.py` | Exercise loop controls, evidence verification policies, and evaluation metric calculations. |

### Agent workflow

```text
Ticket → validate/reference extraction → model chooses one approved tool
      → dispatcher validates → read-only ERP observation → update state
      → model chooses next step or submits diagnosis → output/evidence validation
      → verified resolution or forced escalation
```

The controller stores conversation messages, a per-ticket tool cache, tool trace,
retry/duplicate counters, usage data, and a termination reason. An exact repeated
tool call uses the cached observation once; a second duplicate safely terminates the
investigation. `database_error` and `tool_error` may receive one bounded retry.

Final answers are checked twice: `DiagnosisValidator` first confirms the strict
output schema and that cited fields were retrieved during the run; then
`verify_diagnosis` confirms database values, transaction relationships, root-cause
support, and the approved action. Any failure forces `ESCALATE`.

## Tests and evaluation

Run deterministic unit tests without an API key:

```bash
python -m unittest discover -s tests -v
```

Run baseline-only evaluation offline:

```bash
python -m eval.evaluate --systems b0 --limit 10
```

Run a comparative live evaluation after configuring the required model keys:

```bash
python -m eval.evaluate --systems b0 b1 agent --provider voyager --agent-provider voyager --limit 10 --out outputs/evaluation.json
```

`eval/score.py` reads the hidden `fault_log` only after a run has produced an
answer. It reports root-cause accuracy, full-resolution rate, evidence validity and
relevance, hallucinated evidence, escalation behavior, latency, tokens, and tool
usage. The agent itself has no tool or prompt access to the answer key.

## More baseline commands

```bash
python run_baseline.py --all --baseline b0 --score            # rule baseline
python run_baseline.py --all --baseline b1 --provider gemini --score
python run_baseline.py --ticket TKT-1005 --score
```

## Layout

```
erpsim/                 schema, taxonomy, world builder, and synthetic tickets
tools/                  read-only database lookups and dispatcher allowlist
agents/                 baselines, agent loop, diagnosis validator, model clients,
                        and evidence verification
ui/terminal.py          readable live agent-run output
eval/                   scorer, evaluation harness, and case-generation helpers
tests/                  deterministic unit tests
run_baseline.py         B0/B1 CLI
run_agent_loop.py       verified tool-using agent CLI
```

`data/erp.db` is regenerated from the seed, not committed.

## The answer key

`fault_log` holds the ground truth. It lives in the same SQLite file but will
have **no accessor in the tool layer**: the scorer reads it directly, the
agent never can.

## Troubleshooting

Free-tier API keys (especially Gemini) may throttle and return `429`/`503`.
`llm.py` retries automatically with backoff, but for quick pipeline testing,
use `--provider mock`: it's deterministic, keyless, and exercises the same
code path without hitting rate limits.
