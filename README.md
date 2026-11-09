# Order-to-Cash Ticket Diagnosis Agent

Diagnose why an ERP transaction is stuck, given only a vague support ticket.

Every ticket is generated from a fault deliberately injected into a synthetic
ERP, so the correct answer is known before anything runs and scoring is fully
automatic. No real data is used anywhere.

## Baselines

Two baselines are included: **B0** (regex rules, no model, no database) and
**B1** (a single LLM call, ticket text only, no database access). Neither can
query the ERP, and is what a future tool-using agent will
have to improve on.

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
setting and the model consistent when comparing the baseline and future agent.
Higher reasoning settings may require a larger output budget.

`VoyagerToolClient` in `agents/voyager_client.py` also exposes
`chat(messages, tools=None)`, returning the same message, finish reason, token usage,
retry count, and model fields as `GroqToolClient`. The future agent can pass the
approved dispatcher definitions through this interface. This adds a provider,
not the agent loop. Tool choice and call limits remain the controller's responsibility.
Requests use a 120-second timeout and at most two retries for transient errors.
Busy models can queue longer; a timeout is reported rather than treated as a diagnosis.

See the [Voyager API documentation](https://docs.rc.asu.edu/ai/api/) and
[chat and tool-calling reference](https://docs.rc.asu.edu/ai/api/chat/).
Keep `.env` private. No real API key belongs in source control.

## Agent terminal (work in progress)

The investigation loop has been removed so the assigned team member can implement
it. The agent implementation, terminal entry point, and associated agent tests have
been deleted. There is currently no agent command to run.

The terminal layout and colors use standard Python with no Rich dependency.
The optional `requirements-ui.txt` provides a CA bundle for Python HTTPS.
Database tools, the dispatcher, evidence verification, and terminal rendering remain
available for the future loop to use. See
[agent terminal usage](docs/agent_terminal.md) for the previous design.
B0/B1 remain unchanged.

## More baseline commands

```bash
python run_baseline.py --all --baseline b0 --score            # rule baseline
python run_baseline.py --all --baseline b1 --provider gemini --score
python run_baseline.py --ticket TKT-1005 --score
```

## Layout

```
erpsim/build_world.py   world + fault injection + ticket generation
erpsim/schema.sql       12 ERP tables, plus tickets and fault_log
erpsim/taxonomy.py      the closed root-cause vocabulary
agents/llm.py           provider-agnostic client (urllib, no SDKs)
agents/baseline.py      B0 keyword rules, B1 single LLM call
eval/score.py           grading against fault_log
run_baseline.py         CLI
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
