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
agents/agent_loop.py    bounded tool-using investigation controller
agents/groq_client.py   Groq chat/tool-calling client
tools/dispatcher.py     approved tool allowlist and argument validation
eval/score.py           grading against fault_log
run_baseline.py         CLI
run_agent_loop.py       tool-using agent CLI
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
