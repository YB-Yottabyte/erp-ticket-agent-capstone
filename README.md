<h1 align="center">ERP Ticket Agent</h1>

<p align="center">
  <strong>Investigate blocked ERP transactions using read-only database tools and verified evidence.</strong>
</p>

The agent reads a support ticket, chooses which ERP records to check, and returns
an explanation, supporting evidence, and a recommended action. It escalates cases
when it cannot verify the diagnosis. It does not change ERP records.

This guide uses **ASU Voyager with `qwen38-27b`**. All examples use synthetic data.

## Quickstart

### 1. Clone the repository

Requirements: Git, Python **3.9+** (tested on 3.13), and a Voyager API key.
The commands below use macOS or Linux with Bash/Zsh.

```bash
git clone https://github.com/YB-Yottabyte/erp-ticket-agent-capstone.git
cd erp-ticket-agent-capstone
python3 --version
```

If you already cloned the repository, open its folder instead.

### 2. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-ui.txt
```

The controller, SQLite tools, and evaluation code use Python's standard library.
`requirements-ui.txt` installs `certifi`, which provides trusted HTTPS certificates
when the Python installation has no certificate roots.

### 3. Get your ASU Voyager API key

1. Connect to the ASU network or ASU VPN to open the
   [Voyager portal](https://voyager.rc.asu.edu/).
2. Sign in, open your profile, and select **Keys**. The API-key button on the
   **AI LLM** tab also opens this page.
3. Choose **Create API key** and copy the key when it is shown.
4. Check the **AI LLM** tab for models available to your account.

The portal requires the ASU network or VPN. The API itself works on or off campus
without VPN. Run this project on your own computer; the commands do not submit a
Sol job. See the [ASU API access guide](https://docs.rc.asu.edu/ai/api/).

### 4. Configure Voyager

Create `.env` only if you do not already have one:

```bash
test -f .env || cp .env.example .env
```

Open `.env` in your editor and fill in these settings:

```dotenv
VOYAGER_API_KEY=your_voyager_api_key_here
VOYAGER_MODEL=qwen38-27b
VOYAGER_REASONING_EFFORT=none
```

The application loads `.env` automatically. You do not need to run `source .env`.
Keep the real key in `.env`, which Git ignores. Leave keys blank in `.env.example`.

Voyager requests use `https://openai.rc.asu.edu/v1/chat/completions`. The client
uses verified HTTPS, a 120-second timeout, and at most two retries for temporary
provider errors. The default output limit is 4096 tokens per request.

### 5. Generate the synthetic ERP database

```bash
python3 -B -m erpsim.build_world --seed 42 --out data/erp.db
```

This creates the ERP records, 18 support tickets, a hidden scoring table, and
`examples/test1.txt`. Running it again regenerates `data/erp.db` and the example
file. Keep evaluation databases in separate folders when comparing experiments.

### 6. Read the example ticket

```bash
cat examples/test1.txt
```

For seed 42, this is **TKT-1011**, an invoice held up for Everline Supply Co.
The invoice reference **INV-0342** appears in the ticket body.

### 7. Run the agent with Voyager

```bash
python3 -B run_agent_loop.py \
  --ticket TKT-1011 \
  --provider voyager \
  --model qwen38-27b \
  --max-tool-calls 8 \
  --score \
  --out outputs/voyager-demo.jsonl
```

The terminal shows each lookup, the diagnosis, cited evidence, recommended action,
escalation decision, and usage summary. `--out` saves the full JSON result and
tool trace. `--score` reads the hidden answer key **after** the investigation;
the agent's tools cannot access it.

One recorded Voyager run returned:

```text
Records checked
[1/8] Read invoice INV-0342: ok
[2/8] Check customer C-136: ok
[3/8] Check payments C-136: ok
[4/8] Check background jobs credit_sync: ok

Result: Verified diagnosis
Diagnosis code: CREDIT_LIMIT_STALE_SYNC
Recommended action: Review and rerun the background job
Target: credit_sync
Escalation: No

Lookups: 4/8 | Model calls: 5 | Time: 16.25s
Fully resolved: YES
```

This is an observed example, not a guaranteed result. Model output, lookup order,
and timing can vary. “Fully resolved” means a verified diagnosis and recommendation;
the system does not execute the recommended fix.

## Other run commands

Use a ticket text file:

```bash
python3 -B run_agent_loop.py --input examples/test1.txt --provider voyager --score
```

Run a small batch:

```bash
python3 -B run_agent_loop.py --all --limit 3 --provider voyager --score --out outputs/voyager-batch.jsonl
```

Print the complete JSON instead of the readable terminal summary:

```bash
python3 -B run_agent_loop.py --ticket TKT-1011 --provider voyager --json
```

Use `--db PATH` to select another generated database. Run `python3 run_agent_loop.py
--help` for all options. Specify `--provider voyager`: the agent's default provider
is Groq.

## Run the baselines

| System | Database access | Model behavior |
| --- | --- | --- |
| B0 | None | Keyword rules; no model |
| B1 | None | One model call using ticket text |
| B1-DB | Fixed read-only sequence | One model call after retrieval |
| Adaptive agent | Model-selected read-only tools | Chooses the next lookup from earlier observations |

```bash
# B0: no API key needed
python3 -B run_baseline.py --ticket TKT-1011 --baseline b0 --score

# B1: ticket text only
python3 -B run_baseline.py --ticket TKT-1011 --baseline b1 --provider voyager --model qwen38-27b --score

# B1-DB: fixed database lookups, then one model call
python3 -B run_baseline.py --ticket TKT-1011 --baseline b1-db --provider voyager --model qwen38-27b --score
```

B1-DB checks invoice, order, customer, deliveries, payments, price list, credit-sync
history, and price-sync history. Invoice tickets use eight lookups; order tickets
omit the invoice lookup and use seven. Job history covers the ticket date and
preceding 14 days. The model cannot request more records or repair its answer.

B1 may return `UNKNOWN` because ticket text alone does not establish the cause.
That is different from a failed API request. Failed evidence or action verification
can also cause B1-DB or the agent to escalate.

For an offline baseline smoke check, use `--provider mock` with B1. Mock responses
are software checks, not model evaluation results. The live agent supports Groq
and Voyager, not the mock provider.

## Reproduce the preliminary evaluation

Generate the separate dataset collection in a new output folder:

```bash
python3 -B -m eval.datasets --out outputs/comparison/datasets
```

It creates 60 development cases, 30 isolated supported test cases, 12 unsupported
cases, and the original 18-ticket reference set. The development cases are excluded
from the following comparison:

```bash
python3 -B -m eval.experiment \
  --manifest outputs/comparison/datasets/manifest.json \
  --out outputs/comparison/run \
  --provider voyager \
  --model qwen38-27b \
  --workers 2 \
  --systems b0 b1 fixed agent
```

This evaluates **60 tickets across four systems: 240 system runs**. It makes live
model requests and takes longer than the single-ticket demo. Results, settings,
and tool traces are saved under `outputs/comparison/run/`. To evaluate only B1-DB,
use `--systems fixed`. Interrupted runs resume only when their settings and
implementation fingerprints match.

The generator uses fixed seeds. A new folder reproduces the same cases; it does
not create a fresh unseen dataset. Both datasets are synthetic and share wording
templates. The original reference tickets were previously used for development.

The recorded comparison and its limitations are in
[the preliminary evaluation report](reports/section3-with-fixed-baseline-2026-11-08/section3-preliminary-results.txt).
These are pilot results, not evidence of production ERP performance.

## Run the tests

```bash
python3 -B -m unittest discover -s tests -v
```

These tests check the database tools, agent behavior, evidence verification, and
evaluation logic without making live model requests.

## Demo screenshots

Use screenshots from an actual run and keep their text readable:

| Screenshot | What to show | Suggested file |
| --- | --- | --- |
| Main demo image | Verified diagnosis, evidence, action, and escalation decision | `images/verified-diagnosis.png` |
| Ticket input | TKT-1011 and invoice reference INV-0342 | `images/ticket-input.png` |
| Investigation trace | Tool names, arguments, returned facts, and lookup count | `images/tool-trace.png` |

After adding the main image, place this below the title:

```html
<p align="center">
  <img src="images/verified-diagnosis.png"
       alt="ERP Ticket Agent showing a verified diagnosis and supporting evidence"
       width="100%">
</p>
```

Crop screenshots to the relevant terminal output. Keep API keys and `.env` contents
out of the images. Label screenshots with the ticket and provider used; keep timing
and token figures tied to that recorded run.

## Repository structure

```text
agents/                 agent loop, model clients, baselines, diagnosis and evidence checks
tools/                  read-only database functions and approved tool dispatcher
erpsim/                 synthetic database schema, fault injection, and ticket generation
eval/                   scoring, dataset generation, and experiment runner
ui/                     readable terminal output
tests/                  offline automated checks
examples/               example ticket text
reports/                saved preliminary results and analysis
run_agent_loop.py       adaptive-agent entry point
run_baseline.py         B0, B1, and B1-DB entry point
.env.example            environment template with empty API keys
```

## Troubleshooting

- **`python: command not found`:** use `python3` as shown above.
- **Missing Voyager key:** check `VOYAGER_API_KEY` in the root `.env`, not
  `.env.example`. Existing shell variables take precedence over `.env`.
- **Cannot open Voyager:** connect to the ASU network or VPN to manage keys.
- **HTTP 401/403:** check the key and your account's model access.
- **HTTP 429 or a timeout:** the service may be busy. The Voyager client retries
  temporary errors up to twice; if it still fails, try later and retain the error
  in evaluation results.
- **HTTPS certificate errors:** activate the environment and install
  `requirements-ui.txt`. Keep certificate verification enabled.
- **No database or ticket:** complete the seed-42 generation step, or check `--db`.
- **Verification failed or escalation:** inspect the saved JSON audit and
  verification errors. A predicted cause alone does not mean the ticket was resolved.

API references: [Voyager access and keys](https://docs.rc.asu.edu/ai/api/) ·
[Chat completions and tool calling](https://docs.rc.asu.edu/ai/api/chat/).

## Project origin and license

This CSE 598 capstone extends
[Sachin's ERP ticket-agent baseline](https://github.com/sachinsss/cse598-erp-ticket-agent).
The project uses synthetic ERP data and retains the repository's MIT license.
See [LICENSE](LICENSE).
