# Voyager development smoke checks

Checked on November 8, 2026 with Voyager `qwen38-27b`, temperature 0,
reasoning effort `none`, the existing seed-42 synthetic world, and an eight-lookup
limit. These known development cases were used to debug and tune the implementation.
They are not a held-out benchmark or proof that the Milestone 1 targets are met.

The original pasted three-ticket run had 66.7% root-cause accuracy and 0% full
resolution. Failures included text masquerading as a tool call, missing action and
confidence fields, combined evidence fields, and incorrect citation values.

Final development checks after the fixes:

| Ticket | Expected cause | Database lookups | Evidence verification | Full resolution |
| --- | --- | --- | --- | --- |
| TKT-1001 | CUSTOMER_ON_HOLD | 2 | Passed | Yes |
| TKT-1002 | MISSING_TAX_ID | 3 | Passed | Yes |
| TKT-1003 | STALE_PRICE_LIST | 2 | Passed | Yes |
| TKT-1011 | CREDIT_LIMIT_STALE_SYNC | 4 | Passed | Yes |

The first three cases were run together; the credit-sync case was checked
separately with the same final settings. Actual latency and paths can vary.
No corrections were needed on these final live runs. Scripted tests exercise
the correction paths, including failed corrections and safe escalation.

During development, a strict response-schema request failed because the model
advertises `supports_response_schema=false`. A separate final-turn HTTP 400 was
traced to inserting a system message between tool observations. The final design
uses native `submit_diagnosis` arguments plus local validation, and keeps controller
updates in the initial system message. An intermediate price-list run also reached
the budget and correctly escalated; narrower stopping instructions reduced its
final run to two lookups. Failed intermediate runs were not treated as resolved.

The controller never receives the fault-log answer key. Scoring runs separately
after the final result, and full resolution now also requires successful evidence
verification, valid output, and no escalation. Sindhu's existing verifier fixes
are included with their original Git authorship.

Reproduce the sample checks:

```bash
python3 run_agent_loop.py --all --limit 3 --provider voyager --score --out outputs/voyager-three.jsonl
python3 run_agent_loop.py --ticket TKT-1011 --provider voyager --score --out outputs/voyager-credit.jsonl
python3 -B -m unittest discover -s tests
```

Further evaluation must cover all six causes, unsupported tickets, isolated
development/test cases, repeated runs, and same-model baseline comparisons.
