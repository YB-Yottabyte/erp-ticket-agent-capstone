# Tool definitions and dispatch

This layer describes the eight approved ERP lookups to a model and executes
validated requests. It does not call a model or implement the investigation loop.
The B0/B1 baselines remain separate and unchanged.

```python
from tools.database import ERPDatabase
from tools.dispatcher import ToolDispatcher, ToolRegistry

registry = ToolRegistry(ERPDatabase("data/erp.db"))
dispatcher = ToolDispatcher(registry)
definitions = registry.definitions()  # Supply as tools in a Groq chat request.

# Example of a model response, not a real model invocation.
call = {
    "id": "call_invoice",
    "type": "function",
    "function": {
        "name": "get_invoice",
        "arguments": '{"invoice_id": "INV-0009"}',
    },
}
execution = dispatcher.dispatch_call(call)
print(execution.to_message())  # Append after the assistant tool-call message.
print(execution.to_record())   # Optional record for the research audit log.
```

Only explicit registered methods can execute. Connection helpers, arbitrary SQL,
and `fault_log` have no dispatch entry. The underlying database uses read-only
connections. This is a tool-access boundary, not a sandbox against arbitrary
Python code with direct access to the database file.

The dispatcher validates required fields, rejects extra fields, checks string
types and ID formats, validates nested date fields, and rejects duplicate JSON
keys. Arguments are capped at 8192 encoded bytes. Its validator covers only the
schema features used by this registry, not arbitrary JSON Schema. IDs are strict
and do not accept surrounding whitespace at this boundary. Tool-level checks
also apply, including chronological date order. Job dates follow the simulated
ticket timeline rather than today's date.

Results distinguish `invalid_tool`, `invalid_input`, `not_found`,
`database_error`, and unexpected `tool_error`. Malformed tool-call envelopes
raise `ValueError` because no correctly linked tool-result message can be built.
The future controller must handle this by retrying within its budget or escalating.

Each execution includes its call ID, validated arguments, observation, exception
type if applicable, and local elapsed time. Invalid arguments are not retained.
Timing covers parsing, validation, and local execution, not model latency or API
retries. An `ok` lookup is not a correct diagnosis. Rejected calls must still
count toward the future eight-call budget, which this layer does not enforce.
No agent accuracy or model integration is established by this layer.
The previous automated test suite has been removed; new tests will need to be
added when the team resumes testing.

Protocol reference: [Groq local tool calling](https://console.groq.com/docs/tool-use/local-tool-calling).
