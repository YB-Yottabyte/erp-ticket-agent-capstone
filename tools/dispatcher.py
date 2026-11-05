from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import json
import re
from time import perf_counter
from typing import Any, Callable, Optional

from tools.database import ERPDatabase


class ToolRegistry:
    # Register only the eight approved database tools and their input schemas.
    def __init__(self, database: ERPDatabase):
        self._handlers: dict[str, Callable[..., dict]] = {
            "find_document": database.find_document,
            "get_invoice": database.get_invoice,
            "get_sales_order": database.get_sales_order,
            "get_customer": database.get_customer,
            "get_delivery": database.get_delivery,
            "get_payments": database.get_payments,
            "get_price_list": database.get_price_list,
            "get_job_runs": database.get_job_runs,
        }
        configurations = [
            (
                "find_document",
                "Find candidate invoices or orders by exact document ID, customer ID, or customer name. Multiple matches are not a confirmed diagnosis.",
                "reference",
                None,
            ),
            (
                "get_invoice",
                "Read an invoice and its customer and sales-order references.",
                "invoice_id",
                "INV-",
            ),
            (
                "get_sales_order",
                "Read a sales order, its block reason, and price-list reference.",
                "order_id",
                "SO-",
            ),
            (
                "get_customer",
                "Read customer tax ID, account hold, and credit details.",
                "customer_id",
                "C-",
            ),
            (
                "get_delivery",
                "Read all delivery records for a sales order, including ordered and shipped quantities.",
                "order_id",
                "SO-",
            ),
            (
                "get_payments",
                "Read a customer's payments and whether they were applied to credit usage. Unapplied payments alone do not establish a failed sync.",
                "customer_id",
                "C-",
            ),
            (
                "get_price_list",
                "Read price-list validity dates. Interpret dates relative to the simulated ticket, not the current system date.",
                "price_list_id",
                "PL-",
            ),
        ]
        self._definitions = {}
        for name, description, argument, prefix in configurations:
            field = {"type": "string", "minLength": 1, "maxLength": 200}
            if prefix:
                field.update(
                    pattern=f"^{prefix}[0-9]+$",
                    description=f"Record ID starting with {prefix}.",
                )
            else:
                field.update(
                    minLength=3,
                    description="Document ID, customer ID, or literal customer-name substring; 3-200 characters.",
                )
            self._definitions[name] = self._definition(
                name, description, {argument: field}
            )

        self._definitions["get_job_runs"] = self._definition(
            "get_job_runs",
            "Read job history in an inclusive date range based on the ticket's simulated timeline.",
            {
                "job_name": {
                    "type": "string",
                    "enum": ["credit_sync", "price_sync", "ledger_post"],
                },
                "date_range": {
                    "type": "object",
                    "properties": {
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"},
                    },
                    "required": ["start_date", "end_date"],
                    "additionalProperties": False,
                },
            },
        )

    # Build the function-definition format accepted by Groq's chat API.
    @staticmethod
    def _definition(name, description, properties):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            },
        }

    # Return independent definitions so callers cannot change registry validation.
    def definitions(self):
        return deepcopy(list(self._definitions.values()))

    # Resolve an approved name without exposing arbitrary database attributes.
    def resolve(self, name):
        if not isinstance(name, str) or name not in self._handlers:
            return None
        return self._handlers[name], deepcopy(
            self._definitions[name]["function"]["parameters"]
        )


@dataclass
class ToolExecution:
    tool_call_id: Optional[str]
    tool_name: Any
    arguments: Any
    observation: dict
    elapsed_ms: float
    error_type: Optional[str] = None

    # Convert one execution into a tool-result message linked to its model call.
    def to_message(self):
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
            raise ValueError("A nonempty tool_call_id is required for a tool message")
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": json.dumps(self.observation, allow_nan=False),
        }

    # Return a JSON-serializable execution record for later audit and evaluation.
    def to_record(self):
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": deepcopy(self.arguments),
            "observation": deepcopy(self.observation),
            "elapsed_ms": self.elapsed_ms,
            "error_type": self.error_type,
        }


class ToolDispatcher:
    # Keep the registry as the only route from model requests to database tools.
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    # Reject duplicate argument keys instead of silently accepting the last value.
    @staticmethod
    def _unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate argument keys are not allowed")
            result[key] = value
        return result

    # Validate the object and string schema features used by this tool registry.
    @classmethod
    def _validate(cls, value, schema, path="arguments"):
        if schema["type"] == "object":
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            properties = schema["properties"]
            if any(key not in value for key in schema["required"]):
                raise ValueError(f"{path} is missing required fields")
            if any(key not in properties for key in value):
                raise ValueError(f"{path} contains unexpected fields")
            for key, field in properties.items():
                cls._validate(value[key], field, f"{path}.{key}")
        elif schema["type"] == "string":
            if not isinstance(value, str):
                raise ValueError(f"{path} must be text")
            if (
                not schema.get("minLength", 0)
                <= len(value)
                <= schema.get("maxLength", 200)
            ):
                raise ValueError(f"{path} has an invalid length")
            if "enum" in schema and value not in schema["enum"]:
                raise ValueError(f"{path} is not an approved value")
            if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
                raise ValueError(f"{path} has an invalid ID format")
            if (
                schema.get("format") == "date"
                and date.fromisoformat(value).isoformat() != value
            ):
                raise ValueError(f"{path} must use YYYY-MM-DD")

    # Validate and execute one request while recording its observation and duration.
    def dispatch(self, tool_name, arguments, tool_call_id=None):
        if tool_call_id is not None and (
            not isinstance(tool_call_id, str) or not tool_call_id.strip()
        ):
            raise ValueError("Tool call ID must be nonempty text when supplied")
        started = perf_counter()
        parsed = None
        error_type = None
        resolved = self.registry.resolve(tool_name)
        if resolved is None:
            observation = {"status": "invalid_tool", "message": "Tool is not approved"}
        else:
            handler, schema = resolved
            try:
                if not isinstance(arguments, (str, dict)):
                    raise ValueError(
                        "Arguments must be an object or a JSON object string"
                    )
                encoded = (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, allow_nan=False)
                )
                if len(encoded.encode("utf-8")) > 8192:
                    raise ValueError("Arguments exceed the 8192-byte limit")
                parsed = json.loads(encoded, object_pairs_hook=self._unique_object)
                self._validate(parsed, schema)
            except (ValueError, TypeError, RecursionError, UnicodeError) as error:
                parsed = None
                error_type = type(error).__name__
                observation = {
                    "status": "invalid_input",
                    "message": "Arguments do not match the tool definition",
                }
            else:
                try:
                    observation = handler(**parsed)
                    if not isinstance(observation, dict) or not isinstance(
                        observation.get("status"), str
                    ):
                        raise TypeError("Invalid tool result")
                    json.dumps(observation, allow_nan=False)
                except Exception as error:
                    error_type = type(error).__name__
                    observation = {
                        "status": "tool_error",
                        "message": "Tool execution failed",
                    }

        return ToolExecution(
            tool_call_id,
            tool_name if isinstance(tool_name, str) else None,
            parsed,
            observation,
            round((perf_counter() - started) * 1000, 3),
            error_type,
        )

    # Accept one Groq tool-call object and preserve its call ID in the execution.
    def dispatch_call(self, tool_call):
        if not isinstance(tool_call, dict):
            raise ValueError("Tool call must be an object")
        call_id = tool_call.get("id")
        function = tool_call.get("function")
        if (
            not isinstance(call_id, str)
            or not call_id.strip()
            or tool_call.get("type") != "function"
            or not isinstance(function, dict)
        ):
            raise ValueError(
                "Tool call requires an ID, function type, and function object"
            )
        return self.dispatch(function.get("name"), function.get("arguments"), call_id)
