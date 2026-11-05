from copy import deepcopy
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.request

from agents.llm import LLMError, _load_dotenv


class GroqToolClient:
    # Configure a bounded Groq client without changing the text-only baselines.
    def __init__(self, model=None, api_key=None, timeout=45, max_retries=2):
        _load_dotenv()
        self.model = model or os.environ.get("GROQ_MODEL") or "openai/gpt-oss-20b"
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not configured")
        if (
            not math.isfinite(timeout)
            or timeout <= 0
            or type(max_retries) is not int
            or not 0 <= max_retries <= 2
        ):
            raise ValueError("Use a positive timeout and zero to two retries")
        self.timeout = timeout
        self.max_retries = max_retries
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    # Keep HTTPS verified and use Certifi only when Python has no trusted CA roots.
    def _ssl_context(self):
        context = ssl.create_default_context()
        if not context.get_ca_certs():
            try:
                import certifi
            except ImportError:
                raise LLMError(
                    "Configure Python CA certificates or install certifi"
                ) from None
            context = ssl.create_default_context(cafile=certifi.where())
        return context

    # Send tool definitions during investigation or request JSON for the final reply.
    def chat(self, messages, tools=None):
        payload = {
            "model": self.model,
            "messages": deepcopy(messages),
            "temperature": 0,
            "max_completion_tokens": 2048,
        }
        if tools:
            payload.update(tools=tools, tool_choice="auto", parallel_tool_calls=False)
        else:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, allow_nan=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "ERP-Ticket-Agent/1.0",
            },
            method="POST",
        )
        context = self._ssl_context()
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=context
                ) as response:
                    body = json.loads(response.read().decode("utf-8"))
                if not isinstance(body, dict) or not isinstance(
                    body.get("choices"), list
                ):
                    raise ValueError("Invalid response envelope")
                choice = body["choices"][0]
                if not isinstance(choice, dict):
                    raise ValueError("Invalid response choice")
                message = choice["message"]
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ValueError("Invalid assistant message")
                usage = body.get("usage") or {}
                if not isinstance(usage, dict):
                    raise ValueError("Invalid usage object")
                return {
                    "message": message,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                    },
                    "retries": attempt,
                    "model": body.get("model", self.model),
                }
            except urllib.error.HTTPError as error:
                if (
                    error.code not in {429, 500, 502, 503, 504}
                    or attempt == self.max_retries
                ):
                    error.close()
                    raise LLMError(
                        f"Groq request failed with HTTP {error.code}"
                    ) from None
                try:
                    delay = float(error.headers.get("Retry-After", 2**attempt))
                    delay = (
                        min(30, max(0, delay)) if math.isfinite(delay) else 2**attempt
                    )
                except (TypeError, ValueError):
                    delay = 2**attempt
                error.close()
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.max_retries:
                    raise LLMError("Groq connection failed or timed out") from None
                time.sleep(2**attempt)
            except (ValueError, KeyError, IndexError, TypeError):
                raise LLMError("Groq returned an invalid response") from None
