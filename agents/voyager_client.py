from copy import deepcopy
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.request

from agents.llm import LLMError, _load_dotenv


class VoyagerToolClient:
    # Configure Voyager independently from Groq, using the same chat interface.
    def __init__(self, model=None, api_key=None, timeout=120, max_retries=2):
        _load_dotenv()
        self.model = model or os.environ.get("VOYAGER_MODEL") or "qwen38-27b"
        self.api_key = api_key or os.environ.get("VOYAGER_API_KEY")
        if not self.api_key:
            raise LLMError("VOYAGER_API_KEY is not configured (see .env.example)")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
            or type(max_retries) is not int
            or not 0 <= max_retries <= 2
        ):
            raise ValueError("Use a positive timeout and zero to two retries")
        self.reasoning_effort = os.environ.get("VOYAGER_REASONING_EFFORT") or "none"
        if self.reasoning_effort not in {"none", "low", "medium", "high"}:
            raise LLMError(
                "VOYAGER_REASONING_EFFORT must be none, low, medium, or high"
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.url = "https://openai.rc.asu.edu/v1/chat/completions"

    # Keep HTTPS verified and fall back to Certifi when Python has no CA roots.
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

    # Read one assistant reply and token usage without replacing missing answers.
    def _parse_response(self, body, attempt):
        choice = body["choices"][0]
        message = choice["message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise ValueError("Invalid assistant message")
        if not message.get("tool_calls") and (
            not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("Missing answer or tool calls")
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

    # Send approved tool definitions or request a final JSON answer from Voyager.
    def chat(self, messages, tools=None, temperature=0.0):
        payload = {
            "model": self.model,
            "messages": deepcopy(messages),
            "temperature": temperature,
            "max_tokens": 4096,
            "reasoning_effort": self.reasoning_effort,
        }
        if tools:
            payload["tools"] = deepcopy(tools)
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
                return self._parse_response(body, attempt)
            except urllib.error.HTTPError as error:
                if (
                    error.code not in {429, 500, 502, 503, 504}
                    or attempt == self.max_retries
                ):
                    error.close()
                    raise LLMError(
                        f"Voyager request failed with HTTP {error.code}"
                    ) from None
                try:
                    delay = float(error.headers.get("Retry-After", 2**attempt))
                    delay = (
                        min(30, max(0, delay)) if math.isfinite(delay) else 2**attempt
                    )
                except (TypeError, ValueError, AttributeError):
                    delay = 2**attempt
                error.close()
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.max_retries:
                    raise LLMError("Voyager connection failed or timed out") from None
                time.sleep(2**attempt)
            except (ValueError, KeyError, IndexError, TypeError):
                raise LLMError(
                    "Voyager returned an invalid or empty response"
                ) from None
