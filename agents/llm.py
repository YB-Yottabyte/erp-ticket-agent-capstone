import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = {
    "gemini": "gemini-flash-latest",
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "mock": "mock-stub-v1",
}


class LLMError(RuntimeError):
    pass


def _load_dotenv():
    for path in (".env", os.path.join(os.path.dirname(__file__), "..", ".env")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


RETRY_CODES = {429, 500, 502, 503, 504}

BACKOFF = [5, 15, 30, 60, 60, 60]


def _post(url, payload, headers, timeout=60, attempts=7):
    data = json.dumps(payload).encode("utf-8")
    last = ""
    for i in range(attempts):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")[:400]
            if e.code not in RETRY_CODES or i == attempts - 1:
                raise LLMError(f"HTTP {e.code}: {body}") from None
            last = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            if i == attempts - 1:
                raise LLMError(f"network error: {e.reason}") from None
            last = f"network error: {e.reason}"
        wait = BACKOFF[min(i, len(BACKOFF) - 1)]
        print(f"  {last} - retrying in {wait}s ({i + 1}/{attempts - 1})")
        time.sleep(wait)
    raise LLMError("exhausted retries")


def complete(prompt, provider="mock", model=None, temperature=0.0):
    _load_dotenv()
    model = (model
             or os.environ.get(f"{provider.upper()}_MODEL")
             or DEFAULT_MODEL.get(provider, ""))

    if provider == "mock":
        return _mock(prompt), {"input_tokens": len(prompt) // 4, "output_tokens": 60}

    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY not set (see .env.example)")
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature,
                                     "responseMimeType": "application/json"}}
        r = _post(url, body, {"Content-Type": "application/json"})
        text = r["candidates"][0]["content"]["parts"][0]["text"]
        u = r.get("usageMetadata", {})
        return text, {"input_tokens": u.get("promptTokenCount", 0),
                      "output_tokens": u.get("candidatesTokenCount", 0)}

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY not set (see .env.example)")
        body = {"model": model, "max_tokens": 1024, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        r = _post("https://api.anthropic.com/v1/messages", body,
                  {"content-type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"})
        text = r["content"][0]["text"]
        u = r.get("usage", {})
        return text, {"input_tokens": u.get("input_tokens", 0),
                      "output_tokens": u.get("output_tokens", 0)}

    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY not set (see .env.example)")
        body = {"model": model, "temperature": temperature,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        r = _post("https://api.openai.com/v1/chat/completions", body,
                  {"Content-Type": "application/json",
                   "Authorization": f"Bearer {key}"})
        text = r["choices"][0]["message"]["content"]
        u = r.get("usage", {})
        return text, {"input_tokens": u.get("prompt_tokens", 0),
                      "output_tokens": u.get("completion_tokens", 0)}

    raise LLMError(f"unknown provider: {provider}")


def _mock(prompt):
    body = prompt.split("TICKET ", 1)[-1].lower()
    if "invoice" in body:
        code, conf = "INCOMPLETE_DELIVERY_BLOCKS_INVOICE", 0.35
    elif "order" in body:
        code, conf = "CUSTOMER_ON_HOLD", 0.33
    else:
        code, conf = "UNKNOWN", 0.20
    return json.dumps({
        "root_cause_code": code,
        "root_cause_explanation": ("Guessed from ticket wording alone; no ERP "
                                   "records were consulted."),
        "evidence": [],
        "proposed_fix": {"action": "ESCALATE", "target": "tier-2 support"},
        "confidence": conf,
        "escalate": True,
    })
