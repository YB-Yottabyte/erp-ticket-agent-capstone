from io import BytesIO
import json
from unittest.mock import patch
import unittest
import urllib.error

from agents.groq_client import GroqToolClient
from agents.llm import LLMError


class GroqClientTests(unittest.TestCase):
    # Build a minimal valid provider response with known token usage.
    def _response(self):
        return BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "{}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                }
            ).encode()
        )

    # Supply definitions and preserve the tool-call protocol in investigation mode.
    def test_tool_payload(self):
        with patch("urllib.request.urlopen", return_value=self._response()) as send:
            result = GroqToolClient(api_key="test-only").chat(
                [{"role": "user", "content": "test"}], tools=[{"type": "function"}]
            )
        request = send.call_args.args[0]
        payload = json.loads(request.data)
        self.assertFalse(payload["parallel_tool_calls"])
        self.assertNotIn("response_format", payload)
        self.assertEqual(result["usage"]["input_tokens"], 2)

    # Request final JSON without supplying tools after the budget is exhausted.
    def test_final_payload(self):
        with patch("urllib.request.urlopen", return_value=self._response()) as send:
            GroqToolClient(api_key="test-only").chat([])
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", payload)

    # Retry rate limits within the configured bound without sleeping in tests.
    def test_rate_limit_retry(self):
        error = urllib.error.HTTPError(
            "https://example.test", 429, "limited", {"Retry-After": "0"}, BytesIO()
        )
        with patch(
            "urllib.request.urlopen", side_effect=[error, self._response()]
        ), patch("time.sleep") as sleep:
            result = GroqToolClient(api_key="test-only").chat([])
        self.assertEqual(result["retries"], 1)
        sleep.assert_called_once_with(0)

    # Stop permanent API errors and avoid reflecting provider payloads or secrets.
    def test_permanent_failure(self):
        error = urllib.error.HTTPError(
            "https://example.test", 401, "private", {}, BytesIO(b"secret")
        )
        with patch("urllib.request.urlopen", side_effect=error), self.assertRaises(
            LLMError
        ) as caught:
            GroqToolClient(api_key="test-only").chat([])
        self.assertNotIn("secret", str(caught.exception))

    # Reject malformed provider responses rather than guessing a usable message.
    def test_invalid_response(self):
        with patch(
            "urllib.request.urlopen", return_value=BytesIO(b"{}")
        ), self.assertRaises(LLMError):
            GroqToolClient(api_key="test-only").chat([])

    # Keep certificate and hostname verification enabled for live HTTPS requests.
    def test_https_verification(self):
        import ssl

        context = GroqToolClient(api_key="test-only")._ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertTrue(context.get_ca_certs())


if __name__ == "__main__":
    unittest.main()
