from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.tools.execution_tools import OpenAIChatClient


class _FakeChatCompletions:
    def __init__(self, response):
        self.response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        return self.response


class _FakeClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(response))


def _build_chat_client(response) -> OpenAIChatClient:
    client = OpenAIChatClient.__new__(OpenAIChatClient)
    client.default_model = "test-model"
    client._client = _FakeClient(response)
    return client


class OpenAIChatClientCompatibilityTests(unittest.TestCase):
    def test_complete_accepts_plain_string_response(self) -> None:
        client = _build_chat_client("plain text response")

        answer = client.complete([{"role": "user", "content": "hello"}], temperature=0.2)

        self.assertEqual(answer, "plain text response")
        self.assertEqual(client._client.chat.completions.last_kwargs["model"], "test-model")

    def test_complete_extracts_text_from_choices_payload(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "structured"},
                            {"type": "text", "text": "response"},
                        ]
                    }
                }
            ]
        }
        client = _build_chat_client(response)

        answer = client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(answer, "structured\nresponse")

    def test_complete_reassembles_sse_event_stream_text(self) -> None:
        response = "\n".join(
            [
                'event: message',
                'data: {"choices":[{"delta":{"role":"assistant"}}]}',
                'data: {"choices":[{"delta":{"content":"{\\"department\\":"}}]}',
                'data: {"choices":[{"delta":{"content":"\\"神经科\\"}"}}]}',
                'data: [DONE]',
            ]
        )
        client = _build_chat_client(response)

        answer = client.complete([{"role": "user", "content": "hello"}])

        self.assertEqual(answer, '{"department":"神经科"}')


if __name__ == "__main__":
    unittest.main()
