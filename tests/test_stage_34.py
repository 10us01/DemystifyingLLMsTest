"""
Challenge Stage 34 Test (Opaque)
"""
import pytest


def test_stage_34():
    from serving.api.request_handler import RequestHandler

    class MockTokenizer:
        def encode(self, text):
            return [10, 20, 30]

        def decode(self, token_ids):
            return "Decoded completion text"

    class MockEngine:
        def generate_tokens(self, prompt_tokens, **kwargs):
            return [10, 20, 30, 40, 50]

    tokenizer = MockTokenizer()
    engine = MockEngine()
    handler = RequestHandler(engine=engine, tokenizer=tokenizer)

    req = {"prompt": "Sample prompt", "max_tokens": 5}
    response = handler.handle_completion(req)

    assert "text" in response, "Stage 34 criteria not met: response must contain 'text' key"
    text_val = response["text"]

    assert not isinstance(text_val, list), (
        f"Stage 34 criteria not met: response 'text' contains raw token IDs {text_val} "
        f"instead of decoded text"
    )
    assert isinstance(text_val, str), (
        f"Stage 34 criteria not met: expected decoded string in 'text' field, got {type(text_val).__name__}"
    )
    assert text_val == "Decoded completion text", (
        f"Stage 34 criteria not met: expected 'Decoded completion text', got '{text_val}'"
    )
